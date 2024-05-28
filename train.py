import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import h5py
import cv2
import shutil
from model import CSRNet
from model import CANNet
from torchvision.transforms import functional as F


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar', save_dir='./model/'):  # 添加保存目录参数
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(
            save_dir, task_id + 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)


def load_data(rgb_path, tir_path, gt_path, train=True):
    rgb_img = Image.open(rgb_path).convert('RGB')
    tir_img = Image.open(tir_path).convert('L')
    mix_img = Image.new("RGBA", rgb_img.size)
    mix_img.paste(rgb_img, (0, 0))
    mix_img.putalpha(tir_img)
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(
        target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64
    target = torch.tensor(target)
    return mix_img, target


class ImgDataset(Dataset):
    def __init__(self, img_dir, gt_dir, shape=None, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
        self.rgb_dir = os.path.join(img_dir, 'rgb')
        self.tir_dir = os.path.join(img_dir, 'tir')
        self.gt_dir = gt_dir
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_names = list(map(lambda filename: os.path.splitext(filename)[0], os.listdir(self.rgb_dir)))

        if shuffle:
            random.shuffle(self.img_names)

        self.nSamples = len(self.img_names)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name = self.img_names[index]
        rgb_path = os.path.join(self.rgb_dir, img_name + '.jpg')
        tir_path = os.path.join(self.tir_dir, img_name + 'R.jpg')
        gt_path = os.path.join(self.gt_dir, img_name + '.h5')
        img, target = load_data(rgb_path, tir_path, gt_path, self.train)
        assert self.transform is not None
        img = self.transform(img)
        target = target.unsqueeze(0)

        degree = np.random.uniform(-10, 10)
        img = F.rotate(img, degree)
        target = F.rotate(target, degree)

        horizontal_flip = bool(np.random.randint(0, 2))
        if horizontal_flip:
            img = F.hflip(img)
            target = F.hflip(target)

        target = target.mean(0)
        return img, target


lr = 1e-5
original_lr = lr
batch_size = 1
momentum = 0.95
decay = 5*1e-4
epochs = 400
steps = [-1, 1, 100, 150]
scales = [1, 0.5, 0.2, 0.1]
workers = 4
seed = time.time()
print_freq = 30
img_dir = "./dataset/train/"
gt_dir = "./dataset/train/hdf5s/"
pre = "./model/checkpoint.pth.tar"
# pre = None
task = ""


def main():
    start_epoch = 0
    best_prec1 = 1e6

    torch.cuda.manual_seed(seed)

    model = CANNet()

    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=decay)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.449], std=[
                             0.229, 0.224, 0.225, 0.226]),
    ])

    dataset = ImgDataset(
        img_dir,
        gt_dir, transform=transform, train=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    if pre:
        if os.path.isfile(pre):
            print("=> loading checkpoint '{}'".format(pre))
            checkpoint = torch.load(pre)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(pre))

    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        train(model, criterion, optimizer, epoch, train_loader)
        prec1 = validate(model, val_loader)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, task)


def train(model, criterion, optimizer, epoch, train_loader):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), lr))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))


def validate(model, val_loader):
    print('begin test')

    model.eval()
    mae = 0

    for i, (img, target) in enumerate(val_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() -
                   target.sum().type(torch.FloatTensor).cuda())

    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = original_lr

    for i in range(len(steps)):

        scale = scales[i] if i < len(scales) else 1

        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
