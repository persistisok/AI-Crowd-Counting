import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
from model import CSRNet
from model import CANNet
from torchvision import transforms
from torch.autograd import Variable

test_path = "./dataset/test/"
rgb_paths = [f"{test_path}rgb/{i}.jpg" for i in range(1, 1001)]
tir_paths = [f"{test_path}tir/{i}R.jpg" for i in range(1, 1001)]

model = CANNet(load_weights=True)
model = model.cuda()
checkpoint = torch.load('./model/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.449], std=[
        0.229, 0.224, 0.225, 0.226]),
])

for i, (rgb_path, tir_path) in enumerate(zip(rgb_paths, tir_paths)):
    rgb_img = Image.open(rgb_path).convert('RGB')
    tir_img = Image.open(tir_path).convert('L')
    mix_img = Image.new("RGBA", rgb_img.size)
    mix_img.paste(rgb_img, (0, 0))
    mix_img.putalpha(tir_img)
    img = transform(mix_img)
    img = img.cuda()
    img = Variable(img)
    output = model(img.unsqueeze(0))
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")
