import os
import shutil
import json
from xml.etree import ElementTree as ET
import random

def convert_xml_to_json(xml_file, json_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {}
    points = []

    for obj in root.findall('.//object'):
        for point in obj.findall('.//point'):
            x = float(point.find('x').text)
            y = float(point.find('y').text)
            points.append([x, y])

    data['points'] = points
    data['count'] = len(points)

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)



def process_dataset(input_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理 train 和 test 文件夹
    for phase in ['train', 'test']:
        input_phase_dir = os.path.join(input_dir, phase)
        output_phase_dir = os.path.join(output_dir, phase)

        # 创建输出阶段目录
        if not os.path.exists(output_phase_dir):
            os.makedirs(output_phase_dir)

        # 遍历 rgb 和 tir 文件夹
        for folder in ['rgb', 'tir']:
            input_folder_dir = os.path.join(input_phase_dir, folder)

            # 复制文件并重命名
            for file in os.listdir(input_folder_dir):
                input_file_path = os.path.join(input_folder_dir, file)
                if folder == 'rgb':
                    output_file_path = os.path.join(output_phase_dir, file.replace('.jpg', '_RGB.jpg'))
                else:
                    output_file_path = os.path.join(output_phase_dir, file.replace('R.jpg', '_T.jpg'))

                shutil.copyfile(input_file_path, output_file_path)

        # 处理 labels 文件夹
        if phase == 'train':
            input_labels_dir = os.path.join(input_phase_dir, 'labels')

            # 转换 XML 文件为 JSON 文件并重命名
            for file in os.listdir(input_labels_dir):
                input_file_path = os.path.join(input_labels_dir, file)
                output_file_path = os.path.join(output_phase_dir, file.replace('R.xml', '_GT.json'))

                convert_xml_to_json(input_file_path, output_file_path)

def split_train_val(output_dir, val_ratio=0.2):
    val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    train_dir = os.path.join(output_dir, 'train')
    
    json_files = []
    for file in os.listdir(train_dir):
        if file.endswith('.json'):
            json_files.append(os.path.join(train_dir, file))
    random.shuffle(json_files)

    # 计算需要移动的文件数
    num_val_files = int(len(json_files) * val_ratio)

    # 移动文件到 val 文件夹下
    for file in json_files[:num_val_files]:
        input_file_path_rgb = file.replace('_GT.json', '_RGB.jpg')
        input_file_path_t = file.replace('_GT.json', '_T.jpg')
        input_file_path_gt = file
        output_file_path = file.replace('train', 'val')
        shutil.move(input_file_path_rgb, output_file_path.replace('_GT.json', '_RGB.jpg'))
        shutil.move(input_file_path_t, output_file_path.replace('_GT.json', '_T.jpg'))
        shutil.move(input_file_path_gt, output_file_path.replace('_GT.json', '_GT.json'))

if __name__ == "__main__":
    input_dataset_dir = './dataset'
    output_processed_dir = './prepreprocessed-dataset'

    process_dataset(input_dataset_dir, output_processed_dir)
    split_train_val(output_processed_dir)
    print("数据集处理完成！")
