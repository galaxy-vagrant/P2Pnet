p2p train install 

## Environment configuration.

[pytorch official](https://pytorch.org/)

```bash
conda create -n p2pnet python=3.7 -y
conda activate p2pnet
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

结果我的python解释器不支持3.8以下的

```bash
conda create -n p2pnet38 python=3.8 -y
conda activate p2pnet38
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Dataset creation.

Dataset creation needs to be combined with the README.md.

我是在xml文件中直接做成点的信息的，直接转换成 x y format的文件

```python
import xml.etree.ElementTree as ET
import os
# Convert an XML file of a document into a format with x and y coordinates.

def convert_to_center_points(xml_file_path, txt_file_path):
    xml_file_path.endswith('.xml')
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    with open(txt_file_path, 'w') as txt_file:
        for obj in root.findall('object'):
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            txt_file.write(f"{x} {y}\n")

if __name__ == "__main__":
    xml_file_floder = r'G:\CASIA_Windows\Code_Repository\yolox-pytorch-main\yolox-pytorch-main\VOCdevkit\VOC2007\Annotations'  
    txt_file_floder = r'G:\CASIA_Windows\Code_Repository\yolox-pytorch-main\yolox-pytorch-main\VOCdevkit\VOC2007\txt' 
    xml_file_list = os.listdir(xml_file_floder)
    sample_num = len(xml_file_list)
    print("len(xml_file_list) : ",sample_num)
    for xml_file in  xml_file_list:
        xml_file_path = os.path.join(xml_file_floder,xml_file)
        xml_name=os.path.basename(xml_file_path).split(".")[0]
        # print("xml_name : ",xml_name)
        txt_file_path = os.path.join(txt_file_floder,xml_name+".txt")
        #print("txt_file_path : ",txt_file_path)
        convert_to_center_points(xml_file_path, txt_file_path)
```

检测一下转化的有没有问题 可视化

```python
from PIL import Image, ImageDraw
import os
# Visualize the point annotation information to ensure the processed data is correct.


def draw_points_on_image(txt_file_path, input_image_path, output_folder_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        points = [tuple(map(float, line.strip().split(' '))) for line in lines]
    try:
            image = Image.open(input_image_path)
    except FileNotFoundError:
            print(f"Image file {input_image_path} not found. Skipping.")
    for point in points:
        x, y = point
        draw = ImageDraw.Draw(image)
        draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')
    image.save(output_folder_path)
    print(f"Point ({x}  {y}) drawn on {input_image_path} and saved to {output_folder_path}")


if __name__ == "__main__":
    txt_file = r'G:\CASIA_Windows\Code_Repository\yolox-pytorch-main\yolox-pytorch-main\VOCdevkit\VOC2007\txt'   
    image_folder = r'G:\CASIA_Windows\Code_Repository\yolox-pytorch-main\yolox-pytorch-main\VOCdevkit\VOC2007\JPEGImages'   
    output_folder = r'G:\CASIA_Windows\Code_Repository\yolox-pytorch-main\yolox-pytorch-main\VOCdevkit\VOC2007\output_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    txt_file_list = os.listdir(txt_file)
    txt_num = len(txt_file)

    for i in txt_file_list:
        name = i.split(".")[0]
        txt_file_path = os.path.join(txt_file,i)
        image_folder_path = os.path.join(image_folder,name+".jpg")
        output_folder_path=os.path.join(output_folder,name+".jpg")
        draw_points_on_image(txt_file_path, image_folder_path, output_folder_path)
```

将已将处理好的 image 和 txt文件移动到对应的位置  分别为train  test文件夹

我事先生成了对应比例的train.txt  val.test  test.txt 所以我直接按照这个比例的名称直接放在对应的文件夹了



```python
import os
import shutil


# Move the processed files to the corresponding folders.
def move_images(source_folder, destination_folder, file_list_path):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    with open(file_list_path, 'r') as file:
        for line in file:
            image_name = line.strip()
            image_path = os.path.join(source_folder, image_name+".txt")
            if os.path.isfile(image_path):
                shutil.move(image_path, destination_folder)
                print(f"Moved: {image_name}")
            else:
                print(f"File not found: {image_name}")

if __name__ == "__main__":
    source_folder = r'G:\CASIA_Windows\Code_Repository\yolox-pytorch-main\yolox-pytorch-main\VOCdevkit\VOC2007\txt'   
    destination_folder = r"crowd_datasets\SHHA\train"    
    file_list_path = r'crowd_datasets\SHHA\train_val_test\train.txt'    # floder train  <--- train.txt  |  floder test <--- val.txt 
    move_images(source_folder, destination_folder, file_list_path)
```

下面就是按照数据集训练对应的格式生成对应的。list文件

[Crowd Counting P2PNet 复现](https://blog.csdn.net/zqq19980906_/article/details/125656654)

```python
def generate_file_paths(input_txt_path, output_txt_path, img_folder, txt_folder):
    """
    生成包含.jpg和.txt文件路径的列表，并将它们写入到一个新的文件中。

    :param input_txt_path: 输入的train.txt文件路径
    :param output_txt_path: 输出文件路径
    :param img_folder: 存放.jpg文件的文件夹路径
    :param txt_folder: 存放.txt文件的文件夹路径
    """
    with open(input_txt_path, 'r') as input_file, open(output_txt_path, 'w') as output_file:
        for line in input_file:
            img_name = line.strip()
            img_path = f"{img_folder}/{img_name}.jpg"
            txt_path = f"{txt_folder}/{img_name}.txt"
            output_file.write(f"{img_path} {txt_path}\n")

if __name__ == "__main__":
    input_txt_path = r"crowd_datasets\SHHA\train_val_test\val.txt"     # floder train  <--- train.txt  |  floder test <--- val.txt 
    output_txt_path = r"crowd_datasets\SHHA\test.txt"  
    img_folder = r'crowd_datasets\SHHA\train' 
    txt_folder = r'crowd_datasets\SHHA\train' 
    generate_file_paths(input_txt_path, output_txt_path, img_folder, txt_folder)
```



## Issues encountered during the training process.

问题1： util\misc.py

cannot import name ‘_new_empty_tensor‘ from ‘torchvision.ops

[Python问题: cannot import name ‘_new_empty_tensor‘ from ‘torchvision.ops](https://blog.csdn.net/a1228136188/article/details/118891791)

```python
'''
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
'''
```

问题2：train.py

注释掉即可 assert args.masks这个参数没找到

```python
  if args.frozen_weights is not None:
    pass
	\# assert args.masks, "Frozen training is meant for segmentation only"
  	\# backup the arguments
```

问题3：models\vgg_.py

修改要用到的路径

```python
model_paths = {
    'vgg16_bn': r"weights\vgg16_bn-6c64b313.pth" , #  '/apdcephfs/private_changanwang/checkpoints/vgg16_bn-6c64b313.pth',
    'vgg16': '/apdcephfs/private_changanwang/checkpoints/vgg16-397923af.pth',

}
```

问题4：crowd_datasets\SHHA\SHHA.py

[Errno 2] No such file or directory: 'crowd_datasets\\SHHA\\shanghai_tech_part_a_train.list'

修改

```python
        self.train_lists = "shanghai_tech_part_a_train.list"
        self.eval_list = "shanghai_tech_part_a_test.list"
```

修改到自己正确的位置

比如我的是

```python
        self.train_lists = r"crowd_datasets\SHHA\train.list"
        self.eval_list = r"crowd_datasets\SHHA\test.list"
```

注意

```python
  parser.add_argument('--data_root', default=r'./',

​            help='path where the dataset is')
```

crowd_datasets\SHHA这个是默认地址 

问题5：train.py

'P2PNet' object has no attribute 'detr'

```python
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
```

改为

```python
model_without_ddp.load_state_dict(checkpoint['model'])
```

问题6：训练过程中

engine.py

Caught error in DataLoader worker process 0.

[迭代DataLoader时出现TypeError: Caught TypeError in DataLoader worker process 0.TypeError: 'NoneType' obj。](https://blog.csdn.net/qinglingLS/article/details/104411589)

train，py

```py
    parser.add_argument('--num_workers', default=8, type=int)
```

改为0

```python
 parser.add_argument('--num_workers', default=0, type=int)
```

问题7：crowd_datasets\SHHA\SHHA.py

```python
def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

```python
OpenCV(4.10.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:196: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
```

注意train.list和test.list里面的路径转义字符

## 开始训练

参数记录

```python
Namespace(backbone='vgg16_bn', batch_size=32, checkpoints_dir='./ckpt', clip_max_norm=0.1, data_root='./', dataset_file='SHHA', eos_coef=0.5, epochs=3500, eval=False, eval_freq=5, frozen_weights='weights\\SHTechA.pth', gpu_id=0, line=2, lr=0.0001, lr_backbone=1e-05, lr_drop=3500, num_workers=0, output_dir='./log', point_loss_coef=0.0002, resume='', row=2, seed=42, set_cost_class=1, set_cost_point=0.05, start_epoch=0, tensorboard_dir='./runs', weight_decay=0.0001)
number of params: 21579344
```

eval可能会有问题  先将频率改为1

## 测试

运行run_test.py文件，但是这是对单个图片进行验证

如果是对文件进行检测 运行run_test_processfor_folder.py文件。

一定要注意,这种插值方法会出问题

```python
        # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        img_raw = img_raw.resize((new_width, new_height))
```

要确保// 128 * 128，确定是整数，

```
new_width = width // 128 * 128
new_height = height // 128 * 128
```

不然的话后面经过处理，会导致维度不匹配

```
        P4_x = self.P4_1(C4)  # P4_x--->torch.Size([1, 256, 189, 240])
        P4_x = P5_upsampled_x + P4_x  # P5_upsampled_x--->torch.Size([1, 256, 188, 240])
```

# 分析

[Errno 13] Permission denied: 'crowd_datasets\\SHHA\\val_real_test'

```
f = open(args.test_text, 'w', encoding='utf-8') 
```

当你打开的是一个文件夹而不是一个文件时经常会遇到这样的情况，确保要打开的是一个文件（txt jpg等）

输出pre_gd_cnt.txt

result_anlysis_tools\calculate_mae.py  计算mae