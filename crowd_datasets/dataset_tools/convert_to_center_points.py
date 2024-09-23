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