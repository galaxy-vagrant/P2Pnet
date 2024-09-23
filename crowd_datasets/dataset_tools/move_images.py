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
    destination_folder = r"crowd_datasets\SHHA\val_real_test"    
    file_list_path = r'crowd_datasets\SHHA\train_val_test\test.txt'    # floder train  <--- train.txt  |  floder test <--- val.txt 
    move_images(source_folder, destination_folder, file_list_path)