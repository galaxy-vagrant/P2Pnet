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