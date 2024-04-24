import shutil
import os
from PIL import Image
png_path = '/home/xiancong/Data_set/GTOT_jpg' # png格式图片所在文件夹的路径
#jpg_path = '/home/xiancong/Data_set/GTOT_jpg'  # jpg格式图片存放文件夹的路径
file_walk = os.listdir(png_path)
# 获取指定目录下的所有png图片
def get_all_png_files(dir):
    files_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                files_list.append(os.path.join(root, file))
    return files_list
    # 批量转换png图片为jpg格式
def png2jpg(files_list,file_name):
    for file in files_list:
        img = Image.open(file)
        new_file =os.path.splitext(file)[0] + '.jpg'
        img.convert('RGB').save(new_file)


if __name__ == '__main__':
  for file_name in file_walk:
    target_path_i = os.path.join(png_path, file_name, 'i')
    target_path_v = os.path.join(png_path, file_name, 'v')
    files_list_i = get_all_png_files(target_path_i)
    files_list_v=get_all_png_files(target_path_v )
    png2jpg(files_list_i,file_name)
    png2jpg(files_list_v,file_name)
