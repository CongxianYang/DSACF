# import os
#
# # 定义要遍历的目录
# list = '/home/xiancong/Data_set/GTOT'
# # 定义保存文件名的文件
# filenames= "/home/xiancong/Project_all/SiamCSR-master/data/gtot.txt"
# def creat_gtot(list,filenames):
# # 打开文件以写入文件名
#  with open(filenames, "w") as f:
#     # 遍历目录中的所有文件
#     for filename in os.listdir(list):
#         # 获取文件的完整路径
#         filepath = os.path.join(list, filename)
#         print(filepath)
#         # 判断文件是否是一个普通文件（而不是目录或特殊文件）
#         if os.path.isfile(filepath):
#             # 将文件名写入文件
#             f.write(filename + '\n')
# if __name__ == '__main__':
#     creat_gtot(list,filenames)
import os
img_path = '/root/autodl-tmp/DataSet/LasHeR/trainSet/trainingset'
img_list = os.listdir(img_path)
print('img_list: ', img_list)
with open('trainingsetList_LasHer.txt', 'w') as f:
    for img_name in img_list:
        f.write(img_name + '\n')