import numpy as np
import os
import shutil

def get_view_files(folder_path):
    view_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith("out.txt"):
            file_path = os.path.join(folder_path, filename)
            view_files.append((filename, file_path))
    return view_files

def get_view_files_recursive(folder_path):
    view_files = []

    # 遍历当前文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith("out.txt"):
                file_path = os.path.join(root, filename)
                view_files.append(file_path)

    return view_files

def copy_view_files(view_files1, view_files2):
    for file in view_files1:
        output_file_path = file.replace("data/part1/","AllData/DualOcnnData/ViewInfo/")
        output_folder_path = os.path.dirname(output_file_path)
        # 判断文件夹是否存在，不存在则创建
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        shutil.copy2(file, output_file_path)
        
    for file in view_files2:
        output_file_path = file.replace("data/part2/","AllData/DualOcnnData/ViewInfo/")
        output_folder_path = os.path.dirname(output_file_path)
        # 判断文件夹是否存在，不存在则创建
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        shutil.copy2(file, output_folder_path)

def collection_view_files():
    view_folder_path1 = "/home/daipinxuan/bas_relief/data/part1/"
    view_folder_path2 = "/home/daipinxuan/bas_relief/data/part2/"
    
    view_files1 = get_view_files_recursive(view_folder_path1)
    view_files2 = get_view_files_recursive(view_folder_path2)
    
    
    copy_view_files(view_files1, view_files2)

def main():
    collection_view_files()
        
    

if __name__ == "__main__":
    # 只有在作为主程序运行时才会调用main函数
    main()