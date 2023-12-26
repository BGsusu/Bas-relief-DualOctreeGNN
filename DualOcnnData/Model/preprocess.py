import numpy as np
import os
import trimesh


def get_obj_files(folder_path):
    obj_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".obj"):
            file_path = os.path.join(folder_path, filename)
            obj_files.append((filename, file_path))
    return obj_files

def get_obj_files_recursive(folder_path):
    obj_files = []

    # 遍历当前文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 检查文件扩展名是否为.obj
            if filename.endswith(".obj"):
                file_path = os.path.join(root, filename)
                obj_files.append(file_path)

    return obj_files


# 相关文件目录
obj_folder_path = "../../OriginalData"
view_folder_path = "../ViewInfo"


def main():
    # 主要的执行逻辑写在这里
    
    # 首先获取所有需要操作的模型文件
    obj_files = get_obj_files_recursive(obj_folder_path)
    # print(obj_files)
    
    # 对每个模型进行操作
    for file in obj_files:
        
        # --- 归一化包围盒 --- #
        # 读取obj模型
        mesh = trimesh.load(file)
        # 获取模型的顶点坐标
        vertices = mesh.vertices
        # 计算包围盒
        aabb = mesh.bounding_box
        center = aabb.centroid
        # 将模型和包围盒移动到世界空间原点
        mesh.apply_translation(-center)
        
        # --- 根据视角旋转模型 --- #
        # 读取模型对应的模型名称
        obj_file = os.path.basename(file)
        obj_name, obj_extension = os.path.splitext(obj_file)
        
        class_name = ""
        if "Animation" in file:
            class_name = "Animation-Relief"
        elif "Character" in file:
            class_name = "Character-Relief"
        elif "source" in file:
            class_name = "Relief"
        else:
            class_name = None      
        # 获取视角文件
        view_file_path = "{}/{}/{}/out.txt".format(view_folder_path,class_name,obj_name)
        with open(view_file_path,'r') as view_file:
            lines = view_file.readlines()

        for index, line in enumerate(lines):
            line = line.strip()
            if "Focus" in line:
                nextline = lines[index+1].strip()
                focus_list = nextline.split()
                focus = [float(coord) for coord in focus_list]
                focus = np.array(focus)
            elif ".obj" in line:
                nextline = lines[index+1]
                pos_list =  (nextline.strip())[13:].split()
                pos = [float(coord) for coord in pos_list]
                pos = np.array(pos)
                up = np.array([0.0, 1.0, 0.0])
                # 计算view matrix
                forward = focus-pos
                vec_norm = np.linalg.norm(forward)
                forward = forward/vec_norm 
                right = np.cross(forward,up)
                vec_norm = np.linalg.norm(right)
                right = right/vec_norm
                # right = np.cross(forward,up)
                up = np.cross(right, forward)
                # up = np.cross(right, forward)
                
                view_matrix = np.eye(4)
                view_matrix[0, :3] = right
                view_matrix[1, :3] = up
                view_matrix[2, :3] = -forward
                view_matrix[0, 3] = -np.dot(right, pos)
                view_matrix[1,3] = -np.dot(up, pos)
                view_matrix[2,3] = np.dot(forward, pos)
                # 应用旋转
                homogeneous_coordinates = np.column_stack((vertices, np.ones(len(vertices))))
                rotated_vertices = np.dot(view_matrix,homogeneous_coordinates.T).T[:,:3]
                
                #旋转后的模型包围盒放到世界原点
                rotated_mesh = trimesh.Trimesh(vertices=rotated_vertices, faces=mesh.faces)
                # 计算包围盒
                aabb = rotated_mesh.bounding_box
                center = aabb.centroid
                # 将模型和包围盒移动到世界空间原点
                rotated_mesh.apply_translation(-center)
                
                
                
                # 保存处理好的模型为新的OBJ文件
                output_file_path = file.replace("OriginalData","DualOcnnData/Model")
                output_folder_path = os.path.dirname(output_file_path)
                output_folder_path += "/{}".format(obj_name)
                output_file_path = output_folder_path+"/{}.obj".format(line[10:13])
                # 判断文件夹是否存在，不存在则创建
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                rotated_mesh.export(output_file_path, file_type="obj")
                


        
        
        break
        
    

if __name__ == "__main__":
    # 只有在作为主程序运行时才会调用main函数
    main()