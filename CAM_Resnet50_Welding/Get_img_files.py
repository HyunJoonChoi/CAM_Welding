import os
import pdb
import cv2

def list_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith('.' + extension)]

# if __name__ == '__main__':
#     date = "0828"
#     directory = "/media/choihy/새 볼륨/datasets/Welding_original/train/OK/" + date + "/OK"
#     dir_name_list = os.listdir(directory)
#     full_dir_path = [os.path.join(directory, dir_name) for dir_name in dir_name_list]
#
#     save_directory = "/media/choihy/새 볼륨/datasets/Welding/Train/OK"
#     i = 0
#     for dir_path in full_dir_path:
#         img_name_list = os.listdir(dir_path)
#         full_img_path_list = [os.path.join(dir_path, img_name) for img_name in img_name_list]
#         for full_img_path in full_img_path_list:
#             img_array = cv2.imread(full_img_path)
#             cv2.imwrite(os.path.join(save_directory, "0828_OK_%04d.jpg"%(i)), img_array)
#             i += 1


if __name__ == '__main__':
    date = "0828"
    part = "크레이터"
    directory = "/home/choihy/Pycharm_project/CAM/Weakly_Supervised_Localization_Welding/Dataset/NG/" + part
    # dir_name_list = os.listdir(directory)
    # full_dir_path = [os.path.join(directory, dir_name) for dir_name in dir_name_list]

    full_dir_path = [directory]

    save_directory = "/media/choihy/새 볼륨/datasets/Welding_ver2/NG/"

    i = 0
    for dir_path in full_dir_path:
        img_name_list = os.listdir(dir_path)
        full_img_path_list = [os.path.join(dir_path, img_name) for img_name in img_name_list]
        for full_img_path in full_img_path_list:
            img_array = cv2.imread(full_img_path)
            cv2.imwrite(os.path.join(save_directory, date + "_NG_" + part + "%05d.jpg"%(i)), img_array)
            i += 1