# 导入所需模块
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import random


# plt显示灰度图片
def plt_show(img, path):
    plt.imshow(img, cmap='gray')
    plt.savefig(path)
    plt.show()


# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    faces_addr = []
    for filename in os.listdir(directory_name):
        faces_addr.append(directory_name + "/" + filename)
    return faces_addr


def prepare_train_test():
    input_directories = read_directory("./data")
    input_files, input_labels, input_ids = [], [], []
    for input_directory in input_directories:
        files = os.listdir(input_directory)
        input_files += [os.path.join(input_directory, file) for file in files]
        input_labels += [int(input_directory.split('s')[-1])] * len(files)
        input_ids += [ids + 1 for ids in range(len(files))]
    assert len(input_files) == len(input_labels)
    dataset = [(file, label, id) for file, label, id in zip(input_files, input_labels, input_ids)]
    random.shuffle(dataset)
    train_len = round(len(dataset) * 0.8)
    train_dataset = dataset[:train_len]
    test_dataset = dataset[train_len:]
    train_input = np.array([cv2.imread(item[0], cv2.IMREAD_GRAYSCALE).reshape(-1) for item in train_dataset])
    train_labels = np.array([item[1] for item in train_dataset])
    train_ids = np.array([item[2] for item in train_dataset])
    test_input = np.array([cv2.imread(item[0], cv2.IMREAD_GRAYSCALE).reshape(-1) for item in test_dataset])
    test_labels = np.array([item[1] for item in test_dataset])
    test_ids = np.array([item[2] for item in test_dataset])
    return train_input, train_labels, train_ids, test_input, test_labels, test_ids

def save_picture(rebuild_inputs, rebuild_labels, rebuild_ids):
    for input, label, id in zip(rebuild_inputs, rebuild_labels, rebuild_ids):
        directory = "./rebuild_data/s" + str(label)
        file_name = str(id) + ".pgm"
        path = os.path.join(directory, file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(path, input)


def show_rubuild(rebuild_test_input):
    pictures_show = []
    for i in range(6):
        pictures = []
        for j in range(6):
            idx = random.randint(0, len(rebuild_test_input) - 1)
            pictures.append(rebuild_test_input[idx])
        pictures_show.append(np.concatenate(pictures, axis=1))
    new_data = np.concatenate(pictures_show, axis=0)
    plt_show(new_data, "./my_feature.png")


def show_comparison(train_input, test_input, rebuild_train_input, rebuild_test_input):
    comparison_show = []
    for i in range(5):
        pictures = []
        train_idx = random.randint(0, len(rebuild_train_input) - 1)
        test_idx = random.randint(0, len(rebuild_test_input) - 1)
        pictures.append(train_input[train_idx])
        pictures.append(rebuild_train_input[train_idx])
        pictures.append(test_input[test_idx])
        pictures.append(rebuild_test_input[test_idx])
        comparison_show.append(np.concatenate(pictures, axis=1))
    new_data = np.concatenate(comparison_show, axis=0)
    plt_show(new_data, "./my_restore.png")


def curve_show(curve, path):
    plt.plot(curve)
    plt.savefig(path)
    plt.show()


def show_info_curve(pca):
    ratio = pca.explained_variance_ratio_
    info = [0]
    for i in range(150):
        info.append(info[i] + ratio[i])
    curve_show(info, "./my_information.png")


def train_knn(pca, train_input, train_labels, test_input, test_labels):
    new_train_input = pca.transform(train_input)
    new_test_input = pca.transform(test_input)
    knn = KNeighborsClassifier(n_neighbors=3)
    accs = [0]
    for i in range(150):
        cur_train_input = new_train_input[:, : i + 1]
        cur_test_input = new_test_input[:, : i + 1]
        knn = knn.fit(cur_train_input, train_labels)
        test_labels_predict = knn.predict(cur_test_input)
        acc = (test_labels_predict == test_labels).sum() / len(test_labels_predict)
        accs.append(acc)
    curve_show(accs, "./my_acc.png")


if __name__ == '__main__':
    train_input, train_labels, train_ids, test_input, test_labels, test_ids = prepare_train_test()

    # 任务1
    pca = PCA(n_components=100)
    new_train_input = pca.fit_transform(train_input)
    new_test_input = pca.transform(test_input)

    print("任务1：")
    print("压缩后的训练集维度为：")
    print(new_train_input.shape)
    print("压缩后的测试集维度为：")
    print(new_test_input.shape)
    print("经过PCA得到的特征向量维度为：")
    print(pca.components_.shape)
    print("\n")

    # 任务2
    rebuild_train_input = pca.inverse_transform(new_train_input)
    rebuild_train_input = rebuild_train_input.reshape(-1, 112, 92)
    rebuild_test_input = pca.inverse_transform(new_test_input)
    rebuild_test_input = rebuild_test_input.reshape(-1, 112, 92)

    save_picture(rebuild_train_input, train_labels, train_ids)
    save_picture(rebuild_test_input, test_labels, test_ids)
    show_rubuild(rebuild_test_input)
    print("任务2：\n用这些向量重建的特征脸已保存在 ./rebuild_data 目录下。\n并输出部分示例，示例保存在 ./my_feature.png 中。\n")

    # 任务3
    show_comparison(train_input.reshape(-1, 112, 92), test_input.reshape(-1, 112, 92), rebuild_train_input, rebuild_test_input)
    print("任务3：\n对比图已输出，并保存在 ./my_restore.png 中。\n")

    # 任务4
    print("任务4：\n降维后每个新特征向量所占的信息量占原始数据总信息量的百分比如下：")
    print(pca.explained_variance_ratio_)
    print("所有返回特征所携带的信息量总和占原始数据的百分比如下：")
    print(sum(pca.explained_variance_ratio_))
    print("\n")

    # 任务5
    pca = PCA(n_components=150)
    pca.fit_transform(train_input)
    show_info_curve(pca)
    print("任务5：\n特征个数和所携带信息数的曲线图已输出，并保存在 ./my_information.png 中。\n")

    # 任务6
    train_knn(pca, train_input, train_labels, test_input, test_labels)
    print("任务6：\n不同的特征保留数和准确率的曲线图已输出，并保存在 ./my_acc.png 中。\n")

