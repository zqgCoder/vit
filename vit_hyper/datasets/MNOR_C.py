# -*- coding: utf-8 -*-

"""
@File    : MNOR_C.py
@Description: NUAA,MSU, OULU, REPLAY ---> CASIA
        train:val == 1:4, 将4个数据库打散，按照1：4的比例分配。
@Author  : zqgCcoder
@Time    : 2023/2/1 14:09
"""
import os
import random


def read_split_data(root: str, val_rate: float = 0.8):
    # db_list = ['CASIA', 'MSU', 'NUAA','OULU', 'REPLAY']
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    images_list_dict = []
    db_list = ['MSU', 'NUAA', 'OULU', 'REPLAY']
    for db in db_list:
        mos_path = '{}/{}/mos.txt'.format(root, db)

        with open(mos_path, 'r') as txt_file:
            for line in txt_file.readlines():
                images_list_dict.append({'db': db, 'img': line.split('\n')[0]})

    random.shuffle(images_list_dict)
    val_path = random.sample(images_list_dict, k=int(len(images_list_dict) * val_rate))

    for img_path in images_list_dict:
        db = img_path['db']
        label_img = img_path['img']
        label, img_path2 = int(label_img.split(' ')[0]), '{}/{}/alls/{}'.format(root, db, label_img.split(' ')[1])
        if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            val_images_path.append(img_path2)
            val_images_label.append(label)
        else:  # 否则存入训练集
            train_images_path.append(img_path2)
            train_images_label.append(label)

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label


# read_split_data('F:/db_tf')
