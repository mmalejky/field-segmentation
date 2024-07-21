import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import albumentations as A

DATASET_NAMES = [
    'BIPED',
    'BSDS',
    'BSDS-RIND',
    'BSDS300',
    'CID',
    'DCD',
    'MDBD', #5
    'PASCAL',
    'NYUD',
    'CLASSIC'
]  # 8


def dataset_info(dataset_name, is_linux=True):
    if is_linux:

        config = {
            'BSDS': {
                'img_height': 512, #321
                'img_width': 512, #481
                'train_list': 'train_pair.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                'yita': 0.5
            },
            'BSDS-RIND': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair2.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS-RIND',  # mean_rgb
                'yita': 0.5
            },
            'BSDS300': {
                'img_height': 512, #321
                'img_width': 512, #481
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/BSDS300',  # NIR
                'yita': 0.5
            },
            'PASCAL': {
                'img_height': 416, # 375
                'img_width': 512, #500
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                'yita': 0.3
            },
            'CID': {
                'img_height': 512,
                'img_width': 512,
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/CID',  # mean_rgb
                'yita': 0.3
            },
            'NYUD': {
                'img_height': 448,#425
                'img_width': 560,#560
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                'yita': 0.5
            },
            'MDBD': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_pair.lst',
                'train_list': 'train_pair.lst',
                'data_dir': '/opt/dataset/MDBD',  # mean_rgb
                'yita': 0.3
            },
            'BIPED': {
                'img_height': 720, #720 # 1088
                'img_width': 1280, # 1280 5 1920
                'test_list': 'test_pair.lst',
                'train_list': 'train_rgb.lst',
                'data_dir': '/opt/dataset/BIPED',  # mean_rgb
                'yita': 0.5
            },
            'CLASSIC': {
                'img_height': 512,
                'img_width': 512,
                'test_list': None,
                'train_list': None,
                'data_dir': 'data',  # mean_rgb
                'yita': 0.5
            },
            'DCD': {
                'img_height': 352, #240
                'img_width': 480,# 360
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2
            }
        }
    else:
        config = {
            'BSDS': {'img_height': 512,  # 321
                     'img_width': 512,  # 481
                     'test_list': 'test_pair.lst',
                     'train_list': 'train_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': 'C:/Users/xavysp/dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': 'C:/Users/xavysp/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/CID',  # mean_rgb
                    'yita': 0.3},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/NYUD',  # mean_rgb
                     'yita': 0.5},
            'MDBD': {'img_height': 720,
                         'img_width': 1280,
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_pair.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/MDBD',  # mean_rgb
                         'yita': 0.3},
            'BIPED': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5},
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data',  # mean_rgb
                        'yita': 0.5},
            'DCD': {'img_height': 240,
                    'img_width': 360,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/DCD',  # mean_rgb
                    'yita': 0.2}
        }
    return config[dataset_name]

# This dataset is modified TestDataset for handling one image
class OneImageDataset(Dataset):
    def __init__(self,
                 image,
                 mean_bgr,
                 img_height,
                 img_width
                 ):

        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()
        self.image = image
        #print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        labels_path = None
        sample_indices = [0]
        
        return sample_indices

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # get data sample
        image = self.image
        label = None
        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names="", image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        img_height = self.img_height
        img_width = self.img_width
        #print(f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
        # img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.resize(img, (img_width,img_height))
        gt = None

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        gt = np.zeros((img.shape[:2]))
        gt = torch.from_numpy(np.array([gt])).float()

        return img, gt

class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):

        # path to folder with images
        self.data_root = data_root

        self.test_list = test_list
        self.args=arg

        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        #print(f"mean_bgr: {self.mean_bgr}")
        print(data_root)

    def _build_index(self):
        sample_indices = []
        # for single image testing
        images_path = os.listdir(self.data_root)
        labels_path = None
        sample_indices = [images_path, labels_path]
        
        return sample_indices

    def __len__(self):
        return len(self.data_index[0])

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx]
        else:
            image_path = self.data_index[idx][0]
        label_path = None
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        img_dir = self.data_root
        gt_dir = None


        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        img_height = self.img_height
        img_width = self.img_width
        #print(f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
        # img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.resize(img, (img_width,img_height))
        gt = None

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        gt = np.zeros((img.shape[:2]))
        gt = torch.from_numpy(np.array([gt])).float()

        return img, gt




class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 ):
        self.data_root = data_root
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img

        self.data_index = self._build_index()

    def _build_index(self):

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        
        images_path = os.path.join(data_root,
                                    'imgs')

        labels_path = os.path.join(data_root,
                                    'edge_maps')
        for file_name_ext in os.listdir(images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(images_path, file_name + '.png'),
                    os.path.join(labels_path, file_name + '.png'),)
            )

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        #image = Image.open(image_path).convert("RGB")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)


    def transform(self, img, gt):
        # data augmentation

        transforms_size_and_orientation = [A.RandomCrop(height=70, width=70, p=1), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
                 # A.RandomRotate90(p=1),
                  A.RandomRotate90(p=0.5)]
        transforms_color = [A.GaussNoise(p=0.5), A.ColorJitter(p=0.5)]

        transforms = transforms_size_and_orientation + transforms_color
        
        for t in transforms:
            augmented = t(image=img, mask=gt)
            img = augmented['image']
            gt = augmented['mask']

        #cv2.imshow("ag", img)
        #cv2.imshow("ed", gt)
        #cv2.waitKey(0)
        # below is default transorm to tensor

        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w,_ = img.shape

        crop_size = self.img_height if self.img_height == self.img_width else None#448# MDBD=480 BIPED=480/400 BSDS=352

        # New addidings
        if crop_size != None:
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        gt[gt > 0.2] += 0.6# 0.5 for BIPED/BSDS-RIND
        gt = np.clip(gt, 0., 1.) # BIPED/BSDS-RIND
        # # for MDBD
        # gt[gt > 0.1] +=0.7
        # gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt
