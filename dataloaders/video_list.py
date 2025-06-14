import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import time
from glob import glob
import os.path as osp
from natsort import natsorted
import pdb
from dataset_path import Path

# several data augumentation strategies
def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, label

def randomCrop(imgs, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    return imgs, label.crop(random_region)

def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return imgs, label

def colorEnhance(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255       
    return Image.fromarray(img)

class VideoDataset(data.Dataset):
    def __init__(self, dataset, trainsize, split):
        self.trainsize = trainsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []
        self.split = split  
        
        root = Path.db_root_dir(dataset)
        data_root = osp.join(root, split)
        videos = natsorted(os.listdir(data_root))
        
        for scene in videos:
            image_names = natsorted(os.listdir(osp.join(data_root, scene, 'images')))
            images = []
            gt_list= []
            for image_name in image_names:
                if dataset != 'IGS-Few':
                    gt_name = image_name.replace('jpg', 'png')
                else: 
                    gt_name = image_name
                images  += [osp.join(data_root, scene, 'images', image_name)]
                gt_list += [osp.join(data_root, scene, 'masks', gt_name)]
            
            for i in range(len(images)-2):
                self.extra_info += [(scene, i)]  # scene and frame_id
                self.gt_list    += [gt_list[i] ]
                self.image_list += [[images[i],
                                     images[i+1], 
                                     images[i+2]]]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform_train = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform_val = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        names= []
        index = index % len(self.image_list)

        for i in range(len(self.image_list[index])):
            imgs += [self.rgb_loader(self.image_list[index][i])]
            names+= [self.image_list[index][i].split('/')[-1]]
        scene= self.image_list[index][0].split('/')[-3]  
        gt = self.binary_loader(self.gt_list[index])
        image_path = self.image_list[index]

        if self.split == 'train':
            imgs, gt = cv_random_flip(imgs, gt)
            imgs, gt = randomRotation(imgs, gt)
            imgs = colorEnhance(imgs)
            gt = randomPeper(gt)
            gt = self.gt_transform_train(gt)
        else:
            gt = self.gt_transform_val(gt)

        for i in range(len(imgs)):
            imgs[i] = self.img_transform(imgs[i])

        return imgs, gt, scene, names, image_path

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 

    def __len__(self):
        return len(self.image_list)

# dataloader for training
def get_loader(dataset, batchsize, size, split, shuffle, num_workers=32, pin_memory=True):
    dataset = VideoDataset(dataset, size, split=split)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader