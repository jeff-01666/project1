import torch
import numpy as np
#import cv2
from torch.utils.data import Dataset
from .processing import ImgProcessing

class BreastDataset(Dataset):
    def __init__(self,file_names,method='train'):
        self.file_names = file_names
        self.method = method

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self,idx):
        '''
        训练集和标签图像命名方式要一样
        '''
        img_name = self.file_names[idx]
        img_c = ImgProcessing(img_name)
        img_c.inten_normal()
        img_c.z_score()
        img = img_c.get_array()
        #img = (img - np.mean(img,dtype=float))/np.std(img,dtype=float) #标准化
        #print(img.shape)
        
        if self.method == 'train':
            mask = load_mask(img_name)
            return torch.from_numpy(np.expand_dims(img,0)).float(),torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return (torch.from_numpy(np.expand_dims(img,0)).float(), img_name)



def load_img(path):
    img = ImgProcessing(path)#读入图片
    img.inten_normal()
    img.z_score()
    return img.get_array()

def load_mask(path):
    mask = ImgProcessing(path.replace('images','masks'))
    mask_array = mask.get_array()
    return (mask_array / 255).astype(np.uint8)   # 0 1