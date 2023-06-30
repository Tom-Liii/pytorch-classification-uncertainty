import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize
)
from torchvision.transforms import ToTensor
from torchvision import transforms

target_img_size = 256
def image_transform(p=1):
    return Compose([
        Resize(target_img_size, target_img_size, cv2.INTER_LINEAR),
        # Normalize(p=1)
    ], p=p)


def mask_transform(p=1):
    return Compose([
        Resize(target_img_size, target_img_size, cv2.INTER_NEAREST)
    ], p=p)

def get_id(input_img_path):
    data_path = Path(input_img_path) # added
    pred_file_name = []
    pred_file_name.append(data_path)
    return pred_file_name

def get_split():
    # data_path = Path('/research/d5/gds/hzyang22/data/ESD_organized')
    # data_path = Path('/research/d5/gds/hzyang22/data/new_esd_seg') # original
    # data_path = Path('C:/Users/student/Desktop/EDL/new_esd_seg')  
    data_path = Path('C:/Users/tom/Desktop/summer_research/ESD_seg') # added

    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    train_data_file_names = []
    val_data_file_names = []
    test_data_file_names = []
    # data_ids = ['01', '02', '03', '04', '05', '06'
    #             '07', '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '20', '21', '22', '23',
    #             '24', '25', '26', '27', '28', '29', '30', '31',
    #             '32', '33', '34', '35', '36']

    data_ids = ['01', '03', '05', '06',
                '08', '09', '11', '12', '13', '14', '15', '16', '18', '23',
                '24', '25', '26', '27', '28', '29', '30', '31',
                '32', '33', '34', '35', '36']
    # data_ids = ['05_A16756254_clipped']
    random.shuffle(data_ids)
    train_ids =  ['06', '24', '15', '27', '09', '01', '12', '31', '29', '28', '33', '35', '08', '05', '32', '11', '16']
    val_ids = ['30', '14', '36', '25']
    test_ids =  ['26', '13', '03', '23', '34', '18']
    # print(train_ids, val_ids, test_ids)
    for data_id in train_ids:
        train_data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))
    for data_id in val_ids:
        val_data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))
    for data_id in test_ids:
        test_data_file_names += list((data_path / (str(data_id)) / 'image').glob('*'))
    # print(data_file_names)
    # print(len(train_data_file_names), len(val_data_file_names), len(test_data_file_names))
    # random.shuffle(data_file_names)
    # training_num = 900
    # train_file_names = data_file_names[:700]
    # val_file_names = data_file_names[700:900]
    # test_file_names = data_file_names[900:]

    return train_data_file_names, val_data_file_names, test_data_file_names


class ESD_Dataset(Dataset):
    def __init__(self, file_names, ids=False):
        self.file_names = file_names
        self.image_transform = image_transform()
        self.mask_transform = mask_transform()
        self.ids = ids
        self.transforms = ToTensor()


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]

        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        data = {"image": image, "mask": mask}
        augmented = self.mask_transform(**data)
        mask = augmented["mask"]
        image = self.image_transform(image=image)

        image = image['image']
        # print(image.shape)
        image = self.transforms(image)
        label = torch.from_numpy(mask).long()
        # print(image.shape, label.shape)
        sample = {'image': image, 'label': label, 'id': str(img_file_name).split('\\')[-1]}
        if self.ids: 
            return sample['image'], sample['label'], sample['id']
        return sample['image'], sample['label']


def load_image(path):
    # print(str(path))
    img = cv2.imread(str(path))
    # print('Done Done Done')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_folder = 'mask'
    path = str(path).replace('image', mask_folder)
    # print(path)
    identifier = path.split('/')[-1]
    path = path.replace(identifier, identifier[:-4] + '_mask' + '.png')
    # mask = cv2.imread(path, 0)
    mask = cv2.imread(str(path), 0)
    # mask = (mask / factor).astype(np.uint8)
    # print(np.unique(mask))
    # if len(np.unique(mask)) == 7:
    #     print(np.unique(mask))

    # print(mask.all)
    # print(mask == 0)
    # print("____________________")
    # np.set_printoptions(threshold=np.inf)
    # print(mask == 255)
    mask[mask == 255] = 4
    mask[mask == 212] = 0
    mask[mask == 170] = 0
    mask[mask == 128] = 3
    mask[mask == 85] = 2
    mask[mask == 42] = 1
    # print(mask.all)
    
    return mask.astype(np.uint8)

