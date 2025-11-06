# datasets_kd.py

from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio
import cv2

# --- Helper functions from original crowd.py ---

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

# Note: gen_discrete_map is not needed for density map regression

class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):
        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, gt_dmap):
        wd, ht = img.size
        
        # --- Resize logic from Crowd_sh ---
        st_size = 1.0 * min(wd, ht)
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            gt_dmap = cv2.resize(gt_dmap, (wd, ht), interpolation=cv2.INTER_CUBIC) * (rr * rr)
        # --- End Resize logic ---

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        gt_dmap = gt_dmap[i:i + h, j:j + w]

        if random.random() > 0.5:
            img = F.hflip(img)
            gt_dmap = np.fliplr(gt_dmap)
        
        gt_dmap = torch.from_numpy(gt_dmap.copy()).float().unsqueeze(0)

        return self.trans(img), gt_dmap


class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size, downsample_ratio=8, method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        
    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        img = Image.open(img_path).convert('RGB')
        
        # MODIFICATION: Load .npy density map
        # This assumes you have run preprocess_dataset_qnrf.py which should also save density maps.
        # Your 'vis_densityMap.py' loads '.npy' files. We follow that logic.
        # QNRF gt path logic seems to be different in your repo.
        # 'vis_densityMap.py' implies gt_dmap_path = image_path.replace('.jpg', '.npy').replace('images', 'density_maps')
        # Let's assume a 'density_maps' folder exists parallel to 'train'/'val' folders.
        # This is a major assumption about your data structure.
        
        # Let's follow Crowd_sh logic, as it's simpler.
        gd_path = img_path.replace('jpg', 'npy') # Assumes .npy is next to .jpg
        gt_dmap = np.load(gd_path) # This is points, not dmap.
        
        # ABANDONING: Your dataset classes (Crowd_qnrf, Crowd_nwpu) load .npy files as *points*.
        # Your Crowd_sh loads .mat files as *points*.
        # The KD paper *requires* density maps for L_hard.
        # Your 'vis_densityMap.py' *DOES* load a density map:
        # gt_dmap_path = image_path.replace('.jpg', '.npy').replace('images', 'density_maps')
        # gt_dmap = np.load(gt_dmap_path)
        # We will use THIS logic.
        
        gt_dmap_path = img_path.replace('.jpg', '.npy').replace('images', 'density_maps')
        if not os.path.exists(gt_dmap_path):
             # Fallback for SHA/SHB structure
             gt_dmap_path = img_path.replace('.jpg', '.npy').replace('images', 'ground_truth_npy')
             if not os.path.exists(gt_dmap_path):
                raise FileNotFoundError(f"Could not find density map at {gt_dmap_path}. " \
                    "Please generate ground-truth .npy density maps and place them in a folder " \
                    "like 'density_maps' or 'ground_truth_npy'.")
        
        gt_dmap = np.load(gt_dmap_path)
        
        if self.method == 'train':
            return self.train_transform(img, gt_dmap)
        elif self.method == 'val':
            # Validation logic
            img = self.trans(img)
            gt_count = np.sum(gt_dmap)
            name = os.path.basename(img_path).split('.')[0]
            return img, gt_count, name


class Crowd_nwpu(Crowd_qnrf): # Inherits QNRF logic
    pass

class Crowd_sh(Base):
    def __init__(self, root_path, crop_size, downsample_ratio=8, method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        
        # MODIFICATION: Load .npy density map
        # Your 'vis_densityMap.py' shows this loading logic:
        # gt_dmap_path = image_path.replace('.jpg', '.npy').replace('images', 'density_maps')
        
        gt_dmap_path = img_path.replace('.jpg', '.npy').replace('images', 'density_maps')
        if not os.path.exists(gt_dmap_path):
            # Fallback: Your repo is inconsistent. 'test_image_patch.py' uses .mat files.
            # 'vis_densityMap.py' uses .npy files. We MUST use .npy for KD.
            # Assuming 'density_maps' folder exists as per 'vis_densityMap.py'
            gt_dmap_path = img_path.replace('.jpg', '.npy').replace('images', 'ground_truth_density_maps') # Another guess
            if not os.path.exists(gt_dmap_path):
                 raise FileNotFoundError(f"Could not find density map at {gt_dmap_path}. " \
                    "Please generate ground-truth .npy density maps for SHA/SHB.")
        
        img = Image.open(img_path).convert('RGB')
        gt_dmap = np.load(gt_dmap_path)

        if self.method == 'train':
            return self.train_transform(img, gt_dmap)
        elif self.method == 'val':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                img = img.resize((wd, ht), Image.BICUBIC)
            
            img = self.trans(img)
            gt_count = np.sum(gt_dmap)
            return img, gt_count, name