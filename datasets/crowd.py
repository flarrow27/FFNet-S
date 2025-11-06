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
import pandas as pd
import time

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    
    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h* im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


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

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name


class Crowd_nwpu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name

class Crowd_sh(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))

        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        # cv2.ocl.setUseOpenCL(False)   #设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        # cv2.setNumThreads(0)  #设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground_truth', 'GT_{}.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

class Crowd_UCF_CC_50(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))

        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        # cv2.ocl.setUseOpenCL(False)   #设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        # cv2.setNumThreads(0)  #设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'ground_truth', '{}_ann.mat'.format(name))
        img = Image.open(img_path).convert('RGB')
        keypoints = sio.loadmat(gd_path)['annPoints']
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), name
    
class CustomDataset(Base):
    '''
    Class that allows training for a custom dataset. The folder are designed in the following way:
    root_dataset_path:
        -> images_1
        ->another_folder_with_image
        ->train.list
        ->valid.list

    The content of the lists file (csv with space as separator) are:
        img_xx__path label_xx_path
        img_xx1__path label_xx1_path

    where label_xx_path contains a list of x,y position of the head.
    '''
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'valid', 'test']:
            raise Exception("not implement")

        # read the list file
        self.img_to_label = {}
        list_file = f'{method}.list' # train.list, valid.list or test.list
        with open(os.path.join(self.root_path, list_file)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_to_label[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_to_label.keys()))


        print('number of img [{}]: {}'.format(method, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        gt_path = self.img_to_label[img_path]
        img_name = os.path.basename(img_path).split('.')[0]

        img = Image.open(img_path).convert('RGB')
        keypoints = self.load_head_annotation(gt_path)
       
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'valid' or self.method == 'test':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), img_name

    def load_head_annotation(self, gt_path):
        annotations = []
        with open(gt_path) as annotation:
            for line in annotation:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                annotations.append([x, y])
        return np.array(annotations)

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

class Crowd_visdrone(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val', 'test']: # VisDrone uses 'val' often, let's treat testlist as val
             raise Exception("not implement, use 'train' or 'val'")

        self.sequences_folder = os.path.join(self.root_path, 'sequences')
        self.anno_folder = os.path.join(self.root_path, 'annotations')
        self.im_list = []

        # Read trainlist.txt or testlist.txt to get image paths
        list_file = 'trainlist.txt' if method == 'train' else 'testlist.txt' # Use testlist for validation
        list_file_path = os.path.join(self.root_path, list_file)

        if not os.path.exists(list_file_path):
             raise FileNotFoundError(f"Could not find {list_file} in {self.root_path}")

        try:
            with open(list_file_path, 'r') as f:
                for line in f:
                    seq_name = line.strip()
                    if seq_name:
                        # Find all jpg images in that sequence folder
                        seq_path = os.path.join(self.sequences_folder, seq_name)
                        images_in_seq = sorted(glob(os.path.join(seq_path, '*.jpg')))
                        self.im_list.extend(images_in_seq)
        except Exception as e:
            print(f"Error reading {list_file_path}: {e}")

        if not self.im_list:
             print(f"Warning: No images found for method '{method}' in {self.sequences_folder} based on {list_file}")

        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def load_visdrone_annotation(self, txt_path):
        """Loads points from a VisDrone annotation txt file."""
        points = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    # VisDrone format might vary, often it's like:
                    # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
                    # We need the center point (x, y)
                    parts = line.strip().split(',')
                    # Check category is pedestrian (often category '1') - IMPORTANT check VisDrone docs
                    # For crowd counting challenges, often only category '1' (pedestrian) is counted
                    if len(parts) >= 6 and int(parts[5]) == 1: # Assuming category 1 is pedestrian
                        x1, y1, w, h = map(int, parts[0:4])
                        # Calculate center point
                        cx = x1 + w / 2.0
                        cy = y1 + h / 2.0
                        points.append([cx, cy])
        except FileNotFoundError:
            # For validation/test, annotations might not be provided, return empty
            return np.empty([0, 2])
        except Exception as e:
            print(f"Error reading annotation file {txt_path}: {e}")
            return np.empty([0, 2])
        return np.array(points)

    def __getitem__(self, item):
        print(f"[DataLoader {os.getpid()}] Attempting to get item {item}...")
        start_time = time.time()
        img_path = self.im_list[item]
        # Extract sequence name and image name for annotation path
        parts = img_path.split(os.sep)
        img_name_only = parts[-1].split('.')[0]
        seq_name = parts[-2]
        anno_path = os.path.join(self.anno_folder, seq_name, f'{img_name_only}.txt') # Annotation path includes sequence name

        print(f"[DataLoader {os.getpid()}] Loading image: {img_path}")
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[DataLoader {os.getpid()}] ERROR loading image {img_path}: {e}")
            # Return dummy data to prevent crash, but signal error
            dummy_img = self.trans(Image.new('RGB', (self.c_size, self.c_size)))
            dummy_points = np.empty([0, 2])
            dummy_discrete = torch.zeros(1, self.dc_size, self.dc_size)
            return dummy_img, torch.from_numpy(dummy_points).float(), dummy_discrete

        # --- START ADDED DEBUG CODE ---
        print(f"[DataLoader {os.getpid()}] Loading annotation: {anno_path}")
        # --- END ADDED DEBUG CODE ---
        keypoints = self.load_visdrone_annotation(anno_path)
        # --- START ADDED DEBUG CODE ---
        print(f"[DataLoader {os.getpid()}] Found {len(keypoints)} keypoints for item {item}")
        # --- END ADDED DEBUG CODE ---

        if self.method == 'train':
            # --- START ADDED DEBUG CODE ---
            print(f"[DataLoader {os.getpid()}] Applying train transform for item {item}...")
            # --- END ADDED DEBUG CODE ---
            result = self.train_transform(img, keypoints) # Calls the inherited Crowd_sh transform
            # --- START ADDED DEBUG CODE ---
            end_time = time.time()
            print(f"[DataLoader {os.getpid()}] Finished get item {item} in {end_time - start_time:.2f} seconds.")
            # --- END ADDED DEBUG CODE ---
            return result # Should return (transformed_img, keypoints_tensor, gt_discrete_tensor)
        elif self.method == 'val':
            # ... (keep existing validation logic) ...
            # --- START ADDED DEBUG CODE ---
            end_time = time.time()
            print(f"[DataLoader {os.getpid()}] Finished get item {item} (val) in {end_time - start_time:.2f} seconds.")
            # --- END ADDED DEBUG CODE ---
            return img, gt_count, f"{seq_name}_{img_name_only}"

    # Inherit the train_transform method from Crowd_sh (or Base, adjust if needed)
    # Ensure it handles keypoints correctly (cropping, flipping)
    # The version in Crowd_sh should work
    train_transform = Crowd_sh.train_transform