# generate_sha_shb_dmaps.py

import os
import cv2
import numpy as np
import scipy.io as sio
from glob import glob
from PIL import Image

def gen_density_map_gaussian(im_height, im_width, points, sigma=4):
    """
    Generates a Gaussian density map.
    Adapted from preprocess/preprocess_dataset_nwpu.py (with corrections)
    """
    density_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = 0
    if points is not None:
        # Filter points to be within image bounds
        points = points[
            (points[:, 0] >= 0) & (points[:, 0] < w) &
            (points[:, 1] >= 0) & (points[:, 1] < h)
        ]
        num_gt = points.shape[0]
        if num_gt == 0:
            return density_map
    else:
        return density_map

    for p in points:
        p = np.round(p).astype(int)
        # p[0] is x (width/col), p[1] is y (height/row)
        
        gaussian_radius = sigma * 2 - 1
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma),
            cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma).T
        )
        
        # --- Slices for Gaussian Map ---
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        
        # --- Slices for Density Map ---
        y_start = max(0, p[1] - gaussian_radius)
        y_end = min(h, p[1] + gaussian_radius + 1)
        x_start = max(0, p[0] - gaussian_radius)
        x_end = min(w, p[0] + gaussian_radius + 1)
        
        # --- Adjust Gaussian Slices for Edges ---
        if p[0] < gaussian_radius: # near left edge
            x_left = gaussian_radius - p[0]
        if p[1] < gaussian_radius: # near top edge
            y_up = gaussian_radius - p[1]
        if p[0] + gaussian_radius >= w: # near right edge
            x_right = gaussian_map.shape[1] - (p[0] + gaussian_radius - w) - 1
        if p[1] + gaussian_radius >= h: # near bottom edge
            y_down = gaussian_map.shape[0] - (p[1] + gaussian_radius - h) - 1
            
        # Ensure slice indices are valid
        x_left, x_right = max(0, x_left), max(0, x_right)
        y_up, y_down = max(0, y_up), max(0, y_down)

        if x_left >= x_right or y_up >= y_down:
            continue # Skip this point if logic is bad

        # Get the correctly sliced gaussian kernel
        gaussian_map_sliced = gaussian_map[y_up:y_down, x_left:x_right]

        # Get the slice from the density map
        density_map_slice = density_map[y_start:y_end, x_start:x_end]

        # Check shapes before adding
        if density_map_slice.shape != gaussian_map_sliced.shape:
            # This can happen on extreme corners.
            # Adjust by slicing the gaussian map again to match the density map.
            gaussian_map_sliced = gaussian_map_sliced[
                0:density_map_slice.shape[0],
                0:density_map_slice.shape[1]
            ]
            
        density_map[y_start:y_end, x_start:x_end] += gaussian_map_sliced

    if np.sum(density_map) > 0:
        # Normalize to match ground truth count
        density_map = (density_map / np.sum(density_map)) * num_gt
    
    return density_map

def process_dataset(data_path, sigma=4):
    print(f"Processing dataset at: {data_path}")
    image_files = sorted(glob(os.path.join(data_path, 'images', '*.jpg')))

    for img_path in image_files:
        name = os.path.basename(img_path).split('.')[0]
        gt_mat_path = os.path.join(data_path, 'ground_truth', f'GT_{name}.mat')

        # Define where to save the .npy file
        save_folder = os.path.join(data_path, 'density_maps')
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f'{name}.npy')

        if os.path.exists(save_path):
            print(f"Skipping {name}, .npy file already exists.")
            continue

        # Load image to get shape
        try:
            img = Image.open(img_path).convert('RGB')
            im_width, im_height = img.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Load points
        try:
            keypoints = sio.loadmat(gt_mat_path)['image_info'][0][0][0][0][0]
        except Exception as e:
            print(f"Error loading .mat file {gt_mat_path}: {e}")
            keypoints = None

        print(f"Generating density map for {name}...")
        dmap = gen_density_map_gaussian(im_height, im_width, keypoints, sigma)

        # Re-scale density map to match count
        if keypoints is not None:
            gt_count = len(keypoints)
            if np.sum(dmap) > 0:
                dmap = (dmap / np.sum(dmap)) * gt_count

        np.save(save_path, dmap)

if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    # Adjust this to point to your ShanghaiTech Part A or B folder
    shb_path = 'C:/Users/jabin/All/College/gemini/Fuss-Free-structure/datasets/part_B_final'

    # Process Part B
    process_dataset(os.path.join(shb_path, 'train_data'))
    process_dataset(os.path.join(shb_path, 'test_data'))

    print("All density maps generated and saved as .npy files.")