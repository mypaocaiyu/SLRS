import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm


def compute_density_radius(points, k=3):
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    return np.mean(dists[:, 1:], axis=1)  # exclude self


def compute_edge_map(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    return edge


def compute_local_entropy(gray_img, window_size=9):
    return entropy(gray_img.astype(np.uint8), disk(window_size))


def compute_adaptive_radius_map(scribble, image, label_value=1, alpha=0.7, k=3, radius_scale=15):
    coords = np.argwhere(scribble == label_value)
    H, W = scribble.shape

    if len(coords) == 0:
        return [], []

    # Density factor
    d = compute_density_radius(coords, k)
    d_norm = d / np.max(d + 1e-5)

    # Edge factor
    edge_map = compute_edge_map(image)
    b = edge_map[coords[:, 0], coords[:, 1]]
    b_norm = b / np.max(b + 1e-5)

    # Texture complexity factor (local entropy)
    e_map = compute_local_entropy(image)
    c = e_map[coords[:, 0], coords[:, 1]]
    c_norm = c / np.max(c + 1e-5)

    # Weighted radius fusion
    w_d, w_b, w_c = 1/3, 1/3, 1/3
    r = alpha * (w_d * d_norm + w_b * (1 - b_norm) + w_c * (1 - c_norm))
    # r = np.sqrt(r * radius_scale)
    # r = r .astype(int)
    r = (r * radius_scale) .astype(int)
    r[r < 2] = 2  # min radius

    return coords, r


def expand_points_by_radius(scribble_shape, coords, radii, label_value):
    canvas = np.zeros(scribble_shape, dtype=np.uint8)
    for (y, x), r in zip(coords, radii):
        cv2.circle(canvas, center=(x, y), radius=int(r), color=int(label_value), thickness=-1)
    return canvas



# ====== 主流程 ======
scribble_dir = r'D:\pyProject\SLRS\CodDataset\train\Scribble'     # 涂鸦图像路径（灰度图：0/1/2）
image_dir = r'D:\pyProject\SLRS\CodDataset\train\Imgs'
save_dir = './Scribble' # 扩展后mask保存路径
os.makedirs(save_dir, exist_ok=True)
from PIL import Image
from pathlib import Path

for file in tqdm(os.listdir(scribble_dir)):

    scribble_path = os.path.join(scribble_dir, file)
    base_name = Path(file).stem  # 去掉扩展名，比如 "001.png" => "001"
    image_path = os.path.join(image_dir, base_name + '.jpg')  # 假设image是.jpg

    if not os.path.exists(image_path):
        print(f"⚠️ 图像文件不存在: {image_path}")
        continue

    scribble = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or scribble is None:
        print(f"❌ Skip {file}, image or label 读取失败.")
        continue


    # 前景
    fg_coords, fg_radii = compute_adaptive_radius_map(scribble, image, label_value=1, alpha=0.7)
    fg_mask = expand_points_by_radius(scribble.shape, fg_coords, fg_radii, label_value=1)

    # 背景
    bg_coords, bg_radii = compute_adaptive_radius_map(scribble, image, label_value=2, alpha=0.7)
    bg_mask = expand_points_by_radius(scribble.shape, bg_coords, bg_radii, label_value=2)

    final_mask = np.maximum(fg_mask, bg_mask)
    cv2.imwrite(os.path.join(save_dir, file), final_mask)

print("所有图像处理完成，结果保存在:", save_dir)
