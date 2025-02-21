import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor


def save_img(img_path, img_mask_path, mask):
    plt.figure(figsize=(9, 16))
    plt.imshow(Image.open(img_path))

    cmap = plt.get_cmap("tab10")
    color = np.array([*cmap(1)[:3], 0.6])
    H, W = mask.shape[-2:]
    mask_img = mask.reshape(H, W, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_img)
    plt.savefig(img_mask_path)


predictor = build_sam2_video_predictor('../sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                       '../checkpoints/SAM/sam2.1_hiera_large.pt', device='cuda:0')
video_dir = "../data/JayChou"
out_dir = "../data/seg"
frames = sorted(os.listdir(video_dir))
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state('../data/JayChou.mp4')
    box1 = np.array([31, 427, 540, 960], dtype=np.float32)
    box2 = np.array([31, 192, 540, 960], dtype=np.float32)

    predictor.add_new_points_or_box(inference_state=state, box=box1, frame_idx=0, obj_id=0)
    predictor.add_new_points_or_box(inference_state=state, box=box2, frame_idx=0, obj_id=1)

    for idx, ids, masks in predictor.propagate_in_video(state):
        save_img(os.path.join(video_dir, frames[idx]), os.path.join(out_dir, frames[idx]),
                 (masks[0][0] > -1).cpu().numpy())
