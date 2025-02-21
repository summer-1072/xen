import os
import time
import requests
import pandas as pd
from pathlib import Path
from util import video2img, img2video


def detect_img(url, prompts, img_path, detect_path):
    requests.post(url, json={'prompts': prompts, 'img_path': img_path, 'detect_path': detect_path})
    while True:
        if Path(detect_path).is_file():
            return pd.read_csv(detect_path).values.tolist()
        time.sleep(1)


def segment_video(url, detects, video_path, segment_dir):
    requests.post(url, json={'detects': detects, 'video_path': video_path, 'segment_dir': segment_dir})
    segments = {}
    while True:
        if Path(os.path.join(segment_dir, 'status.json')).is_file():
            obj_dirs = sorted([i for i in os.listdir(segment_dir) if 'status.json' != i])
            objs = list(set([i.split('_')[0] for i in obj_dirs]))
            for obj in objs:
                segments[obj] = []
                for obj_dir in obj_dirs:
                    if obj_dir.split('_')[0] == obj:
                        segments[obj].append(obj_dir)
            return segments
        time.sleep(1)


def do_pipline(in_video, out_dir, prompts):
    img_dir = os.path.join(out_dir, 'img')
    detect_dir = os.path.join(out_dir, 'detect')
    segment_dir = os.path.join(out_dir, 'segment')

    os.makedirs(img_dir, exist_ok=True)
    video2img(in_video, img_dir)

    img_path = os.path.join(img_dir, sorted(os.listdir(img_dir))[0])
    detect_path = os.path.join(detect_dir, 'detect.txt')
    detects = detect_img(url_detect, prompts, img_path, detect_path)
    print(detects)

    segments = segment_video(url_segment, detects, in_video, segment_dir)
    print(segments)


if __name__ == "__main__":
    url_detect = 'http://127.0.0.1:8081/detect'
    url_segment = 'http://127.0.0.1:8082/segment'

    in_video = '/home/uto/data/code/xen/data/JayChou.mp4'
    out_dir = '/home/uto/data/code/xen/data/ann'
    prompts = {'person': 1, 'clothing': 1}  # "chair . clothing . person . dog ."
    do_pipline(in_video, out_dir, prompts)
