import os
import cv2

video_path = '/home/uto/data/code/xen/data/JayChou.mp4'
out_dir = '/home/uto/data/code/xen/data/JayChou'

cap = cv2.VideoCapture(video_path)
idx = 0
while True:
    sign, frame = cap.read()
    if not sign:
        break
    image_path = os.path.join(out_dir, str(idx) + '.png')
    cv2.imwrite(image_path, frame)
    idx += 1

cap.release()