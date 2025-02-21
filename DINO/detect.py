import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from groundingdino.util.inference import load_model, load_image, predict


def get_dino(name, device):
    if name == 'swinb':
        model = load_model('../DINO/groundingdino/config/GroundingDINO_SwinB_cfg.py',
                           '../checkpoints/DINO/groundingdino_swinb_cogcoor.pth', device=device)
    elif name == 'swint':
        model = load_model('../DINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                           '../checkpoints/DINO/groundingdino_swint_ogc.pth', device=device)
    else:
        raise Exception('DINO model is not exists')

    return model


def detect_img(model, prompts, img_path, detect_path, box_t=0.5, text_t=0.5):
    text = ' . '.join(list(prompts.keys()))
    image, image_ = load_image(img_path)
    boxes, confs, objs = predict(model, image_, text, box_t, text_t)

    idxs = []
    mark = pd.DataFrame({'obj': objs, 'conf': confs.numpy(), 'idx': [i for i in range(len(objs))]})
    for k, v in mark.groupby('obj'):
        idxs.extend(v.sort_values('conf', ascending=False)['idx'].values[:prompts[k]])

    objs = [objs[i] for i in idxs]
    boxes = boxes.numpy()[idxs]
    confs = confs.numpy()[idxs]

    H, W, _ = image.shape
    boxes = boxes * np.array([W, H, W, H])
    x1 = ((boxes[:, 0] - boxes[:, 2] / 2).astype(int) - 4).clip(min=0, max=W)
    y1 = ((boxes[:, 1] - boxes[:, 3] / 2).astype(int) - 4).clip(min=0, max=H)
    x2 = ((boxes[:, 0] + boxes[:, 2] / 2).astype(int) + 4).clip(min=0, max=W)
    y2 = ((boxes[:, 1] + boxes[:, 3] / 2).astype(int) + 4).clip(min=0, max=H)

    ann = pd.DataFrame({'obj': objs, 'conf': confs, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    ann.to_csv(detect_path, index=False)


app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def app_detect():
    prompts = request.json['prompts']
    img_path = request.json['img_path']
    detect_path = request.json['detect_path']

    threading.Thread(target=detect_img, args=(model, prompts, img_path, detect_path)).start()

    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    model = get_dino('swinb', 'cuda:0')
    app.run(host='0.0.0.0', port=8081)
