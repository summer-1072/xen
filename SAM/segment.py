import os
import json
import torch
import threading
import numpy as np
from flask import Flask, request, jsonify
from sam2.build_sam import build_sam2_video_predictor


def get_sam(name, device):
    if name == 'tiny':
        model = build_sam2_video_predictor('../sam2/configs/sam2.1/sam2.1_hiera_t.yaml',
                                           '../checkpoints/SAM/sam2.1_hiera_tiny.pt', device=device)
    elif name == 'small':
        model = build_sam2_video_predictor('../sam2/configs/sam2.1/sam2.1_hiera_s.yaml',
                                           '../checkpoints/SAM/sam2.1_hiera_small.pt', device=device)
    elif name == 'large':
        model = build_sam2_video_predictor('../sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                                           '../checkpoints/SAM/sam2.1_hiera_large.pt', device=device)
    elif name == 'base_plus':
        model = build_sam2_video_predictor('../sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
                                           '../checkpoints/SAM/sam2.1_hiera_base_plus.pt', device=device)
    else:
        raise Exception('SAM model is not exists')

    return model


def segment_video(model, detects, video_path, segment_dir):
    obj_dirs = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = model.init_state(video_path)
        for id, detect in enumerate(detects):
            obj, conf, x1, y1, x2, y2 = detect

            obj_dir = os.path.join(segment_dir, obj + '_' + str(id))
            obj_dirs[id] = obj_dir
            os.makedirs(obj_dir, exist_ok=True)

            model.add_new_points_or_box(inference_state=state, box=[x1, y1, x2, y2], frame_idx=0, obj_id=id)

        for idx, ids, masks in model.propagate_in_video(state):
            for id in ids:
                np.savetxt(os.path.join(obj_dirs[id], str(idx) + '.txt'), masks[id][0].cpu().numpy())

    with open(os.path.join(segment_dir, 'status.json'), 'w') as f:
        status = {'status': 'ok'}
        f.write(json.dumps(status, indent=2))


app = Flask(__name__)


@app.route('/segment', methods=['POST'])
def app_segment():
    detects = request.json['detects']
    video_path = request.json['video_path']
    segment_dir = request.json['segment_dir']

    threading.Thread(target=segment_video, args=(model, detects, video_path, segment_dir)).start()

    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    model = get_sam('large', 'cuda:0')
    app.run(host='0.0.0.0', port=8082)
