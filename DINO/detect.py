import cv2
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict, annotate


def get_model(base_path, name):
    if name == 'swinb':
        model = load_model(base_path + '/DINO/groundingdino/config/GroundingDINO_SwinB_cfg.py',
                           base_path + '/checkpoints/DINO/groundingdino_swinb_cogcoor.pth', device='cuda:0')
    elif name == 'swint':
        model = load_model(base_path + '/DINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                           base_path + '/DINO/checkpoints/groundingdino_swint_ogc.pth', device='cuda:0')
    else:
        raise Exception('DINO model is not exists')

    return model


def detect_img(img_path, prompt, model, box_t=0.5, text_t=0.5):
    image, image_ = load_image(img_path)
    boxes, logits, phrases = predict(model, image_, prompt, box_t, text_t)

    return boxes, logits, phrases, image

    # idxs = []
    # log = pd.DataFrame({'logits': logits.numpy(), 'phrases': phrases, 'idx': [i for i in range(len(phrases))]})
    # for _, phr in log.groupby(['phrases']):
    #     idxs.append(phr[phr['logits'] == phr['logits'].max()]['idx'].values[0])
    # phrases = [phrases[i] for i in idxs]
    # idxs = torch.tensor(idxs)
    # boxes = boxes[idxs]
    # logits = logits[idxs]
    #
    # anno_img = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    #
    # H, W, _ = image_source.shape
    # boxes = boxes * torch.Tensor([W, H, W, H])
    # boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)
    #
    # return boxes, logits, phrases, anno_img


base_path = '/home/uto/data/code/xen'
name = 'swinb'
model = get_model(base_path, name)
img_path = '/home/uto/data/code/xen/data/JayChou/0.png'
prompt = 'person . clothing .'  # "chair . clothing . person . dog ."
boxes, logits, phrases, anno_img = detect_img(img_path, prompt, model)
cv2.imwrite('/home/uto/data/code/xen/data/0_d.png', anno_img)
print(boxes)
print(logits)
print(phrases)
