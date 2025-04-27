import os
import torch
import cv2
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms

from lib.config import cfg
from lib.utils.utils import create_logger, select_device
from lib.models import get_net
from lib.dataset import LoadImages
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def run_inference(input_path, output_dir, weights='weights/End-to-end.pth', device='cpu'):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'web_demo')
    device = select_device(logger, device)

    model = get_net(cfg)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    dataset = LoadImages(input_path, img_size=640)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    for path, img, img_det, _, shapes in dataset:
        img_tensor = transform(img).to(device).unsqueeze(0).float()

        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img_tensor)

        inf_out, _ = det_out
        det = non_max_suppression(inf_out)[0]

        # Segmentation masks
        da_seg_mask = da_seg_out.argmax(1).squeeze().cpu().numpy()
        ll_seg_mask = ll_seg_out.argmax(1).squeeze().cpu().numpy()

        # Resize masks to match img_det shape
        h, w, _ = img_det.shape
        da_seg_mask = cv2.resize(da_seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = cv2.resize(ll_seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply to image
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), None, None, is_demo=True)

        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img_det.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f"{int(cls)} {conf:.2f}"
                plot_one_box(xyxy, img_det, label=label, color=colors[int(cls)], line_thickness=2)

        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img_det)

        return save_path
