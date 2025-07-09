import argparse
import os
import sys
import glob

from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from ultralytics import YOLO

# Ensure engine.core is on path for D-FINE/DETR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

def draw_detections(images, labels, boxes, scores, thr, output_prefix):
    """
    Draw bounding boxes and labels on PIL images and save them.
    """
    for idx, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[idx]
        lab_filtered = labels[idx][scr > thr]
        box_filtered = boxes[idx][scr > thr]
        scr_filtered = scr[scr > thr]

        for j, b in enumerate(box_filtered):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), f"{lab_filtered[j].item()} {scr_filtered[j].item():.2f}", fill='blue')

        out_path = f"{output_prefix}_{idx}.jpg"
        im.save(out_path)
        print(f"Saved: {out_path}")


def process_image_detr(model, device, path, thr, output_dir):
    im_pil = Image.open(path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    labels, boxes, scores = model(im_data, orig_size)
    base = os.path.splitext(os.path.basename(path))[0]
    prefix = os.path.join(output_dir, base)
    draw_detections([im_pil], labels, boxes, scores, thr, prefix)


def process_video_detr(model, device, path, thr, output_dir):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path))[0]}.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        labels, boxes, scores = model(im_data, orig_size)
        draw_detections([frame_pil], labels, boxes, scores, thr,
                        os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0]))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        out.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video saved: {out_path}")


def process_yolo(inputs, weights, thr, output_dir):
    model = YOLO(weights)
    results = model(inputs, conf=thr)
    for i, r in enumerate(results):
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        out_path = os.path.join(output_dir, f"{i}.jpg")
        im_rgb.save(out_path)
        print(f"Saved: {out_path}")


def collect_image_files(path):
    """
    Collects image files from a directory or returns a single file path.
    """
    if os.path.isdir(path):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
        return sorted(files)
    else:
        return [path]


def main():
    parser = argparse.ArgumentParser(description="Unified inference for YOLO and DETR-family models.")
    parser.add_argument('-t', '--type', choices=['yolo', 'detr'], required=True,
                        help="Model type to run: yolo or detr.")
    parser.add_argument('-w', '--weights', type=str, help="Path to YOLO weights (required for yolo).")
    parser.add_argument('-c', '--config', type=str, help="Path to YAML config (required for detr).")
    parser.add_argument('-r', '--resume', type=str, help="Path to checkpoint (required for detr).")
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help="Input image(s), video(s), or directories of images.")
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Device for DETR models (e.g., cpu or cuda:0).")
    parser.add_argument('-o', '--output', type=str, default='results', help="Directory to save outputs.")
    parser.add_argument('--thresh', type=float, default=0.5, help="Confidence threshold (0-1).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.type == 'yolo':
        if not args.weights:
            parser.error('YOLO requires --weights')
        process_yolo(args.input, args.weights, args.thresh, args.output)
    else:
        # DETR path
        if not args.config or not args.resume:
            parser.error('DETR requires --config and --resume')
        # Load config and checkpoint
        cfg = YAMLConfig(args.config, resume=args.resume)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint.get('ema', checkpoint).get('module', checkpoint.get('model', checkpoint))
        cfg.model.load_state_dict(state)

        class ModelWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.post = cfg.postprocessor.deploy()
            def forward(self, imgs, sizes):
                out = self.model(imgs)
                return self.post(out, sizes)

        device = args.device
        model = ModelWrapper().to(device)

        for inp in args.input:
            # expand directories to individual image files
            paths = collect_image_files(inp)
            for p in paths:
                ext = os.path.splitext(p)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    process_image_detr(model, device, p, args.thresh, args.output)
                else:
                    process_video_detr(model, device, p, args.thresh, args.output)

if __name__ == '__main__':
    main()
