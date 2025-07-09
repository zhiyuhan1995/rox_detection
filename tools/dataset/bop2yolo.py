import os
import json
import shutil
from tqdm import tqdm
from PIL import Image

import argparse

def convert_bbox_to_yolo(x, y, w, h, img_w, img_h):
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return cx, cy, w, h

subsets = ["train", "test"]

def delete_folder_contents(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def convert_subset(bop_root:str, subset:str):
    bop_subset = os.path.join(bop_root, f"./{subset}_pbr")
    output_root = f"./data_yolo/{subset}_yolo"

    images_dir = os.path.join(output_root, "images")
    labels_dir = os.path.join(output_root, "labels")

    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
    else:
        delete_folder_contents(images_dir)

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir, exist_ok=True)
    else:
        delete_folder_contents(labels_dir)

    scene_ids = sorted(os.listdir(bop_subset))
    global_image_index = 0

    for scene_id in tqdm(scene_ids, desc=f"Converting {subset}_pbr"):
        scene_path = os.path.join(bop_subset, scene_id)
        coco_path = os.path.join(scene_path, "scene_gt_coco.json")

        if not os.path.exists(coco_path):
            continue

        with open(coco_path, 'r') as f:
            coco_data = json.load(f)

        images_info = {img["id"]: img for img in coco_data["images"]}
        annotations_by_image = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            annotations_by_image.setdefault(img_id, []).append(ann)

        for img_id, img_info in images_info.items():
            file_name = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            new_image_name = f"{global_image_index:06d}.png"
            new_label_name = f"{global_image_index:06d}.txt"
            global_image_index += 1

            # 拷贝图像
            src_img_path = os.path.join(scene_path,  file_name[:-3]+"png")
            dst_img_path = os.path.join(images_dir, new_image_name)

            with Image.open(src_img_path) as im:
                im.convert("RGB").save(dst_img_path, "PNG")

            # 写入 label
            label_path = os.path.join(labels_dir, new_label_name)
            with open(label_path, "w") as f:
                for ann in annotations_by_image.get(img_id, []):
                    x, y, w, h = ann["bbox"]
                    cx, cy, bw, bh = convert_bbox_to_yolo(x, y, w, h, width, height)
                    class_id = 0 #ann["category_id"] - 1
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def convert_dataset(bop_root):
    for subset in subsets:
        convert_subset(bop_root, subset)
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, help='path to your bop dataset')

    args = parser.parse_args()


    convert_dataset(args.data)

if __name__ == '__main__':
    main()

