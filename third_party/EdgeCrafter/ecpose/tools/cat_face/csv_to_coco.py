"""
将猫脸关键点 CSV 数据集转换为 COCO JSON 格式
CSV 格式: filename, left_eye_x, left_eye_y, right_eye_x, right_eye_y, mouth_x, mouth_y,
          left_ear1_x, left_ear1_y, left_ear2_x, left_ear2_y, left_ear3_x, left_ear3_y,
          right_ear1_x, right_ear1_y, right_ear2_x, right_ear2_y, right_ear3_x, right_ear3_y

COCO 关键点顺序 (9个点):
0: left_eye
1: right_eye
2: mouth
3: left_ear1
4: left_ear2
5: left_ear3
6: right_ear1
7: right_ear2
8: right_ear3
"""

import csv
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# 关键点名称
KEYPOINT_NAMES = [
    "left_eye",
    "right_eye",
    "mouth",
    "left_ear1",
    "left_ear2",
    "left_ear3",
    "right_ear1",
    "right_ear2",
    "right_ear3",
]

# 骨架连接 (用于可视化)
SKELETON = [
    [0, 1],  # left_eye - right_eye
    [0, 2],  # left_eye - mouth
    [1, 2],  # right_eye - mouth
    [0, 3],  # left_eye - left_ear1
    [3, 4],  # left_ear1 - left_ear2
    [4, 5],  # left_ear2 - left_ear3
    [1, 6],  # right_eye - right_ear1
    [6, 7],  # right_ear1 - right_ear2
    [7, 8],  # right_ear2 - right_ear3
]


def csv_to_coco(csv_path, img_dir, output_path, start_ann_id=0):
    """将 CSV 转换为 COCO JSON 格式"""

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "cat_face",
            "supercategory": "animal",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON,
        }]
    }

    ann_id = start_ann_id

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in tqdm(reader, desc="Converting"):
            filename = row["filename"]
            if not filename.endswith(".jpg"):
                filename = filename + ".jpg"

            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            # 获取图片尺寸
            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue

            image_id = int(row["filename"])

            # 添加图片信息
            coco["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": w,
                "height": h,
            })

            # 解析关键点
            keypoints = []
            x_coords = []
            y_coords = []

            kp_pairs = [
                ("left_eye_x", "left_eye_y"),
                ("right_eye_x", "right_eye_y"),
                ("mouth_x", "mouth_y"),
                ("left_ear1_x", "left_ear1_y"),
                ("left_ear2_x", "left_ear2_y"),
                ("left_ear3_x", "left_ear3_y"),
                ("right_ear1_x", "right_ear1_y"),
                ("right_ear2_x", "right_ear2_y"),
                ("right_ear3_x", "right_ear3_y"),
            ]

            valid = True
            for x_key, y_key in kp_pairs:
                try:
                    x = float(row[x_key])
                    y = float(row[y_key])
                except (ValueError, KeyError):
                    valid = False
                    break

                # visibility: 2=可见, 1=被遮挡, 0=不存在
                # 这里假设所有关键点都可见
                v = 2
                keypoints.extend([x, y, v])
                x_coords.append(x)
                y_coords.append(y)

            if not valid:
                continue

            # 计算边界框 (从关键点推导)
            x_min = max(0, min(x_coords) - 20)
            y_min = max(0, min(y_coords) - 20)
            x_max = min(w, max(x_coords) + 20)
            y_max = min(h, max(y_coords) + 20)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            # 计算面积
            area = bbox_w * bbox_h

            ann_id += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": 9,
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "area": area,
                "iscrowd": 0,
            })

    # 保存 JSON
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    num_images = len(coco["images"])
    num_anns = len(coco["annotations"])
    print(f"Saved {num_images} images, {num_anns} annotations to {output_path}")
    return ann_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert cat face CSV to COCO JSON")
    parser.add_argument("--data-dir", type=str,
                        default="/data/lihb/Datasets/catrec/aistarted_cat_face",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str,
                        default="/data/lihb/Datasets/catrec/aistarted_cat_face/annotations",
                        help="Output directory for JSON files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 转换训练集
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_img_dir = os.path.join(args.data_dir, "train")
    train_output = os.path.join(args.output_dir, "train.json")

    print("Converting training set...")
    last_id = csv_to_coco(train_csv, train_img_dir, train_output)

    # 转换测试集 (如果有的话)
    test_csv = os.path.join(args.data_dir, "test.csv")
    if os.path.exists(test_csv):
        test_img_dir = os.path.join(args.data_dir, "test")
        test_output = os.path.join(args.output_dir, "test.json")
        print("Converting test set...")
        csv_to_coco(test_csv, test_img_dir, test_output, start_ann_id=last_id)
    else:
        print("No test.csv found, skipping test set conversion")
