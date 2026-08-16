import csv
import json
import os
import random
import hashlib
from PIL import Image
from tqdm import tqdm

random.seed(42)

KEYPOINT_NAMES = ["left_eye", "right_eye", "mouth", "left_ear1", "left_ear2", "left_ear3", "right_ear1", "right_ear2", "right_ear3"]
SKELETON = [[0,1],[0,2],[1,2],[0,3],[3,4],[4,5],[1,6],[6,7],[7,8]]

def parse_cat_file(cat_path):
    with open(cat_path, "r") as f:
        parts = f.read().strip().split()
    num_kp = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    keypoints = []
    for i in range(num_kp):
        keypoints.extend([coords[i*2], coords[i*2+1], 2])
    return keypoints

def parse_csv_row(row):
    kp_pairs = [("left_eye_x","left_eye_y"),("right_eye_x","right_eye_y"),("mouth_x","mouth_y"),
                ("left_ear1_x","left_ear1_y"),("left_ear2_x","left_ear2_y"),("left_ear3_x","left_ear3_y"),
                ("right_ear1_x","right_ear1_y"),("right_ear2_x","right_ear2_y"),("right_ear3_x","right_ear3_y")]
    keypoints = []
    for x_key, y_key in kp_pairs:
        keypoints.extend([float(row[x_key]), float(row[y_key]), 2])
    return keypoints

def get_bbox(keypoints, w, h):
    xs = [keypoints[i] for i in range(0, len(keypoints), 3)]
    ys = [keypoints[i] for i in range(1, len(keypoints), 3)]
    return [max(0, min(xs)-20), max(0, min(ys)-20), min(w, max(xs)+20)-max(0, min(xs)-20), min(h, max(ys)+20)-max(0, min(ys)-20)]

def add_entry(coco, img_path, filename, keypoints, ids):
    try:
        img = Image.open(img_path)
        w, h = img.size
    except:
        return ids
    bbox = get_bbox(keypoints, w, h)
    coco["images"].append({"id": ids[0], "file_name": filename, "width": w, "height": h})
    coco["annotations"].append({"id": ids[1], "image_id": ids[0], "category_id": 1, "keypoints": keypoints, "num_keypoints": 9, "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 0})
    return (ids[0]+1, ids[1]+1)

def main():
    old_dir = "/data/lihb/Datasets/catrec/aistarted_cat_face"
    new_dir = "/data/lihb/Datasets/catrec"
    out_dir = "/data/lihb/Datasets/catrec/merged_dataset"
    os.makedirs(out_dir, exist_ok=True)

    coco = {"images":[], "annotations":[], "categories":[{"id":1,"name":"cat_face","supercategory":"animal","keypoints":KEYPOINT_NAMES,"skeleton":SKELETON}]}
    ids = (0, 0)

    # 1. Process old train (CSV)
    print("=== 处理原 train 目录 ===")
    with open(f"{old_dir}/train.csv") as f:
        for row in tqdm(csv.DictReader(f), desc="Train CSV"):
            fn = row["filename"]
            if not fn.endswith(".jpg"): fn += ".jpg"
            img_path = f"{old_dir}/train/{fn}"
            if os.path.exists(img_path):
                ids = add_entry(coco, img_path, f"train_{fn}", parse_csv_row(row), ids)

    # 2. Build new dataset MD5 map
    print("\n=== 建立新数据集 MD5 映射 ===")
    new_md5 = {}
    for d in tqdm(["CAT_00","CAT_01","CAT_02","CAT_03","CAT_04","CAT_05","CAT_06"], desc="New dataset"):
        dp = f"{new_dir}/{d}"
        if not os.path.exists(dp): continue
        for f in os.listdir(dp):
            if f.endswith(".jpg"):
                ip = f"{dp}/{f}"
                cp = ip + ".cat"
                if os.path.exists(cp):
                    with open(ip,"rb") as fp: md5 = hashlib.md5(fp.read()).hexdigest()
                    new_md5[md5] = (cp, ip, f)

    # 3. Process old test (match with new dataset)
    print("\n=== 处理原 test 目录 ===")
    matched = unmatched = 0
    for fn in tqdm(os.listdir(f"{old_dir}/test"), desc="Test dir"):
        if not fn.endswith(".jpg"): continue
        ip = f"{old_dir}/test/{fn}"
        with open(ip,"rb") as f: md5 = hashlib.md5(f.read()).hexdigest()
        if md5 in new_md5:
            ids = add_entry(coco, ip, f"test_{fn}", parse_cat_file(new_md5[md5][0]), ids)
            matched += 1
        else:
            unmatched += 1
    print(f"匹配: {matched}, 未匹配: {unmatched}")

    # 4. Build old dataset MD5 set
    print("\n=== 处理新数据集独有图片 ===")
    old_md5 = set()
    for d in ["train","test"]:
        for fn in tqdm(os.listdir(f"{old_dir}/{d}"), desc=f"Hashing {d}"):
            if fn.endswith(".jpg"):
                with open(f"{old_dir}/{d}/{fn}","rb") as f: old_md5.add(hashlib.md5(f.read()).hexdigest())

    unique = 0
    for md5, (cp, ip, fn) in tqdm(new_md5.items(), desc="Unique"):
        if md5 not in old_md5:
            ids = add_entry(coco, ip, f"new_{fn}", parse_cat_file(cp), ids)
            unique += 1
    print(f"新数据集独有: {unique}")

    # 5. Save full dataset
    print(f"\n总图片: {len(coco['images'])}, 总标注: {len(coco['annotations'])}")
    with open(f"{out_dir}/annotations_all.json","w") as f: json.dump(coco, f, indent=2)

    # 6. Split train/test
    print("\n=== 划分 train/test ===")
    imgs = coco["images"]
    random.shuffle(imgs)
    sp = int(len(imgs)*0.9)
    train_ids = set(i["id"] for i in imgs[:sp])
    test_ids = set(i["id"] for i in imgs[sp:])

    train = {"images":imgs[:sp], "annotations":[a for a in coco["annotations"] if a["image_id"] in train_ids], "categories":coco["categories"]}
    test = {"images":imgs[sp:], "annotations":[a for a in coco["annotations"] if a["image_id"] in test_ids], "categories":coco["categories"]}

    with open(f"{out_dir}/train.json","w") as f: json.dump(train, f, indent=2)
    with open(f"{out_dir}/test.json","w") as f: json.dump(test, f, indent=2)

    print(f"Train: {len(train['images'])} images, {len(train['annotations'])} annotations")
    print(f"Test: {len(test['images'])} images, {len(test['annotations'])} annotations")

if __name__ == "__main__":
    main()
