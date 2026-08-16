#!/usr/bin/env python3
"""Create class-agnostic AP-10K split1 trainval/test COCO annotations."""

import argparse
import copy
import json
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def animal_category(source):
    category = copy.deepcopy(source["categories"][0])
    category["id"] = 1
    category["name"] = "animal"
    category["supercategory"] = "animal"
    return category


def remap_annotations(annotations):
    result = copy.deepcopy(annotations)
    for annotation in result:
        annotation["category_id"] = 1
    return result


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(
        f"wrote {path}: {len(payload['images'])} images, "
        f"{len(payload['annotations'])} annotations"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/data1/hebei/huzhangchi/AP-10K/ap-10k/annotations"),
    )
    args = parser.parse_args()
    root = args.annotations

    train = load_json(root / "ap10k-train-split1.json")
    val = load_json(root / "ap10k-val-split1.json")
    test = load_json(root / "ap10k-test-split1.json")

    train_image_ids = {image["id"] for image in train["images"]}
    val_image_ids = {image["id"] for image in val["images"]}
    train_ann_ids = {ann["id"] for ann in train["annotations"]}
    val_ann_ids = {ann["id"] for ann in val["annotations"]}
    if train_image_ids & val_image_ids:
        raise ValueError("train and val image IDs overlap")
    if train_ann_ids & val_ann_ids:
        raise ValueError("train and val annotation IDs overlap")

    category = animal_category(train)
    trainval = copy.deepcopy(train)
    trainval["images"] = train["images"] + val["images"]
    trainval["annotations"] = remap_annotations(
        train["annotations"] + val["annotations"]
    )
    trainval["categories"] = [category]
    trainval.setdefault("info", {})["description"] = (
        "AP-10K split1 train+val, remapped to one generic animal category"
    )

    test_animal = copy.deepcopy(test)
    test_animal["annotations"] = remap_annotations(test["annotations"])
    test_animal["categories"] = [category]
    test_animal.setdefault("info", {})["description"] = (
        "AP-10K split1 test, remapped to one generic animal category"
    )

    write_json(root / "ap10k-trainval-split1-animal.json", trainval)
    write_json(root / "ap10k-test-split1-animal.json", test_animal)


if __name__ == "__main__":
    main()
