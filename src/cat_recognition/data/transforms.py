from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def build_train_transform(cfg):
    color_jitter = cfg.get("color_jitter", {})
    affine_cfg = cfg.get("affine", {})
    perspective_cfg = cfg.get("perspective", {})
    blur_cfg = cfg.get("gaussian_blur", {})
    elastic_cfg = cfg.get("elastic_transform", {})
    erasing_cfg = cfg.get("random_erasing", {})
    normalize_cfg = cfg.get("normalize", {})

    transform_list = [
        T.RandomResizedCrop(
            size=cfg.image_size,
            scale=tuple(cfg.get("crop_scale", [0.7, 1.0])),
            ratio=tuple(cfg.get("crop_ratio", [0.75, 1.3333333333])),
            interpolation=InterpolationMode.BICUBIC,
        )
    ]

    if float(affine_cfg.get("prob", 0.0)) > 0:
        affine = T.RandomAffine(
            degrees=float(affine_cfg.get("degrees", 0.0)),
            translate=tuple(affine_cfg["translate"]) if affine_cfg.get("translate") is not None else None,
            scale=tuple(affine_cfg["scale"]) if affine_cfg.get("scale") is not None else None,
            shear=tuple(affine_cfg["shear"]) if affine_cfg.get("shear") is not None else None,
            interpolation=InterpolationMode.BILINEAR,
            fill=float(affine_cfg.get("fill", 0.0)),
        )
        transform_list.append(T.RandomApply([affine], p=float(affine_cfg["prob"])))

    if float(perspective_cfg.get("prob", 0.0)) > 0:
        transform_list.append(
            T.RandomPerspective(
                distortion_scale=float(perspective_cfg.get("distortion_scale", 0.15)),
                p=float(perspective_cfg["prob"]),
                interpolation=InterpolationMode.BILINEAR,
                fill=float(perspective_cfg.get("fill", 0.0)),
            )
        )

    if cfg.get("horizontal_flip_prob", 0.0) > 0:
        transform_list.append(T.RandomHorizontalFlip(p=cfg.horizontal_flip_prob))

    if any(float(color_jitter.get(key, 0.0)) > 0 for key in ("brightness", "contrast", "saturation", "hue")):
        transform_list.append(
            T.ColorJitter(
                brightness=float(color_jitter.get("brightness", 0.0)),
                contrast=float(color_jitter.get("contrast", 0.0)),
                saturation=float(color_jitter.get("saturation", 0.0)),
                hue=float(color_jitter.get("hue", 0.0)),
            )
        )

    if float(blur_cfg.get("prob", 0.0)) > 0:
        kernel_size = blur_cfg.get("kernel_size", 5)
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
        sigma = blur_cfg.get("sigma", [0.1, 1.5])
        if isinstance(sigma, list):
            sigma = tuple(sigma)
        transform_list.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)],
                p=float(blur_cfg["prob"]),
            )
        )

    if float(elastic_cfg.get("prob", 0.0)) > 0:
        transform_list.append(
            T.RandomApply(
                [
                    T.ElasticTransform(
                        alpha=float(elastic_cfg.get("alpha", 50.0)),
                        sigma=float(elastic_cfg.get("sigma", 5.0)),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=float(elastic_cfg["prob"]),
            )
        )

    transform_list.extend(
        [
            T.ToTensor(),
            T.Normalize(
                mean=list(normalize_cfg.get("mean", [0.485, 0.456, 0.406])),
                std=list(normalize_cfg.get("std", [0.229, 0.224, 0.225])),
            ),
        ]
    )

    if float(erasing_cfg.get("prob", 0.0)) > 0:
        transform_list.append(
            T.RandomErasing(
                p=float(erasing_cfg["prob"]),
                scale=tuple(erasing_cfg.get("scale", [0.02, 0.18])),
                ratio=tuple(erasing_cfg.get("ratio", [0.3, 3.3])),
                value=erasing_cfg.get("value", "random"),
            )
        )

    return T.Compose(transform_list)


def build_eval_transform(cfg):
    normalize_cfg = cfg.get("normalize", {})
    return T.Compose(
        [
            T.Resize(cfg.get("resize_size", cfg.image_size), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=list(normalize_cfg.get("mean", [0.485, 0.456, 0.406])),
                std=list(normalize_cfg.get("std", [0.229, 0.224, 0.225])),
            ),
        ]
    )
