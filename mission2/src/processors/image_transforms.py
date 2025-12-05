#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LeRobot 本体の ImageTransforms に委譲するラッパー。

ローカル実装をやめ、サブモジュール側の設定可能な Transform を再利用することで
実装の重複を避ける。
"""

from collections.abc import Sequence
from typing import Any

from lerobot.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig


def _build_config(enable_augmentation: bool, image_size: Sequence[int]) -> ImageTransformsConfig:
    """ローカル利用向けの ImageTransformsConfig を生成する。

    Args:
        enable_augmentation (bool): 拡張を有効にするか。
        image_size (Sequence[int]): (height, width) で指定する出力解像度。

    Returns:
        ImageTransformsConfig: LeRobot 互換の画像変換設定。

    Raises:
        ValueError: ``image_size`` の長さが 2 でない場合。
    """
    if len(image_size) != 2:
        raise ValueError("image_size は (height, width) の 2 要素で指定してください。")

    return ImageTransformsConfig(
        enable=enable_augmentation,
        output_size=tuple(image_size),
        apply_random_subset=False,
        tfs={
            "color_jitter": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02},
            ),
            "random_affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": 3, "translate": (0.03, 0.03), "scale": (0.98, 1.02)},
            ),
            "gaussian_blur": ImageTransformConfig(
                weight=1.0,
                type="GaussianBlur",
                kwargs={"kernel_size": 3, "sigma": (0.1, 1.0)},
            ),
            "random_erasing": ImageTransformConfig(
                weight=1.0,
                type="RandomErasing",
                kwargs={"p": 0.2, "scale": (0.02, 0.05), "ratio": (0.3, 3.3)},
            ),
        },
    )


def make_train_image_transforms(
    enable_augmentation: bool = True, image_size: Sequence[int] = (256, 256)
) -> ImageTransforms:
    """学習用の画像変換パイプラインを生成する。

    Args:
        enable_augmentation (bool): True の場合はデータ拡張を有効化する。
        image_size (Sequence[int]): (height, width) で指定する出力解像度。

    Returns:
        ImageTransforms: LeRobotDataset に渡せる Transform オブジェクト。
    """
    cfg = _build_config(enable_augmentation, image_size)
    return ImageTransforms(cfg)


def make_eval_image_transforms(image_size: Sequence[int] = (256, 256)) -> ImageTransforms:
    """評価・推論用の最小前処理を生成する。

    Args:
        image_size (Sequence[int]): (height, width) で指定する出力解像度。

    Returns:
        ImageTransforms: 評価用 Transform オブジェクト（リサイズと ToDtype のみ）。
    """
    cfg = _build_config(enable_augmentation=False, image_size=image_size)
    return ImageTransforms(cfg)


def make_lerobot_dataset(
    repo_id: str,
    *,
    train: bool = True,
    enable_augmentation: bool = True,
    image_size: Sequence[int] = (256, 256),
    **dataset_kwargs: Any,
):
    """LeRobotDataset を学習/評価モードに応じた image_transforms 付きで生成する。

    Args:
        repo_id (str): Hugging Face Hub 上のリポジトリ ID またはローカルパス。
        train (bool): True の場合は学習用（データ拡張有無を ``enable_augmentation`` で制御）。
        enable_augmentation (bool): 学習時に拡張を適用するかどうか。
        image_size (Sequence[int]): (height, width) で指定する出力解像度。
        **dataset_kwargs (Any): ``LeRobotDataset`` にそのまま渡す追加引数（``root`` や ``episodes`` など）。

    Returns:
        LeRobotDataset: 指定した変換を適用するデータセットインスタンス。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = _build_config(enable_augmentation if train else False, image_size)
    image_transforms = ImageTransforms(cfg)

    return LeRobotDataset(repo_id=repo_id, image_transforms=image_transforms, **dataset_kwargs)
