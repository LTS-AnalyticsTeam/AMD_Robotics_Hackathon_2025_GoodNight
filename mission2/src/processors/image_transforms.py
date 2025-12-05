#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""軽量な画像データ拡張ヘルパー

学習用と評価用で使い分ける torchvision.transforms.v2 ベースの変換定義。
LeRobotDataset にそのまま渡せる Callable を返す。
"""

from collections.abc import Sequence
from typing import Any

import torch
from torchvision.transforms import v2


def make_train_image_transforms(
    enable_augmentation: bool = True, image_size: Sequence[int] = (256, 256)
) -> v2.Compose:
    """学習用の画像変換パイプラインを生成する。

    リサイズと dtype 変換の後に、ColorJitter / RandomAffine / GaussianBlur /
    RandomErasing を適用する。``enable_augmentation`` が False の場合は、
    拡張を省いて最小限の前処理のみを返す。

    Args:
        enable_augmentation (bool): True の場合はデータ拡張を有効化する。
        image_size (Sequence[int]): (height, width) で指定する出力解像度。

    Returns:
        v2.Compose: LeRobotDataset に渡せる torchvision v2 の Compose オブジェクト。

    Raises:
        ValueError: ``image_size`` の長さが 2 でない場合。
    """
    if len(image_size) != 2:
        raise ValueError("image_size は (height, width) の 2 要素で指定してください。")

    base_transforms: list[Any] = [
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if not enable_augmentation:
        return v2.Compose(base_transforms)

    augmentation_transforms: list[Any] = [
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        v2.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.98, 1.02)),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
    ]

    return v2.Compose(base_transforms + augmentation_transforms)


def make_eval_image_transforms(image_size: Sequence[int] = (256, 256)) -> v2.Compose:
    """評価・推論用の最小前処理を生成する。

    リサイズと ToDtype (scale=True) のみを適用する。

    Args:
        image_size (Sequence[int]): (height, width) で指定する出力解像度。

    Returns:
        v2.Compose: 評価用の torchvision v2 Compose オブジェクト。

    Raises:
        ValueError: ``image_size`` の長さが 2 でない場合。
    """
    if len(image_size) != 2:
        raise ValueError("image_size は (height, width) の 2 要素で指定してください。")

    return v2.Compose(
        [
            v2.Resize(image_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


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

    if train:
        image_transforms = make_train_image_transforms(
            enable_augmentation=enable_augmentation, image_size=image_size
        )
    else:
        image_transforms = make_eval_image_transforms(image_size=image_size)

    return LeRobotDataset(repo_id=repo_id, image_transforms=image_transforms, **dataset_kwargs)
