#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ローカル定義の image_transforms を lerobot-train に差し込むラッパー。

標準の `lerobot-train` CLI と同じ引数を受け取り、データセット生成だけを
`processors.image_transforms` に置き換える。LeRobot 本体に拡張を統合したため、
ここではそれらを呼び出す薄いラッパーとして振る舞う。
"""

from typing import Any, Sequence

import torch

from processors.image_transforms import make_train_image_transforms


def _guess_image_size(cfg) -> Sequence[int]:
    """環境設定などから画像解像度を推測する。

    Returns:
        Sequence[int]: (height, width)。取得できない場合は (256, 256)。
    """
    height = getattr(getattr(cfg, "env", None), "observation_height", None)
    width = getattr(getattr(cfg, "env", None), "observation_width", None)
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return (height, width)
    return (256, 256)


def make_dataset_with_local_transforms(cfg):
    """TrainPipelineConfig を受け取り、画像拡張つきの LeRobotDataset を生成する。

    Args:
        cfg: `TrainPipelineConfig` 相当の設定。

    Returns:
        LeRobotDataset: 学習用にローカル拡張を差し込んだデータセット。

    Raises:
        NotImplementedError: 複数データセット（MultiLeRobotDataset）が指定された場合。
    """
    from lerobot.datasets import factory as dataset_factory
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if not isinstance(cfg.dataset.repo_id, str):
        raise NotImplementedError("MultiLeRobotDataset は本ラッパーでは未対応です。")

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = dataset_factory.resolve_delta_timestamps(cfg.policy, ds_meta)

    image_size = _guess_image_size(cfg)
    image_transforms = make_train_image_transforms(
        enable_augmentation=cfg.dataset.image_transforms.enable, image_size=image_size
    )

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
    )

    if cfg.dataset.use_imagenet_stats:
        imagenet_stats = dataset_factory.IMAGENET_STATS
        for key in dataset.meta.camera_keys:
            for stats_type, stats in imagenet_stats.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def main(argv: list[str] | None = None) -> Any:
    """`lerobot-train` をデータセット差し替えで起動する。

    Args:
        argv (list[str] | None): コマンドライン引数。None の場合は `sys.argv[1:]` を利用。

    Returns:
        Any: `lerobot.scripts.lerobot_train.main` の返り値。
    """
    import sys

    import lerobot.datasets.factory as dataset_factory
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    argv = sys.argv[1:] if argv is None else argv
    dataset_factory.make_dataset = make_dataset_with_local_transforms
    return lerobot_train_main(argv)


if __name__ == "__main__":
    main()
