#!/usr/bin/env python
"""LeRobot 用の画像変換ユーティリティ。

リサイズと dtype 変換などの基本前処理に加えて、学習時のデータ拡張も統合的に扱う。
``ImageTransformsConfig`` を通じてサイズや拡張の有無を制御し、評価時はリサイズ+スケールのみ、
学習時は基本前処理に続けて拡張を適用する。
"""

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """ToDtype 用の dtype を解決する。

    Args:
        dtype (str | torch.dtype): 文字列表記または torch.dtype。

    Returns:
        torch.dtype: 解決した dtype。

    Raises:
        ValueError: 未対応の dtype が指定された場合。
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if resolved is None or not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unsupported dtype: {dtype}")
        return resolved
    raise ValueError(f"Unsupported dtype type: {type(dtype)}")


def _build_base_transforms(output_size: Sequence[int], antialias: bool, dtype: str | torch.dtype) -> list[Any]:
    """リサイズと ToDtype をまとめた基本前処理を組み立てる。

    Args:
        output_size (Sequence[int]): (height, width) で指定する出力解像度。
        antialias (bool): Resize 時にアンチエイリアスを有効にするか。
        dtype (str | torch.dtype): ToDtype に渡す dtype。

    Returns:
        list[Any]: torchvision v2 の Transform 一覧。

    Raises:
        ValueError: ``output_size`` の長さが 2 でない場合。
    """
    if len(output_size) != 2:
        raise ValueError("output_size は (height, width) の 2 要素で指定してください。")

    return [
        v2.Resize(output_size, antialias=antialias),
        v2.ToDtype(_resolve_dtype(dtype), scale=True),
    ]


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    画像前処理と拡張の構成をまとめる設定。

    * ``output_size`` に基づく Resize と ToDtype は常に適用される（学習/評価共通）。
    * ``enable`` が True の場合のみ、その後に拡張を適用する。
    * ``apply_random_subset`` が True の場合は従来通り RandomSubsetApply でサンプリングし、
      False の場合は列挙順で順次適用する。
    """

    # Set this flag to `true` to enable data augmentation after base transforms.
    enable: bool = False
    # Output image size as (height, width).
    output_size: tuple[int, int] = (256, 256)
    # Use anti-aliasing for Resize.
    antialias: bool = True
    # Dtype string or torch.dtype for ToDtype.
    dtype: str | torch.dtype = "float32"
    # Apply subset-sampling strategy (RandomSubsetApply) if True, otherwise apply in order.
    apply_random_subset: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 4
    # By default, transforms are applied in the order defined by tfs.
    # Set this to True to apply them in a random order when subset sampling is enabled.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
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
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig):
    """ImageTransformConfig から対応する torchvision Transform を生成する。

    Args:
        cfg (ImageTransformConfig): 生成に使用する設定。

    Returns:
        Transform: 構築した Transform インスタンス。

    Raises:
        ValueError: 未対応の Transform 名が指定された場合。
    """
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    elif cfg.type == "GaussianBlur":
        return v2.GaussianBlur(**cfg.kwargs)
    elif cfg.type == "RandomErasing":
        return v2.RandomErasing(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """設定に基づいて基本前処理と拡張を合成する Transform。

    Attributes:
        _cfg (ImageTransformsConfig): Transform 構築に使用した設定。
        weights (list[float]): サブセットサンプリング用の重み。
        transforms (dict[str, Transform]): 拡張の名前とインスタンスの対応。
        tf (Transform): 実際に適用する合成 Transform。
    """

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights: list[float] = []
        self.transforms: dict[str, Transform] = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        base_transforms = _build_base_transforms(
            cfg.output_size,
            antialias=cfg.antialias,
            dtype=cfg.dtype,
        )

        augmentation_tf: Transform
        if not cfg.enable or len(self.transforms) == 0:
            augmentation_tf = v2.Identity()
        elif cfg.apply_random_subset:
            n_subset = min(len(self.transforms), cfg.max_num_transforms)
            augmentation_tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset if n_subset > 0 else len(self.transforms),
                random_order=cfg.random_order,
            )
        else:
            augmentation_tf = v2.Compose(list(self.transforms.values()))

        self.tf = v2.Compose([*base_transforms, augmentation_tf])

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
