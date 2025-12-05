#!/usr/bin/env python

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

import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812

from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    RandomSubsetApply,
    SharpnessJitter,
    make_transform_from_config,
)


def _sample_uint8(h: int = 16, w: int = 16) -> torch.Tensor:
    return torch.randint(0, 256, (3, h, w), dtype=torch.uint8)


def test_image_transforms_base_only_applies_resize_and_dtype():
    img_tensor = _sample_uint8(8, 8)
    cfg = ImageTransformsConfig(enable=False, output_size=(32, 32))
    tf = ImageTransforms(cfg)

    output = tf(img_tensor)

    assert output.shape == (3, 32, 32)
    assert output.dtype == torch.float32
    assert output.max() <= 1.0 + 1e-6
    assert output.min() >= 0.0 - 1e-6


def test_image_transforms_with_augmentations_runs_and_preserves_shape_dtype():
    torch.manual_seed(0)
    img_tensor = _sample_uint8(32, 32)
    cfg = ImageTransformsConfig(enable=True, output_size=(64, 64), apply_random_subset=False)
    tf = ImageTransforms(cfg)

    output = tf(img_tensor)

    assert output.shape == (3, 64, 64)
    assert output.dtype == torch.float32


def test_image_transforms_random_subset_respects_max_num_transforms():
    torch.manual_seed(0)
    img_tensor = _sample_uint8(16, 16)
    cfg = ImageTransformsConfig(
        enable=True,
        output_size=(32, 32),
        apply_random_subset=True,
        max_num_transforms=1,
        tfs={
            "color_jitter": ImageTransformConfig(
                weight=1.0, type="ColorJitter", kwargs={"brightness": 0.5, "contrast": 0.5}
            ),
            "random_affine": ImageTransformConfig(
                weight=1.0, type="RandomAffine", kwargs={"degrees": 0, "translate": (0.0, 0.0)}
            ),
        },
    )
    tf = ImageTransforms(cfg)

    output = tf(img_tensor)

    assert output.shape == (3, 32, 32)
    assert output.dtype == torch.float32


@pytest.mark.parametrize(
    "tf_type, kwargs",
    [
        ("ColorJitter", {"brightness": 0.2}),
        ("RandomAffine", {"degrees": 5}),
        ("GaussianBlur", {"kernel_size": 3, "sigma": (0.1, 1.0)}),
        ("RandomErasing", {"p": 1.0, "scale": (0.1, 0.1), "ratio": (1.0, 1.0)}),
    ],
)
def test_make_transform_from_config_runs(tf_type, kwargs):
    img_tensor = _sample_uint8(16, 16)
    tf_cfg = ImageTransformConfig(type=tf_type, kwargs=kwargs)
    tf = make_transform_from_config(tf_cfg)
    output = tf(img_tensor)
    assert output.shape == img_tensor.shape


@pytest.mark.parametrize("p", [[0, 1], [1, 0]])
def test_random_subset_apply_single_choice(p):
    img_tensor = _sample_uint8(8, 8)
    flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
    random_choice = RandomSubsetApply(flips, p=p, n_subset=1, random_order=False)
    actual = random_choice(img_tensor)

    p_horz, _ = p
    if p_horz:
        torch.testing.assert_close(actual, F.horizontal_flip(img_tensor))
    else:
        torch.testing.assert_close(actual, F.vertical_flip(img_tensor))


def test_random_subset_apply_random_order():
    img_tensor = _sample_uint8(8, 8)
    flips = [v2.RandomHorizontalFlip(p=1), v2.RandomVerticalFlip(p=1)]
    random_order = RandomSubsetApply(flips, p=[0.5, 0.5], n_subset=2, random_order=True)
    # We can't really check whether the transforms are actually applied in random order. However,
    # horizontal and vertical flip are commutative. Meaning, even under the assumption that the transform
    # applies them in random order, we can use a fixed order to compute the expected value.
    actual = random_order(img_tensor)
    expected = v2.Compose(flips)(img_tensor)
    torch.testing.assert_close(actual, expected)


def test_random_subset_apply_probability_length_mismatch():
    color_jitters = [v2.ColorJitter(brightness=0.5), v2.ColorJitter(contrast=0.5)]
    with pytest.raises(ValueError):
        RandomSubsetApply(color_jitters, p=[0.5, 0.5, 0.1])


@pytest.mark.parametrize("n_subset", [0, 5])
def test_random_subset_apply_invalid_n_subset(n_subset):
    color_jitters = [v2.ColorJitter(brightness=0.5), v2.ColorJitter(contrast=0.5)]
    with pytest.raises(ValueError):
        RandomSubsetApply(color_jitters, n_subset=n_subset)


def test_sharpness_jitter_valid_range_tuple():
    img_tensor = _sample_uint8(8, 8)
    tf = SharpnessJitter((0.1, 2.0))
    output = tf(img_tensor)
    assert output.shape == img_tensor.shape


def test_sharpness_jitter_valid_range_float():
    img_tensor = _sample_uint8(8, 8)
    tf = SharpnessJitter(0.5)
    output = tf(img_tensor)
    assert output.shape == img_tensor.shape


def test_sharpness_jitter_invalid_range_min_negative():
    with pytest.raises(ValueError):
        SharpnessJitter((-0.1, 2.0))


def test_sharpness_jitter_invalid_range_max_smaller():
    with pytest.raises(ValueError):
        SharpnessJitter((2.0, 0.1))
