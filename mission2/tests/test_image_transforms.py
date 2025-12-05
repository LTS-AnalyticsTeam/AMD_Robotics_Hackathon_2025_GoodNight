import torch

from processors.image_transforms import (
    make_eval_image_transforms,
    make_train_image_transforms,
)


def test_train_transform_without_augmentation_scales_and_resizes():
    torch.manual_seed(0)
    transform = make_train_image_transforms(enable_augmentation=False, image_size=(64, 64))
    sample = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)

    output = transform(sample)

    assert output.shape == (3, 64, 64)
    assert output.dtype == torch.float32


def test_train_transform_with_augmentation_runs():
    torch.manual_seed(0)
    transform = make_train_image_transforms(enable_augmentation=True, image_size=(64, 64))
    sample = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)

    output = transform(sample)

    assert output.shape == (3, 64, 64)
    assert output.dtype == torch.float32


def test_eval_transform_scales_to_unit_range():
    transform = make_eval_image_transforms(image_size=(32, 32))
    sample = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)

    output = transform(sample)

    assert output.shape == (3, 32, 32)
    assert output.dtype == torch.float32
    assert output.max() <= 1.0 + 1e-6
    assert output.min() >= 0.0 - 1e-6
