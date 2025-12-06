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
"""TensorBoard 連携用のユーティリティ。

Todo:
    必要に応じて動画ログなどにも対応する。
"""

from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


class TensorBoardLogger:
    """TensorBoard でスカラー値をロギングする薄いラッパー。

    Attributes:
        writer (SummaryWriter): TensorBoard の writer インスタンス。
        log_dir (Path): ログを書き込むディレクトリ。
    """

    def __init__(self, cfg: "TrainPipelineConfig"):
        """TensorBoard writer を初期化する。

        Args:
            cfg (TrainPipelineConfig): トレーニング設定。
        """
        self.log_dir = (
            Path(cfg.tensorboard.log_dir)
            if cfg.tensorboard.log_dir
            else Path(cfg.output_dir) / "tensorboard"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _to_scalar(self, value: Any) -> float | int | None:
        """スカラーに変換できる値のみを返す。

        Args:
            value (Any): ログ対象の値。

        Returns:
            float | int | None: スカラー変換結果。変換できない場合は None。
        """
        if isinstance(value, (int, float)):
            return value
        if hasattr(value, "item"):
            try:
                item = value.item()
            except Exception:
                return None
            if isinstance(item, (int, float)):
                return item
        return None

    def log_dict(self, metrics: dict[str, Any], step: int, mode: str = "train") -> None:
        """辞書形式のメトリクスを TensorBoard に書き込む。

        Args:
            metrics (dict[str, Any]): ログ対象のメトリクス。
            step (int): グローバルステップ。
            mode (str, optional): 出力時に付与するプレフィックス。デフォルトは ``train``。
        """
        for key, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is None:
                continue
            self.writer.add_scalar(f"{mode}/{key}", scalar, step)
        self.writer.flush()

    def close(self) -> None:
        """Writer を明示的にクローズする。"""
        self.writer.flush()
        self.writer.close()
