#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双腕SO-101系ロボットを推論のみで動かす最小例.

推論ループを外部から呼び出しやすいように関数化し、停止イベントによる終了もサポートする。

Todo:
    * 例外ハンドリングを細分化してUI側への詳細なエラー通知を行う。

"""

from __future__ import annotations

import sys
from pathlib import Path
import threading
from typing import Optional

# mission2ディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robots import bi_so101_follower  # ← これを忘れずに
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep


def run_inference(fps: int = 20, stop_event: Optional[threading.Event] = None) -> None:
    """双腕ロボット向けポリシー推論ループを実行する.

    Args:
        fps (int): 制御ループの更新周波数 (Frames Per Second)。
        stop_event (Optional[threading.Event]): 外部から停止を指示するイベント。``None`` の場合は無限ループ。

    Returns:
        None: 返り値はない。

    Raises:
        RuntimeError: カメラやロボットの初期化に失敗した場合に発生し得る。

    """

    # 1. ロボット設定
    robot_cfg = bi_so101_follower.BiSO101FollowerConfig(
        left_arm_port="/dev/ttyACM2",
        right_arm_port="/dev/ttyACM3",
        id="bi_so101_follower",
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=4,
                width=640,
                height=480,
                fps=30,
            ),
            "above": OpenCVCameraConfig(
                index_or_path=6,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG",
                warmup_s=2,
            ),
        },
    )
    robot = make_robot_from_config(robot_cfg)

    # 2. ポリシー設定（同封メタを優先）
    policy_cfg = PreTrainedConfig.from_pretrained("lt-s/AMD_hackathon2025_blanket_act_drape_slow_002400") ### モデル名を指定
    meta_repo_id = policy_cfg.repo_id or "lt-s/AMD_hackathon_drape_blanket"
    ds_meta = LeRobotDatasetMetadata(repo_id=meta_repo_id, force_cache_sync=False)

    # 3. ポリシーと前後処理
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    preproc, postproc = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map={}),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )

    _, robot_action_proc, robot_obs_proc = make_default_processors()

    # 4. 推論ループ
    robot.connect()
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            raw_obs = robot.get_observation()
            obs = robot_obs_proc(raw_obs)
            # obs_frame = build_dataset_frame(dataset.features, obs, prefix=obs/)
            obs_frame = build_dataset_frame(ds_meta.features, obs, prefix=OBS_STR)

            action_values = predict_action(
                observation=obs_frame,
                policy=policy,
                device=get_safe_torch_device(policy_cfg.device),
                preprocessor=preproc,
                postprocessor=postproc,
                use_amp=policy_cfg.use_amp,
                task="Lift the blanket by the red handle, unfold it, and then gently drape the blanket over the doll's neck.",
                robot_type=robot.robot_type,
            )
            robot_action = make_robot_action(action_values, ds_meta.features)
            action_to_send = robot_action_proc((robot_action, obs))
            robot.send_action(action_to_send)

            precise_sleep(1 / fps)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    run_inference()
