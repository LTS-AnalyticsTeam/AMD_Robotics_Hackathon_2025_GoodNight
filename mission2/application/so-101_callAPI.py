#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双腕SO-101系ロボットを推論のみで動かす最小例."""

import time
import bi_so101_follower  # ← これを忘れずに
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep


def run_inference(fps: int = 20) -> None:
    """推論ループを実行する."""
    # 1. ロボット設定
    robot_cfg = bi_so101_follower.BISo101FollowerConfig(
        left_arm_port="/dev/ttyACM2",
        right_arm_port="/dev/ttyACM3",
        id="bi_so101_follower",
        cameras={
            "base_0_rgb": OpenCVCameraConfig(type="opencv", index_or_path=4, width=640, height=480, fps=30),
            "left_wrist_0_rgb": OpenCVCameraConfig(
                type="opencv", index_or_path=6, width=640, height=480, fps=30, fourcc="MJPG", warmup_s=2
            ),
        },
    )
    robot = make_robot_from_config(robot_cfg)

    # 2. データセットメタ情報
    dataset = LeRobotDataset(
        repo_id="lt-s/eval_AMD_hackathon_place_blanket",
        download_videos=False,
        batch_encoding_size=1,
    )

    # 3. ポリシーと前後処理
    policy_cfg = PreTrainedConfig.from_pretrained("lt-s/AMD_hackathon2025_blanket_pi05_006000")
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preproc, postproc = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, rename_map={}),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )

    _, robot_action_proc, robot_obs_proc = make_default_processors()

    # 4. 推論ループ
    robot.connect()
    try:
        while True:
            raw_obs = robot.get_observation()
            obs = robot_obs_proc(raw_obs)
            obs_frame = build_dataset_frame(dataset.features, obs, prefix="obs/")

            action_values = predict_action(
                observation=obs_frame,
                policy=policy,
                device=get_safe_torch_device(policy_cfg.device),
                preprocessor=preproc,
                postprocessor=postproc,
                use_amp=policy_cfg.use_amp,
                task="Grab the red grip to unfold the blanket, then gently place the blanket over the doll.",
                robot_type=robot.robot_type,
            )
            robot_action = make_robot_action(action_values, dataset.features)
            action_to_send = robot_action_proc((robot_action, obs))
            robot.send_action(action_to_send)

            precise_sleep(1 / fps)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    run_inference()
