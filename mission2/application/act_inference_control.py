#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LeRobot推奨パターンに合わせたACT推論・評価用スクリプト（SO-101両腕）.

LeRobotドキュメントの`record_loop`サンプルをSO-101両腕（BiSO101Follower）向けに
移植し、指定モデルで推論しながら評価エピソードを記録する。

Todo:
    * UI連携（Streamlitなど）でパラメーター変更を可能にする。
    * ローカル保存のみでHubへは後でまとめてpushするモードを追加する。

exsample
python mission2/application/act_inference_control.py --model-id lt-s/AMD_hackathon2025_blanket_act_drape_006000 --meta-repo-id lt-s/AMD_hackathon_drape_blanket --task "Grab the red grip to unfold the blanket, then gently place it." --num-episodes 5 --fps 30 --episode-time-s 60 --left-arm-port /dev/ttyACM2 --right-arm-port /dev/ttyACM3 --front-cam-index 4 --above-cam-index 6 --device cuda --no-save-dataset
"""

from __future__ import annotations

import argparse
import sys
import warnings
import time
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.processor import make_default_processors
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR
from lerobot.policies.utils import make_robot_action
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from robots import BiSO101Follower, BiSO101FollowerConfig


def _resolve_device(requested: Optional[str]) -> str:
    """CUDA使用可否を判定し、安全なデバイス名を返す.

    Args:
        requested (Optional[str]): 利用したいデバイス（例 ``"cuda"``、``"cpu"``）。

    Returns:
        str: 実際に使用するデバイス名。

    """

    if requested and requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDAが利用できないためCPUにフォールバックします。", RuntimeWarning)
        return "cpu"
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def run_eval_inference(
    model_id: str,
    dataset_id: Optional[str],
    task_description: str,
    num_episodes: int,
    fps: int,
    episode_time_s: int,
    left_arm_port: str,
    right_arm_port: str,
    front_cam_index: int,
    above_cam_index: int,
    device: Optional[str],
    use_videos: bool,
    image_writer_threads: int,
    push_to_hub: bool,
    save_dataset: bool,
    meta_repo_id: Optional[str],
    stop_event: Optional[threading.Event] = None,
) -> None:
    """ACTポリシーで推論しつつ評価エピソードを記録する、または推論のみ行う.

    Args:
        model_id (str): Hugging Face Hub上のモデルID。
        dataset_id (str): 評価結果を保存するデータセットID。
        task_description (str): タスク文（single_task）として保存。
        num_episodes (int): 実行するエピソード数。
        fps (int): 制御および記録のFPS。
        episode_time_s (int): 1エピソードの制御時間（秒）。
        left_arm_port (str): 左腕フォロワーのシリアルポート。
        right_arm_port (str): 右腕フォロワーのシリアルポート。
        front_cam_index (int): 正面カメラのデバイスID。
        above_cam_index (int): 上部カメラのデバイスID。
        device (Optional[str]): 使用デバイス。``None`` は自動判定。
        use_videos (bool): データセット保存時に動画を含めるかどうか。
        image_writer_threads (int): 動画保存のスレッド数。
        push_to_hub (bool): 実行終了時にHubへpushするかどうか。
        save_dataset (bool): データセットを保存するかどうか。Falseの場合は推論のみ。
        meta_repo_id (Optional[str]): 推論のみのときに参照するメタデータリポジトリID。

    Returns:
        None: 返り値はない。

    """

    safe_device = _resolve_device(device)

    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=front_cam_index, width=640, height=480, fps=fps),
        "above": OpenCVCameraConfig(
            index_or_path=above_cam_index, width=640, height=480, fps=fps, fourcc="MJPG", warmup_s=2
        ),
    }
    robot_config = BiSO101FollowerConfig(
        left_arm_port=left_arm_port,
        right_arm_port=right_arm_port,
        id="bi_so101_follower",
        cameras=camera_config,
    )
    robot = BiSO101Follower(robot_config)

    policy = ACTPolicy.from_pretrained(model_id)
    policy.to(safe_device)
    if hasattr(policy, "device"):
        policy.device = safe_device  # processor側のデフォルトデバイスと揃える

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = None
    dataset_stats = None
    meta_repo = meta_repo_id or dataset_id or getattr(policy.config, "repo_id", None) or model_id
    if save_dataset:
        if dataset_id is None:
            raise ValueError("--dataset-id はデータセットを保存する場合に必須です。")
        dataset = LeRobotDataset.create(
            repo_id=dataset_id,
            fps=fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
        )
        dataset_stats = dataset.meta.stats
    else:
        meta = LeRobotDatasetMetadata(repo_id=meta_repo, force_cache_sync=False)
        dataset_stats = meta.stats

    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()

    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=model_id,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": safe_device}},
    )

    try:
        for episode_idx in range(num_episodes):
            # 停止イベントのチェック
            if stop_event is not None and stop_event.is_set():
                log_say("停止イベントを検知しました。推論を中断します。")
                break
            
            log_say(
                f"{'Recording' if save_dataset else 'Running'} inference episode {episode_idx + 1} of {num_episodes}"
            )
            if save_dataset and dataset is not None:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_proc,
                    robot_action_processor=robot_action_proc,
                    robot_observation_processor=robot_obs_proc,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=episode_time_s,
                    single_task=task_description,
                    display_data=True,
                )
                dataset.save_episode()
            else:
                start_t = time.perf_counter()
                while time.perf_counter() - start_t < episode_time_s:
                    # 停止イベントのチェック
                    if stop_event is not None and stop_event.is_set():
                        log_say("停止イベントを検知しました。推論を中断します。")
                        break
                    
                    obs_raw = robot.get_observation()
                    obs_proc = robot_obs_proc(obs_raw)
                    obs_frame = build_dataset_frame(dataset_features, obs_proc, prefix=OBS_STR)
                    action_values = predict_action(
                        observation=obs_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=task_description,
                        robot_type=robot.robot_type,
                    )
                    robot_action = make_robot_action(action_values, dataset_features)
                    action_to_send = robot_action_proc((robot_action, obs_proc))
                    robot.send_action(action_to_send)
                    precise_sleep(1 / fps)
    finally:
        robot.disconnect()
        if push_to_hub and dataset is not None:
            dataset.push_to_hub()


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する.

    Returns:
        argparse.Namespace: 解析済み引数。

    """

    parser = argparse.ArgumentParser(description="ACTポリシー推論＋評価記録（SO-101両腕）")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face Hub上のモデルID")
    parser.add_argument("--dataset-id", type=str, default=None, help="評価結果を書き込むデータセットID（保存しない場合は省略可）")
    parser.add_argument("--task", type=str, default="Grab the red grip to unfold the blanket, then gently place it.")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-s", type=int, default=60)
    parser.add_argument("--left-arm-port", type=str, default="/dev/ttyACM2")
    parser.add_argument("--right-arm-port", type=str, default="/dev/ttyACM3")
    parser.add_argument("--front-cam-index", type=int, default=4)
    parser.add_argument("--above-cam-index", type=int, default=6)
    parser.add_argument("--device", type=str, default=None, help='例: "cuda" または "cpu"')
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="動画も記録するかどうか（デフォルト: 記録する）",
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--push-to-hub", action="store_true", help="終了時にHubへpushする場合に指定")
    parser.add_argument(
        "--save-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="データセットを保存するかどうか（デフォルト: 保存する）。Falseの場合は推論のみ。",
    )
    parser.add_argument(
        "--meta-repo-id",
        type=str,
        default=None,
        help="推論のみの場合に参照するメタデータリポジトリID（未指定時はモデルやdataset-idから推定）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval_inference(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        task_description=args.task,
        num_episodes=args.num_episodes,
        fps=args.fps,
        episode_time_s=args.episode_time_s,
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        front_cam_index=args.front_cam_index,
        above_cam_index=args.above_cam_index,
        device=args.device,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
        save_dataset=args.save_dataset,
        meta_repo_id=args.meta_repo_id,
    )
