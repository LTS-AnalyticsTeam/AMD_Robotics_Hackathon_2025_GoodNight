#!/usr/bin/python
# -*- coding: utf-8 -*-
"""StreamlitベースのSO-101制御ダッシュボード.

カメラプレビューとLeRobot推論シーケンスの開始・停止をWeb UIから操作できるようにする。

Todo:
    * 例外発生時に詳細ログを表示するビューを追加する。
    * ステータス更新をWebSocket等で非同期化する。

"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

sys.path.insert(0, str(Path(__file__).parent.parent))

from act_inference_control import run_eval_inference

# カメラ設定
CAMERA_ID = 6
PREVIEW_WIDTH = 2180
PREVIEW_HEIGHT = 1440


@dataclass
class RobotInferenceConfig:
    """ロボット推論の設定を保持するデータクラス.

    モデルごとに変更が必要なパラメータと、固定パラメータを管理する。

    Attributes:
        model_id (str): Hugging Face Hub上のモデルID。
        meta_repo_id (str): メタデータリポジトリID。
        task (str): タスクの説明文。
        dataset_id (Optional[str]): 評価結果を保存するデータセットID（保存しない場合はNone）。
        num_episodes (int): 実行するエピソード数。
        fps (int): 制御および記録のFPS。
        episode_time_s (int): 1エピソードの制御時間（秒）。
        left_arm_port (str): 左腕フォロワーのシリアルポート。
        right_arm_port (str): 右腕フォロワーのシリアルポート。
        front_cam_index (int): 正面カメラのデバイスID。
        above_cam_index (int): 上部カメラのデバイスID。
        device (str): 使用デバイス（"cuda" または "cpu"）。
        use_videos (bool): データセット保存時に動画を含めるかどうか。
        image_writer_threads (int): 動画保存のスレッド数。
        push_to_hub (bool): 実行終了時にHubへpushするかどうか。
        save_dataset (bool): データセットを保存するかどうか。Falseの場合は推論のみ。

    """

    model_id: str
    meta_repo_id: str
    task: str
    dataset_id: Optional[str] = None
    num_episodes: int = 1
    fps: int = 30
    episode_time_s: int = 60
    left_arm_port: str = "/dev/ttyACM2"
    right_arm_port: str = "/dev/ttyACM3"
    front_cam_index: int = 4
    above_cam_index: int = 6
    device: str = "cuda"
    use_videos: bool = False
    image_writer_threads: int = 4
    push_to_hub: bool = False
    save_dataset: bool = False


### 各タスクのモデル設定
# 布団をかけるタスクモデル
DRAPE_BLANKET_CONFIG = RobotInferenceConfig(
    model_id="lt-s/AMD_hackathon2025_blanket_act_drape_004000",
    meta_repo_id="lt-s/AMD_hackathon_drape_blanket",
    task="Grab the red grip to unfold the blanket, then gently place it.",
)

# 布団を外すタスクモデル
REMOVE_BLANKET_CONFIG = RobotInferenceConfig(
    model_id="lt-s/AMD_hackathon2025_blanket_act_fold_001600",
    meta_repo_id="lt-s/AMD_hackathon_fold_blanket",
    task="Lift the blanket from the doll's neck. Fold the blanket and place it gently next to the doll.",
)


def _open_capture(camera_id: int) -> cv2.VideoCapture:
    """指定されたIDでOpenCVのVideoCaptureを初期化する.

    Args:
        camera_id (int): 利用するカメラのデバイスID。

    Returns:
        cv2.VideoCapture: 初期化済みのキャプチャインスタンス。

    Raises:
        RuntimeError: キャプチャをオープンできなかった場合に発生。

    """

    capture = cv2.VideoCapture(camera_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
    if not capture.isOpened():
        raise RuntimeError(f"カメラID {camera_id} をオープンできません。")
    return capture


class CameraVideoProcessor(VideoProcessorBase):
    """WebRTC用にOpenCVのカメラフレームを提供するプロセッサ."""

    def __init__(self) -> None:
        self.capture: Optional[cv2.VideoCapture] = None

    def _ensure_capture(self) -> cv2.VideoCapture:
        if self.capture is None or not self.capture.isOpened():
            if self.capture is not None:
                self.capture.release()
            self.capture = _open_capture(CAMERA_ID)
        return self.capture

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """WebRTC経由で送信するフレームを生成する."""

        try:
            capture = self._ensure_capture()
            ret, image = capture.read()
        except RuntimeError:
            self.capture = None
            ret, image = False, None

        if not ret or image is None:
            fallback = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(fallback, format="bgr24")

        resized = cv2.resize(image, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_AREA)
        return av.VideoFrame.from_ndarray(resized, format="bgr24")

    def __del__(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()


def _init_session_state() -> None:
    """Streamlitセッション状態に必要なキーを初期化する.

    Returns:
        None: 返り値はない。

    """

    if "inference_thread" not in st.session_state:
        st.session_state.inference_thread = None
    if "inference_stop_event" not in st.session_state:
        st.session_state.inference_stop_event = None
    if "inference_running" not in st.session_state:
        st.session_state.inference_running = False


def _start_inference(config: RobotInferenceConfig) -> None:
    """指定された設定で推論をバックグラウンドで開始する.

    Args:
        config (RobotInferenceConfig): 推論設定。

    Returns:
        None: 返り値はない。

    """

    if st.session_state.inference_running:
        st.warning("既に推論が実行中です。")
        return

    stop_event = threading.Event()

    def inference_wrapper():
        """推論を実行し、停止イベントに対応するラッパー関数."""
        try:
            run_eval_inference(
                model_id=config.model_id,
                dataset_id=config.dataset_id,
                task_description=config.task,
                num_episodes=config.num_episodes,
                fps=config.fps,
                episode_time_s=config.episode_time_s,
                left_arm_port=config.left_arm_port,
                right_arm_port=config.right_arm_port,
                front_cam_index=config.front_cam_index,
                above_cam_index=config.above_cam_index,
                device=config.device,
                use_videos=config.use_videos,
                image_writer_threads=config.image_writer_threads,
                push_to_hub=config.push_to_hub,
                save_dataset=config.save_dataset,
                meta_repo_id=config.meta_repo_id,
                stop_event=stop_event,
            )
        except Exception as e:
            st.error(f"推論中にエラーが発生しました: {e}")
        finally:
            st.session_state.inference_running = False

    inference_thread = threading.Thread(
        target=inference_wrapper,
        daemon=True,
        name="robot-inference",
    )
    st.session_state.inference_stop_event = stop_event
    st.session_state.inference_thread = inference_thread
    st.session_state.inference_running = True
    inference_thread.start()


def _stop_inference() -> None:
    """推論スレッドを停止し、後処理を行う.

    Returns:
        None: 返り値はない。

    """

    if not st.session_state.inference_running:
        st.info("推論は実行されていません。")
        return

    stop_event: Optional[threading.Event] = st.session_state.inference_stop_event
    inference_thread: Optional[threading.Thread] = st.session_state.inference_thread
    
    if stop_event is not None:
        stop_event.set()
        st.info("停止シグナルを送信しました。ロボットが安全に停止するまでお待ちください...")
    
    if inference_thread is not None and inference_thread.is_alive():
        inference_thread.join(timeout=10)
        if inference_thread.is_alive():
            st.warning("推論スレッドが10秒以内に停止しませんでした。強制終了します。")
    
    st.session_state.inference_stop_event = None
    st.session_state.inference_thread = None
    st.session_state.inference_running = False
    st.success("推論を停止しました。")


def main() -> None:
    """Streamlitアプリケーションのエントリーポイント.

    Returns:
        None: 返り値はない。

    """

    st.set_page_config(page_title="Robot Control", layout="wide")
    _init_session_state()

    st.title("🤖 TEAM13_LTS Robotics_Team：GoodNight")

    st.subheader("📹 ライブビュー")

    webrtc_ctx = webrtc_streamer(
        key="camera-preview",
        mode=WebRtcMode.SENDONLY,
        video_processor_factory=CameraVideoProcessor,
        media_stream_constraints={"video": {"width": PREVIEW_WIDTH, "height": PREVIEW_HEIGHT}, "audio": False},
    )

    st.subheader("🎮 制御ボタン")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 布団掛け開始", key="drape_start", use_container_width=True):
            _start_inference(DRAPE_BLANKET_CONFIG)

    with col2:
        if st.button("🔄 布団外し開始", key="remove_start", use_container_width=True):
            _start_inference(REMOVE_BLANKET_CONFIG)

    if st.button("⏹️ 停止", key="inference_stop", use_container_width=True):
        _stop_inference()

    inference_state = "実行中" if st.session_state.inference_running else "待機中"
    st.info(f"🤖 推論: {inference_state}")

    st.divider()
    st.caption("LeRobot Control Interface v1.0")


if __name__ == "__main__":
    main()
