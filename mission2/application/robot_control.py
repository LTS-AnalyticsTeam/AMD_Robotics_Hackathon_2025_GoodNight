#!/usr/bin/python
# -*- coding: utf-8 -*-
"""StreamlitベースのSO-101制御ダッシュボード.

カメラプレビューとLeRobot推論シーケンスの開始・停止をWeb UIから操作できるようにする。

Todo:
    * 例外発生時に詳細ログを表示するビューを追加する。
    * ステータス更新をWebSocket等で非同期化する。

"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

sys.path.append(str(Path(__file__).resolve().parents[2]))

CAMERA_ID = 6
# PREVIEW_WIDTH = 960
# PREVIEW_HEIGHT = 720
PREVIEW_WIDTH = 2180
PREVIEW_HEIGHT = 1440
INFERENCE_SCRIPT_PATH = Path(__file__).with_name("so-101_callAPI.py")


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
    if "inference_module" not in st.session_state:
        st.session_state.inference_module = None

def _load_inference_module() -> object:
    """so-101推論スクリプトを動的ロードする.

    Returns:
        object: `run_inference` 関数を含むモジュール。

    Raises:
        FileNotFoundError: スクリプトが存在しない場合に発生。
        ImportError: モジュールのロードに失敗した場合に発生。

    """

    if st.session_state.inference_module is not None:
        return st.session_state.inference_module

    if not INFERENCE_SCRIPT_PATH.exists():
        raise FileNotFoundError("推論スクリプトが見つかりません。")

    spec = importlib.util.spec_from_file_location("so_101_callAPI", INFERENCE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("推論スクリプトをロードできません。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    st.session_state.inference_module = module
    return module


def _start_inference() -> None:
    """布団掛けタスクの推論をバックグラウンドで開始する.

    Returns:
        None: 返り値はない。

    """

    if st.session_state.inference_running:
        return
    try:
        module = _load_inference_module()
    except (ImportError, FileNotFoundError) as exc:
        st.error(str(exc))
        return

    if not hasattr(module, "run_inference"):
        st.error("推論スクリプトに run_inference が定義されていません。")
        return

    stop_event = threading.Event()

    inference_thread = threading.Thread(
        target=module.run_inference,
        kwargs={"stop_event": stop_event},
        daemon=True,
        name="so101-inference",
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

    stop_event: Optional[threading.Event] = st.session_state.inference_stop_event
    inference_thread: Optional[threading.Thread] = st.session_state.inference_thread
    if stop_event is not None:
        stop_event.set()
    if inference_thread is not None and inference_thread.is_alive():
        inference_thread.join(timeout=5)
    st.session_state.inference_stop_event = None
    st.session_state.inference_thread = None
    st.session_state.inference_running = False


def main() -> None:
    """Streamlitアプリケーションのエントリーポイント.

    Returns:
        None: 返り値はない。

    """

    st.set_page_config(page_title="Robot Control", layout="wide")
    _init_session_state()

    st.title("🤖 TEAM13_LTS Robotics_Team：GoodNight")

    st.subheader("📹 ライブビュー")
    camera_status_placeholder = st.empty()

    webrtc_ctx = webrtc_streamer(
        key="camera-preview",
        mode=WebRtcMode.SENDONLY,
        video_processor_factory=CameraVideoProcessor,
        media_stream_constraints={"video": {"width": PREVIEW_WIDTH, "height": PREVIEW_HEIGHT}, "audio": False},
    )

    if webrtc_ctx.state.playing:
        camera_status_placeholder.success("プレビュー実行中")
    else:
        camera_status_placeholder.info("プレビューの初期化中です。")

    st.subheader("🎮 制御ボタン")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ 布団掛け開始", key="inference_start", use_container_width=True):
            _start_inference()

    with col_stop:
        if st.button("⏹️ 布団掛け停止", key="inference_stop", use_container_width=True):
            _stop_inference()

    inference_state = "実行中" if st.session_state.inference_running else "待機中"
    st.info(f"🤖 推論: {inference_state}")

    st.divider()
    st.caption("LeRobot Control Interface v1.0")


if __name__ == "__main__":
    main()
