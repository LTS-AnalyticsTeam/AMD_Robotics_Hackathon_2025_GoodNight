import numpy as np
import streamlit as st

# ページ設定
st.set_page_config(page_title="Robot Control", layout="wide")

st.title("🤖 TEAM13_LTS Robotics_Team：GoodNight")


# レイアウト: 左側カメラ、右側ボタン
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 ライブビュー")

    # カメラ画面プレイスホルダー
    camera_placeholder = st.empty()

    # 一時的な画像（実際はカメラから取得）
    # 実際のカメラ使用時: cap = cv2.VideoCapture(0)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    camera_placeholder.image(dummy_image, channels="RGB", use_container_width=True)

with col2:
    st.subheader("🎮 制御ボタン")

    st.write("")  # 間隔
    st.write("")

    # ボタン1
    if st.button("▶️ 布団掛け", key="start", use_container_width=True, type="primary"):
        st.success("✅ start ")
        # ここにLeRobot推論コードを追加
        # observation = get_observation()
        # output = model.predict(observation)

    st.write("")  # 間隔

    # ボタン2
    if st.button("⏹️ 布団を敷く", key="stop", use_container_width=True):
        st.warning("🛑 start")
        # ここに停止コードを追加

    st.write("")
    st.write("")

    # 状態表示
    st.info("💡 状態: 待機中")

# 下部情報
st.divider()
st.caption("LeRobot Control Interface v1.0")
