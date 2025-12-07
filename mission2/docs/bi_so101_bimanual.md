# SO-101 両腕（バイマニュアル）クイックスタート

このリポジトリは LeRobot に外部モジュールとして SO-101 の両腕対応を追加します。実装本体は `src/robots` と `src/teleoperators` に置き、LeRobot が自動検出できるようにプラグインブリッジ（`lerobot_teleoperator_bi_so101` / `lerobot_robot_bi_so101`）を同梱しています。

## 追加ファイル
- `src/robots/bi_so101_follower.py` / `src/robots/config_bi_so101_follower.py`: 2 本の `SO101Follower` を 1 台のロボットとしてまとめた `bi_so101_follower`。
- `src/teleoperators/bi_so101_leader.py` / `src/teleoperators/config_bi_so101_leader.py`: 2 本の `SO101Leader` をまとめたテレオペレーター `bi_so101_leader`。
- テスト: `tests/robots/test_bi_so101_follower.py`（モックバス）。

## セットアップ（プラグイン導入）
```bash
# リポジトリルートで
conda activate lerobot-v042  # 任意の環境名
pip install -e ./lerobot     # 上流 LeRobot を editable で導入
pip install -e .             # 本リポの両腕プラグインを導入
```
インストール後は `PYTHONPATH` や `--teleop.discover_packages_path` を毎回指定する必要はありません。

## テレオペ（リーダー → フォロワー）
1. SO-101 リーダー 2 本とフォロワー 2 本を USB 接続し、ポート名を控える（例 `/dev/ttyACM*` / `/dev/ttyUSB*`）。左右を取り違えないようラベルを付ける。
2. （任意）単腕スクリプトで各腕を事前キャリブレーションしておくと初回の手間が減ります。
3. 両腕テレオペを実行:
```bash
lerobot-teleoperate \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/ttyACM0 \
  --teleop.right_arm_port=/dev/ttyACM1 \
  --teleop.id=bi_so101_leader \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM3 \
  --robot.id=bi_so101_follower \
  --fps=30
```
- 角度モードを使いたい場合:
  `--teleop.left_arm_use_degrees=true --teleop.right_arm_use_degrees=true`
  を付け、ロボット側も `--robot.left_arm_use_degrees=true --robot.right_arm_use_degrees=true` で揃える。
- カメラを使う場合は通常どおり `--robot.cameras.cam0.width ...` などを付与。

## レコーディング
### ブランケットをかける
```bash
lerobot-record \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/ttyACM0 \
  --teleop.right_arm_port=/dev/ttyACM1 \
  --teleop.id=bi_so101_leader \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM3 \
  --robot.id=bi_so101_follower \
  --robot.cameras='{front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}, above: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc":"MJPG", "warmup_s":2}}' \
  --dataset.repo_id=lt-s/AMD_hackathon_drape_blanket_2 \
  --dataset.single_task="Lift the blanket from the doll's neck. Fold the blanket and place it gently next to the doll." \
  --dataset.num_episodes=50
```
### ブランケットをめくる
```bash
lerobot-record \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/ttyACM0 \
  --teleop.right_arm_port=/dev/ttyACM1 \
  --teleop.id=bi_so101_leader \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM3 \
  --robot.id=bi_so101_follower \
  --robot.cameras='{front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}, above: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc":"MJPG", "warmup_s":2}}' \
  --dataset.repo_id=lt-s/AMD_hackathon_lift_blanket \
  --dataset.single_task="Grab the red loop and gently lift the blanket up, then fold it in thirds and place it at the doll's feet." \
  --dataset.num_episodes=50 \
  --dataset.push_to_hub=false
```

## 推論
### smolvla
```bash
lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM3 \
  --robot.id=bi_so101_follower \
  --robot.cameras='{camera1: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}, camera2: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc":"MJPG", "warmup_s":2}}' \
  --dataset.repo_id=lt-s/eval_AMD_hackathon_place_blanket \
  --dataset.single_task="Grab the red grip to unfold the blanket, then gently place the blanket over the doll." \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=50 \
  --dataset.push_to_hub=false \
  --policy.path=lt-s/AMD_hackathon2025_blanket_smolvla_010000
```
### pi05
```bash
lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM3 \
  --robot.id=bi_so101_follower \
  --robot.cameras='{base_0_rgb: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}, left_wrist_0_rgb: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc":"MJPG", "warmup_s":2}}' \
  --dataset.repo_id=lt-s/eval_AMD_hackathon_place_blanket \
  --dataset.single_task="Grab the red grip to unfold the blanket, then gently place the blanket over the doll." \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=50 \
  --dataset.push_to_hub=false \
  --policy.path=lt-s/AMD_hackathon2025_blanket_pi05_all_004000
```
### act
```bash
lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/ttyACM2 \
  --robot.right_arm_port=/dev/ttyACM3 \
  --robot.id=bi_so101_follower \
  --robot.cameras='{front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}, above: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30, "fourcc":"MJPG", "warmup_s":2}}' \
  --dataset.repo_id=lt-s/eval_AMD_hackathon_place_blanket \
  --dataset.single_task="Grab the red grip to unfold the blanket, then gently place the blanket over the doll." \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=50 \
  --dataset.push_to_hub=false \
  --policy.path=lt-s/AMD_hackathon2025_blanket_act_fold_001600
```

## 安全メモ
- 両腕とも安全なニュートラル姿勢から開始する。
- 大きなジャンプを防ぐため相対目標を制限:
  `--robot.left_arm_max_relative_target 15 --robot.right_arm_max_relative_target 15`
- いつでも停止できるよう非常停止手段を用意。`Ctrl+C` でループ終了。

## テスト（モック）
```bash
pytest tests/robots/test_bi_so101_follower.py
```
