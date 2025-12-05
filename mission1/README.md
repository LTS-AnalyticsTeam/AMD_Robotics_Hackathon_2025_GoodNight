# 仮想環境の作成とアクティベート
```bash
# 環境の作成
conda create -y -n lerobot-mission1 python=3.10

# アクティベート
conda activate lerobot-mission1
conda install ffmpeg -c conda-forge
pip install -e ".[feetech]"
```

# テレオペレート
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so_101_follower_03 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so_101_leader_03
```
```bash
lerobot-teleoperate     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=so_101_follower_03     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=so_101_leader_03  --robot.cameras='{wrist: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},}' --display_data=True
```

# レコーディング
```bash
lerobot-record     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=so_101_follower_03     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=so_101_leader_03  --robot.cameras='{wrist: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},}' --display_data=True  --dataset.repo_id=lt-s/AMD_hackathon2025_mission1  --dataset.num_episodes=30  --dataset.single_task="Lift and place the screwdriver"
```