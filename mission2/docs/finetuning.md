# ファインチューニングの実行
## smolvla_001
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps=8000 \
  --policy.scheduler_decay_lr=1e-6 \
  --dataset.repo_id=lt-s/AMD_hackathon_place_blanket \
  --batch_size=32 \
  --steps=15000 \
  --eval_freq=2000 \
  --save_freq=2000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_smolvla_001 \
  --job_name=mission2_smolvla_001 \
  --wandb.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.above": "observation.images.camera2"}'
```
## smolvla_002
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lt-s/AMD_hackathon_place_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{
    "observation.state":{"type":"STATE","shape":[12]},
    "observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},
    "observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]}
  }' \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.above": "observation.images.camera2"}' \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps=8000 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size=32 \
  --steps=12000 \
  --save_freq=2000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_smolvla_002 \
  --job_name=mission2_smolvla_002 \
  --wandb.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## smolvla_003
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lt-s/AMD_hackathon2025_merged_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{
    "observation.state":{"type":"STATE","shape":[12]},
    "observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},
    "observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]}
  }' \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.above": "observation.images.camera2"}' \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=600 \
  --policy.scheduler_decay_steps=11500 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size=32 \
  --steps=16500 \
  --save_freq=2000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_smolvla_003 \
  --job_name=mission2_smolvla_003 \
  --wandb.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## smolvla_004
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lt-s/AMD_hackathon2025_merged_all_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{
    "observation.state":{"type":"STATE","shape":[12]},
    "observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},
    "observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]}
  }' \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.above": "observation.images.camera2"}' \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1200 \
  --policy.scheduler_decay_steps=22000 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size=32 \
  --steps=32000 \
  --save_freq=3000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_smolvla_004 \
  --job_name=mission2_smolvla_004 \
  --wandb.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```

## pi05_001
```bash
lerobot-train \
  --policy.path=lerobot/pi05_base \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps=5700 \
  --policy.scheduler_decay_lr=1e-7 \
  --dataset.repo_id=lt-s/AMD_hackathon_place_blanket \
  --dataset.video_backend=pyav \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --batch_size=32 \
  --steps=6000 \
  --save_freq=1000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_pi05_001 \
  --job_name=mission2_pi05_001 \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.above":"observation.images.left_wrist_0_rgb"}'
```
## pi05_002
```bash
lerobot-train \
  --policy.path=lerobot/pi05_base \
  --policy.max_state_dim=12 \
  --policy.max_action_dim=12 \
  --policy.input_features='{
    "observation.state":{"type":"STATE","shape":[12]},
    "observation.images.base_0_rgb":{"type":"VISUAL","shape":[3,224,224]},
    "observation.images.left_wrist_0_rgb":{"type":"VISUAL","shape":[3,224,224]}
  }' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps=4700 \
  --policy.scheduler_decay_lr=1e-7 \
  --dataset.repo_id=lt-s/AMD_hackathon2025_merged_all_blanket \
  --dataset.video_backend=pyav \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --batch_size=32 \
  --steps=5000 \
  --save_freq=1000 \
  --log_freq=50 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_pi05_002 \
  --job_name=mission2_pi05_002 \
  --wandb.enable=true \
  --tensorboard.enable=true \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.above":"observation.images.left_wrist_0_rgb"}'
  ```

## act_001
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon2025_merged_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --batch_size=32 \
  --steps=6000 \
  --save_freq=1000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_001 \
  --job_name=mission2_act_001 \
  --wandb.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## act_002
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon2025_lift_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --batch_size=32 \
  --steps=6000 \
  --save_freq=1000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_002 \
  --job_name=mission2_act_002 \
  --wandb.enable=false \
  --tensorboard.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## act_003
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon_drape_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --batch_size=64 \
  --steps=6000 \
  --save_freq=1000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_003 \
  --job_name=mission2_act_003 \
  --wandb.enable=true \
  --tensorboard.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## act_004
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon_fold_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --batch_size=96 \
  --steps=4000 \
  --save_freq=800 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_004 \
  --job_name=mission2_act_004 \
  --wandb.enable=true \
  --tensorboard.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## act_005
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon_drape_blanket \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=96 \
  --steps=4000 \
  --save_freq=800 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_005 \
  --job_name=mission2_act_005 \
  --wandb.enable=true \
  --tensorboard.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```
## act_006
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon_drape_blanket_2 \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=96 \
  --steps=4000 \
  --save_freq=500 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_006 \
  --job_name=mission2_act_006 \
  --wandb.enable=true \
  --tensorboard.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```

## act_007
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lt-s/AMD_hackathon_drape_blanket_2 \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[12]}, "observation.images.front":{"type":"VISUAL","shape":[3,256,256]}, "observation.images.above":{"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=0.01 \
  --batch_size=96 \
  --steps=4000 \
  --save_freq=500 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_act_007 \
  --job_name=mission2_act_007 \
  --wandb.enable=true \
  --tensorboard.enable=true \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false
```

# データセットのマージ
```bash
lerobot-edit-dataset \
    --repo_id lt-s/AMD_hackathon2025_merged_blanket \
    --operation.type merge \
    --operation.repo_ids "['lt-s/AMD_hackathon2025_place_blanket', 'lt-s/AMD_hackathon_place_blanket_to_right']"
```

# マージ後にDataset Cardとタグを付ける手順（エラー対策）
`lerobot-train`が`meta/info.json`やコードベースのタグを見つけられずに`RevisionNotFoundError`を出す場合は、以下を実行してHub上のデータセットにREADME（Dataset Card）とバージョンタグを必ず付ける。

1. マージと同時にHubへ反映する（READMEと`meta`配下を含めてアップロードされる）。
   ```bash
   lerobot-edit-dataset \
       --repo_id lt-s/AMD_hackathon2025_merged_left_blanket \
       --operation.type merge \
       --operation.repo_ids "['lt-s/AMD_hackathon2025_place_blanket', 'lt-s/AMD_hackathon_place_blanket_addition1', 'lt-s/AMD_hackathon_place_blanket_addition2']" \
       --push_to_hub
   ```

2. 既にアップロード済みでタグだけ欠けている場合は、`meta/info.json`内の`codebase_version`（例: `v3.0`）でHubにタグを作成する。
   ```bash
   python - <<'PY'
   from huggingface_hub import HfApi

   repo_id = "lt-s/AMD_hackathon2025_merged_blanket"
   tag = "v3.0"  # info.jsonのcodebase_versionに合わせる
   HfApi().create_tag(repo_id, tag=tag, repo_type="dataset")
   PY
   ```

3. 手元のキャッシュに`meta/info.json`があるか確認する（無ければ再マージ or 再ダウンロード）。
   ```bash
   ls ~/.cache/huggingface/lerobot/lt-s/AMD_hackathon2025_merged_blanket/meta
   ```
```
## pi05_002
```bash
lerobot-train \
  --policy.path=lerobot/pi05_base \
  --policy.max_action_dim=12 \
  --policy.output_features='{"action":{"type":"ACTION","shape":[12]}}' \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=300 \
  --policy.scheduler_decay_steps=5700 \
  --policy.scheduler_decay_lr=1e-7 \
  --dataset.repo_id=lt-s/AMD_hackathon_place_blanket \
  --dataset.video_backend=pyav \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --batch_size=32 \
  --steps=6000 \
  --save_freq=1000 \
  --log_freq=100 \
  --policy.device=cuda \
  --output_dir=outputs/train/mission2_pi05_001 \
  --job_name=mission2_pi05_001 \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.above":"observation.images.left_wrist_0_rgb"}'
```
