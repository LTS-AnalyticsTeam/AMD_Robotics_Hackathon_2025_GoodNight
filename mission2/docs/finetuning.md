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