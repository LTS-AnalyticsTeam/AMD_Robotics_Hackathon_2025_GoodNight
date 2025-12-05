# 画像データ拡張ヘルパーの使い方

サブモジュールの LeRobot 側に画像変換を統合し、`ImageTransformsConfig` で解像度や拡張の有無を指定できるようにしました。`torchvision.transforms.v2` ベースの Transform を内部で組み立て、`LeRobotDataset` へそのまま渡せる Callable を提供します。`lerobot-train` もこの設定をそのまま利用します。

## 実装場所
- LeRobot 本体: `lerobot.datasets.transforms.ImageTransformsConfig` / `ImageTransforms`
- ラッパー: `processors.image_transforms`  
  - `make_train_image_transforms(enable_augmentation=True, image_size=(256, 256))`
  - `make_eval_image_transforms(image_size=(256, 256))`
  - `make_lerobot_dataset(repo_id, train=True, enable_augmentation=True, image_size=(256, 256), **dataset_kwargs)`
- トレーニング用ラッパー: `train_wrappers.train_with_local_transforms`（内部で LeRobot の Transform を利用）

## データ拡張の内容（学習時のみ）
`enable_augmentation=True` のとき、以下を順に適用します（デフォルト解像度は 256x256）。括弧内は画素への影響イメージです。
- `Resize((256, 256), antialias=True)`
- `ToDtype(torch.float32, scale=True)`（0〜1 に正規化）
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)`  
  → 明るさ/コントラスト/彩度を±10% 程度、色相を±0.02 ラジアン程度ランダムに揺らす。
- `RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.98, 1.02))`  
  → 3 度以内の回転、小さな平行移動（画素の 3%）、スケール 0.98〜1.02 を適用。
- `GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))`  
  → 3×3 カーネルで軽いボカし（シグマ 0.1〜1.0）。
- `RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3))`  
  → 2〜5% 程度の小さな矩形を 20% の確率で塗りつぶし（軽い Cutout）。

評価/推論時は `Resize` と `ToDtype` のみを適用します。

## 使い方
```python
from processors.image_transforms import (
    make_eval_image_transforms,
    make_lerobot_dataset,
    make_train_image_transforms,
)

# 1) 直接 transform を渡す場合
train_tf = make_train_image_transforms(enable_augmentation=True, image_size=(256, 256))
eval_tf = make_eval_image_transforms(image_size=(256, 256))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
train_ds = LeRobotDataset("my/repo", image_transforms=train_tf, episodes=[0])
eval_ds = LeRobotDataset("my/repo", image_transforms=eval_tf, episodes=[1])

# 2) ラッパーでまとめて生成する場合
train_ds = make_lerobot_dataset("my/repo", train=True, enable_augmentation=True, episodes=[0])
eval_ds = make_lerobot_dataset("my/repo", train=False, episodes=[1])
```

### フラグで拡張を無効化したい場合
- `enable_augmentation=False` を指定すると、学習でもリサイズ+ToDtype のみになります。
- YAML/CLI などの設定にフラグを追加する場合は、この引数をそのまま渡してください。

## `lerobot-train` での利用
`dataset.image_transforms` がそのまま `ImageTransforms` に渡されます。`enable=false` でもリサイズ+ToDtype（scale=True）は常に適用され、`enable=true` で上記拡張が追加されます。解像度は `--dataset.image_transforms.output_size="[H,W]"` で指定したサイズへリサイズされます。

### CLI 例（`lerobot-train` エントリーポイント）
```bash
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes="[0]" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.output_size="[256,256]" \
  --batch_size=2 \
  --steps=10 \
  --eval_freq=2 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --policy.type=act \
  --policy.dim_model=64 \
  --policy.n_action_steps=20 \
  --policy.chunk_size=20 \
  --policy.device=cuda \
  --env.type=aloha \
  --env.episode_length=5 \
  --wandb.enable=false
```
※ `python -m lerobot.scripts.lerobot_train` でも同じ引数で動作します。

### よく使う CLI 引数
- `--dataset.image_transforms.enable` : True で拡張オン。False でもリサイズ+ToDtype は実行。
- `--dataset.image_transforms.output_size="[H,W]"` : リサイズ先の高さ/幅。入力はこのサイズへ変換される。
- `--dataset.image_transforms.dtype=float32` : ToDtype の dtype（既定は float32）。
- `--dataset.image_transforms.apply_random_subset=true` : 拡張をランダムサブセットで適用（既定は順次適用）。
- `--dataset.image_transforms.max_num_transforms` : サブセット使用時に適用する最大個数。

### ラッパースクリプトを使う場合
`src/train_wrappers/train_with_local_transforms.py` も内部で同じ `ImageTransforms` を利用するため、従来のコマンドでも挙動は同じです。

## 環境セットアップ
- `mission2/lerobot` ディレクトリで `pip install -e .` を実行しておくと、`lerobot-train` エントリーポイントが使えます。
- Hugging Face Hub からデータを取得する場合は、`huggingface-cli login` などでトークン設定が必要です。
- 動画デコードで ffmpeg が必要な場合があります（LeRobot の通常要件に準拠）。GPU で学習する場合は CUDA/Torch 環境を整備してください。

## 設計方針
- torchvision v2 の Compose を利用し、パラメータは LeRobot 側の `ImageTransformsConfig` に集約。
- リサイズ/ToDtype（scale=True）は学習・評価ともに必ず適用し、`enable=true` のときだけ拡張を追加する。
- すべて CPU 側の標準 DataLoader ワーカーで動く軽量な変換のみを採用しています。
