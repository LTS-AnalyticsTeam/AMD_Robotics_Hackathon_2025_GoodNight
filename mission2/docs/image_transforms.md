# 画像データ拡張ヘルパーの使い方

本リポジトリでは、サブモジュールの LeRobot 本体には手を入れず、プロジェクト側で学習用/評価用の画像変換をまとめて定義しています。`torchvision.transforms.v2` を利用し、`LeRobotDataset` の `image_transforms` 引数にそのまま渡せる Callable を提供します。

## 実装場所
- `processors.image_transforms` モジュール  
  - `make_train_image_transforms(enable_augmentation=True, image_size=(256, 256))`
  - `make_eval_image_transforms(image_size=(256, 256))`
  - `make_lerobot_dataset(repo_id, train=True, enable_augmentation=True, image_size=(256, 256), **dataset_kwargs)`
- トレーニング用ラッパー: `train_wrappers.train_with_local_transforms`  
  - `lerobot-train` の CLI をそのまま受け取り、データセット生成だけをローカル変換に差し替える。

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
from robots.image_transforms import (
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

## `lerobot-train` で有効にするには？
標準の `lerobot-train` CLI は内部で `ImageTransformsConfig`（LeRobot 標準設定）を使うため、Compose をそのまま CLI で渡すことはできません。最小改変で本ヘルパーを使いたい場合は、ラッパーを用います。

### ラッパースクリプトの例
- ファイル: `src/train_wrappers/train_with_local_transforms.py`
- 実行例:
  ```bash
  python -m train_wrappers.train_with_local_transforms \
    --policy.type=act \
    --policy.dim_model=64 \
    --policy.n_action_steps=20 \
    --policy.chunk_size=20 \
    --policy.device=cuda \
    --env.type=aloha \
    --env.episode_length=5 \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --dataset.episodes="[0]" \
    --dataset.image_transforms.enable=true \
    --batch_size=2 \
    --steps=10 \
    --eval_freq=2 \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --log_freq=1 \
    --wandb.enable=false
  ```
- ポイント:
  - `lerobot-train` と同じ CLI をそのまま渡す。
  - `--dataset.image_transforms.enable=true` がオンなら学習データに本ヘルパーの拡張を適用。False ならリサイズ+ToDtype のみ。
  - データセット生成のみ差し替え、他の処理（ポリシー構築・学習ループ・保存）は既存の `lerobot-train` に依存する。


## 設計方針
- torchvision v2 の Compose をそのまま利用し、パラメータはこのモジュールに一元化。
- LeRobot サブモジュールの `ImageTransformsConfig` には手を加えず、呼び出し側から `image_transforms` を差し込む形。
- すべて CPU 側の標準 DataLoader ワーカーで動く軽量な変換のみを採用しています。
