# 推論の実行サンプル

## SmolVLA での推論
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SmolVLA推論スクリプト

SmolVLAの学習済みモデルをRTC付きでロードし、1チャンクのアクションを生成する最小例。
実運用ではraw_obsをセンサーから得た最新観測に置き換えてください。
"""

import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors


def main():
    """SmolVLAで1チャンク推論を実行する.

    この例ではRTCを有効にし、`predict_action_chunk`にRTC用引数を渡して
    平滑なチャンク接続を行う。

    Returns:
        None: 返り値は標準出力のみ。
    """
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
    policy.init_rtc_processor()

    preproc, postproc = make_smolvla_pre_post_processors(config=policy.config, dataset_stats=None)

    raw_obs = {
        "observation.images.front": torch.rand(3, 480, 640),  # 例: 単眼カメラ1枚
        "observation.state": torch.zeros(policy.config.max_state_dim),
        "task": "Move the blanket to the blue bin.",
    }

    batch = preproc(raw_obs)
    actions = policy.predict_action_chunk(
        batch,
        inference_delay=4,           # 推論レイテンシに合わせて調整
        prev_chunk_left_over=None,   # 直前チャンクの未実行部分があれば渡す
        execution_horizon=None,      # 未指定ならRTCConfigの値を利用
    )
    decoded = postproc(actions)
    print(decoded["action"])


if __name__ == "__main__":
    main()
```

## Pi05 での推論
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pi05推論スクリプト

π₀.₅の学習済みモデルをRTC付きでロードし、1チャンクのアクションを生成する最小例。
"""

import torch

from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def main():
    """Pi05で1チャンク推論を実行する.

    Returns:
        None: 返り値は標準出力のみ。
    """
    policy = PI05Policy.from_pretrained("lerobot/pi05_base")
    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
    policy.init_rtc_processor()

    preproc, postproc = make_pi05_pre_post_processors(config=policy.config, dataset_stats=None)

    raw_obs = {
        "observation.images.front": torch.rand(3, 480, 640),
        "observation.state": torch.zeros(policy.config.max_state_dim),
        "task": "Put the object on the blanket.",
    }

    batch = preproc(raw_obs)
    actions = policy.predict_action_chunk(
        batch,
        inference_delay=6,           # ハードウェア実測に合わせて設定
        prev_chunk_left_over=None,
        execution_horizon=None,
    )
    decoded = postproc(actions)
    print(decoded["action"])


if __name__ == "__main__":
    main()
```
