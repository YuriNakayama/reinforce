"""Kaggle Kernel (Notebook) 上の GPU 学習を起動・管理するローカル CLI 基盤。

Kaggle 公式 Python SDK の module 名 ``kaggle`` と衝突するのを避けるため、
本パッケージは ``gpu.kaggle`` として import し、SDK は ``import kaggle as
kaggle_sdk`` の alias 経由で読む規約。

Vast.ai (``gpu.vast``) / RunPod (``gpu.runpod``) と並ぶ第三の GPU 学習基盤。
Kaggle Notebooks の無料 GPU 枠 (T4x2 / P100、週 30h) を利用し、コスト 0 で
RL case を回すサイクルを実現する。

Current layout:
    - cli: Typer command wiring (train / pull / promote / status / ps / logs / watch / cost-report / dataset)
    - config: training case registry (gpu.runpod.config.cases を共有)
    - dataset: backend/ → Kaggle Dataset への snapshot upload
    - kernel: notebook (.ipynb) 自動生成 + Kaggle API kernel push / status polling
    - artifacts: launch metadata, output pull, free-hour cost-report
    - auth: KAGGLE_USERNAME / KAGGLE_KEY 解決 (env → .env → ~/.kaggle/kaggle.json)
"""
