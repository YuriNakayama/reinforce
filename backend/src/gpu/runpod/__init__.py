"""RunPod 上の GPU 学習を起動・管理するためのローカル CLI 基盤。

RunPod 公式 SDK の module 名 ``runpod`` と衝突するのを避けるため、内部
パッケージ名は ``runpod_io`` とし、SDK は ``import runpod as runpod_sdk`` の
alias 経由で読む規約。

Current layout:
    - cli: Typer command wiring and command implementations
    - config: training case registry and path helpers
    - runpod: RunPod SDK / API adapters
    - execution: onstart template, S3 progress, watcher, run summaries
    - artifacts: launch metadata, run metadata, cost reports
    - auth / git / notify: small cross-cutting helpers
"""
