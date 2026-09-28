"""Vast.ai 上の GPU 学習を起動・管理するためのローカル CLI 基盤。

Public API:
    - run_meta: gpu.common.run_meta の re-export
      (RunMetadata / run_id / config hash / JSON I/O)
    - auth: AWS credentials and VAST_API_KEY loaders
    - offers: vastai SDK の search_offers ラッパ
    - instance: onstart テンプレ render と create_instance
    - cost: run.json 集計と markdown レポート
    - cli: typer ベースのサブコマンド
"""
