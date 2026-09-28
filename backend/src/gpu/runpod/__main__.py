"""`uv run python -m gpu.runpod` のエントリポイント。"""

from __future__ import annotations

from gpu.runpod.cli.app import app

if __name__ == "__main__":
    app()
