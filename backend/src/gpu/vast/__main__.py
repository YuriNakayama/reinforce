"""`uv run python -m gpu.vast` のエントリポイント。"""

from __future__ import annotations

from gpu.vast.cli import app

if __name__ == "__main__":
    app()
