"""`python -m gpu.kaggle` エントリポイント。"""

from __future__ import annotations

from gpu.kaggle.cli.app import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
