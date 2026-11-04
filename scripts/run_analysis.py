"""Выполняет анализ локально, сохраняя выходы вне публичного ноутбука"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if not config.RAW_DATASET.is_file():
        parser.error(
            "Закрытая база не найдена. См. docs/data_dictionary.md и docs/reproduction.md"
        )
    if args.check_only:
        print("База найдена. Расчеты не запускались.")
        return

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook = nbformat.read(ROOT / "notebooks/pipeline.ipynb", as_version=4)
    executor = ExecutePreprocessor(timeout=args.timeout, kernel_name="dka")
    output = ROOT / "reports/pipeline_executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    finally:
        # Сохраняем диагностику и при прерывании расчета
        nbformat.write(notebook, output)
    print(f"Выполненный ноутбук сохранен локально: {output}")


if __name__ == "__main__":
    main()
