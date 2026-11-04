"""Проверяет явный состав публичного комплекта, не читая закрытые каталоги"""

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {"data", "reports", ".local", "tmp", "models", ".claude"}
PRIVATE_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".parquet",
    ".joblib",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".zip",
    ".docx",
    ".db",
    ".sqlite",
}
SECRET_PATTERNS = [
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
]


def inspect_path(name):
    """Отклоняет приватные пути до чтения содержимого"""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return [f"{name}: недопустимый путь"]
    if path.parts[0] in PRIVATE_ROOTS or path.suffix.lower() in PRIVATE_SUFFIXES:
        return [f"{name}: приватный путь или формат"]
    if path.name.startswith(".env"):
        return [f"{name}: файл окружения с возможными секретами"]
    return []


def inspect_content(name, content, entry):
    """Проверяет один файл, не выводя его содержимое в журнал"""
    errors = inspect_path(name)
    if errors:
        return errors
    path = PurePosixPath(name)
    if entry.get("sha256") and hashlib.sha256(content).hexdigest() != entry["sha256"]:
        errors.append(
            f"{name}: изменен проверенный материал, требуется повторное ревью"
        )
    if path.suffix in {".pdf", ".png"}:
        if not entry.get("sha256"):
            errors.append(
                f"{name}: бинарный материал без проверенной контрольной суммы"
            )
        return errors
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return errors + [f"{name}: неизвестный бинарный формат"]
    if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
        errors.append(f"{name}: обнаружен возможный секрет")
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(text)
            if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
                errors.append(
                    f"{name}: непроверенные метаданные ноутбука, включая возможные виджеты"
                )
            for i, cell in enumerate(notebook["cells"]):
                if (
                    cell.get("outputs")
                    or cell.get("execution_count") is not None
                    or cell.get("attachments")
                ):
                    errors.append(
                        f"{name}: ячейка {i} содержит выходы, счетчик или вложение"
                    )
                if cell.get("metadata"):
                    errors.append(
                        f"{name}: ячейка {i} содержит непроверенные метаданные"
                    )
        except (ValueError, KeyError, TypeError):
            errors.append(f"{name}: неверный формат ноутбука")
    if path.suffix == ".csv":
        if not name.startswith("supplementary/tables/") or not entry.get("sha256"):
            errors.append(f"{name}: CSV не включен в проверенные агрегаты")
        rows = list(csv.reader(io.StringIO(text)))
        if (
            not rows
            or rows[0] != entry.get("columns")
            or len(rows) - 1 != entry.get("rows")
        ):
            errors.append(f"{name}: структура таблицы не совпадает с проверенной")
        forbidden = {
            "n пп",
            "иб",
            "фио",
            "patient_id",
            "дата рождения",
            "дата текущего дка",
        }
        if rows and any(c.lower().strip() in forbidden for c in rows[0]):
            errors.append(f"{name}: столбец с идентификатором пациента")
    return errors


def git_paths(root, index):
    args = ["git", "ls-files", "-z"]
    if not index:
        args += ["--cached", "--others", "--exclude-standard"]
    result = subprocess.run(args, cwd=root, check=True, capture_output=True)
    return set(filter(None, result.stdout.decode("utf-8").split("\0")))


def check(root=ROOT, index=False, files_only=False):
    """Возвращает ошибки и манифест; режим index читает staged-версию"""
    manifest_name = "publication_manifest.json"
    if index:
        raw = subprocess.check_output(["git", "show", ":" + manifest_name], cwd=root)
    else:
        raw = (root / manifest_name).read_bytes()
    manifest = json.loads(raw)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    errors = []
    if len(entries) != len(manifest["files"]):
        errors.append("Повтор пути в манифесте")
    expected = set(entries) | {manifest_name}
    if not files_only:
        candidates = git_paths(root, index)
        for extra in sorted(candidates - expected):
            errors.append(f"{extra}: файл вне публичного списка")
        if index:
            for missing in sorted(expected - candidates):
                errors.append(f"{missing}: отсутствует в индексе")
    for name, entry in entries.items():
        path_errors = inspect_path(name)
        if path_errors:
            errors.extend(path_errors)
            continue
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"{name}: ссылка или путь за пределами проекта")
            continue
        try:
            if index:
                content = subprocess.check_output(
                    ["git", "show", ":" + name], cwd=root, stderr=subprocess.DEVNULL
                )
            else:
                content = path.read_bytes()
        except (OSError, subprocess.CalledProcessError):
            errors.append(f"{name}: файл недоступен")
            continue
        errors.extend(inspect_content(name, content, entry))
        if name.endswith(".md"):
            text = content.decode("utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                linked = (path.parent / target.split("#")[0]).resolve()
                if not linked.is_relative_to(root.resolve()):
                    errors.append(f"{name}: ссылка за пределы публичного комплекта")
                elif linked.relative_to(root.resolve()).as_posix() not in expected:
                    errors.append(f"{name}: ссылка на непубличный файл {target}")
    return errors, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", action="store_true", help="проверить подготовленный git index"
    )
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="проверить распакованный комплект без Git",
    )
    args = parser.parse_args()
    errors, manifest = check(index=args.index, files_only=args.files_only)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(
        f"Проверено публичных файлов: {len(manifest['files']) + 1}. Запрещенных материалов не найдено."
    )


if __name__ == "__main__":
    main()
