"""Создает локальный архив только из проверенного списка файлов"""

import zipfile
from datetime import datetime, timezone

from check_publication import ROOT, check


def main():
    errors, manifest = check()
    if errors:
        raise SystemExit("\n".join(errors))
    output = ROOT / ".local/releases"
    output.mkdir(parents=True, exist_ok=True)
    name = (
        "dka-publication-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + ".zip"
    )
    path = output / name
    paths = [entry["path"] for entry in manifest["files"]] + [
        "publication_manifest.json"
    ]
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(paths):
            archive.write(ROOT / relative, relative)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(paths):
            raise RuntimeError("Ошибка проверки архива")
    print(f"Локальный комплект: {path}")


if __name__ == "__main__":
    main()
