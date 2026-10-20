"""Package the built React UI and Python backend as a Windows folder app."""

from pathlib import Path
import argparse
import sys

DESKTOP = Path(__file__).resolve().parent
PROJECT = DESKTOP.parent
sys.path[:0] = [str(DESKTOP / ".vendor"), str(PROJECT / ".deps")]

from PyInstaller.__main__ import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distpath", type=Path, default=DESKTOP / "release")
    args = parser.parse_args()
    if not (DESKTOP / "dist" / "index.html").is_file():
        raise SystemExit("请先在 desktop 目录运行 npm run build")
    run(
        [
            "--noconfirm",
            "--clean",
            "--onedir",
            "--windowed",
            "--name=水印消除",
            f"--icon={DESKTOP / 'assets' / 'app_icon.ico'}",
            f"--distpath={args.distpath.resolve()}",
            f"--paths={PROJECT / '.deps'}",
            f"--paths={DESKTOP / '.vendor'}",
            f"--add-data={DESKTOP / 'dist'};dist",
            f"--add-data={DESKTOP / 'assets' / 'app_icon.ico'};assets",
            f"--add-data={PROJECT / 'tools' / 'watermark_mask_960x1280.png'};assets",
            f"--add-data={PROJECT / 'tools' / 'watermark_mask_11578066187.png'};assets",
            f"--add-data={PROJECT / 'tools' / 'watermark_mask_267777923.png'};assets",
            f"--add-data={DESKTOP / 'models' / 'inpainting_lama_2025jan.onnx'};assets",
            f"--add-data={DESKTOP / 'models' / 'LICENSE'};assets",
            str(DESKTOP / "run_app.py"),
        ]
    )
