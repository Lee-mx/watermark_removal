"""The narrow JavaScript-to-Python API and background job state."""

from __future__ import annotations

import copy
import base64
import json
import os
from pathlib import Path
import sys
import threading
import tempfile
from typing import Any
from uuid import uuid4

import cv2

from . import engine


class DesktopBridge:
    def __init__(self) -> None:
        self._window: Any = None
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._active_job: str | None = None
        override = os.environ.get("CORNER_CLEANER_DATA_DIR")
        if override:
            self._data_root = Path(override).expanduser().resolve()
        elif getattr(sys, "frozen", False):
            self._data_root = Path(os.environ.get("LOCALAPPDATA") or str(Path.home())) / "CornerCleaner"
        else:
            self._data_root = Path(__file__).resolve().parents[1] / ".app-data"
        self._config_path = self._data_root / "config.json"
        self._output_dir = self._load_output_dir()
        staging_root = self._data_root / "DropCache"
        staging_root.mkdir(parents=True, exist_ok=True)
        self._staging = tempfile.TemporaryDirectory(dir=staging_root)

    def _dispose(self) -> None:
        self._staging.cleanup()

    def _load_output_dir(self) -> str:
        fallback = Path.home() / "Pictures" / "WatermarkRemoved"
        try:
            saved = json.loads(self._config_path.read_text(encoding="utf-8"))
            if isinstance(saved.get("outputDir"), str) and saved["outputDir"]:
                return saved["outputDir"]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return str(fallback)

    def _save_output_dir(self, path: str) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(json.dumps({"outputDir": path}, ensure_ascii=False), encoding="utf-8")

    def bootstrap(self) -> dict[str, str]:
        return {"outputDir": self._output_dir}

    def choose_images(self) -> dict[str, Any]:
        import webview

        paths = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("图片 (*.png;*.jpg;*.jpeg;*.webp)",),
        )
        return self.add_paths(paths or [])

    def add_paths(self, paths: list[str] | tuple[str, ...]) -> dict[str, Any]:
        added: list[dict[str, Any]] = []
        errors: list[str] = []
        with self._lock:
            existing = {record["path"].casefold() for record in self._items.values()}
        for raw in paths:
            try:
                path = Path(raw).expanduser().resolve(strict=True)
                canonical = str(path).casefold()
                if canonical in existing:
                    continue
                metadata = engine.inspect(path)
                metadata["id"] = uuid4().hex
                with self._lock:
                    self._items[metadata["id"]] = metadata
                existing.add(canonical)
                added.append(metadata)
            except (OSError, ValueError, cv2.error) as exc:
                errors.append(f"{Path(raw).name}：{exc}")
        return {"items": added, "errors": errors}

    def import_dropped_file(self, name: str, encoded: str) -> dict[str, Any]:
        """Import a dropped browser File without relying on a browser-visible path."""
        safe_name = Path(name).name
        if not safe_name or Path(safe_name).suffix.lower() not in engine.SUPPORTED_EXTENSIONS:
            raise ValueError("拖入的文件不是受支持的图片")
        if len(encoded) > 48 * 1024 * 1024:
            raise ValueError("拖入图片不能超过 35 MB；较大的文件请点击加号选择")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("拖入图片读取失败") from exc
        if len(raw) > 35 * 1024 * 1024:
            raise ValueError("拖入图片不能超过 35 MB；较大的文件请点击加号选择")
        staged = Path(self._staging.name) / f"{uuid4().hex}_{safe_name}"
        staged.write_bytes(raw)
        payload = self.add_paths([str(staged)])
        if payload["items"]:
            payload["items"][0]["name"] = safe_name
            payload["items"][0]["outputStem"] = Path(safe_name).stem
            with self._lock:
                self._items[payload["items"][0]["id"]]["name"] = safe_name
                self._items[payload["items"][0]["id"]]["outputStem"] = Path(safe_name).stem
        else:
            staged.unlink(missing_ok=True)
        return payload

    def choose_output_dir(self) -> str | None:
        import webview

        paths = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=self._output_dir)
        if not paths:
            return None
        folder = str(Path(paths[0]).expanduser().resolve())
        self._output_dir = folder
        self._save_output_dir(folder)
        return folder

    def remove_item(self, image_id: str) -> bool:
        with self._lock:
            if self._active_job is not None:
                raise ValueError("处理期间不能移除图片")
            return self._items.pop(image_id, None) is not None

    def get_editor_preview(self, image_id: str) -> str:
        with self._lock:
            item = self._items.get(image_id)
        if item is None:
            raise ValueError("图片不存在")
        return engine.data_url(engine.read_image(Path(item["path"])), max_side=1100)

    def get_mask_preview(self, image_id: str, template_id: str) -> str | None:
        with self._lock:
            item = self._items.get(image_id)
        if item is None:
            raise ValueError("图片不存在")
        return engine.mask_preview(item["width"], item["height"], template_id)

    def start_job(self, requests: list[dict[str, Any]], output_dir: str) -> str:
        if not requests:
            raise ValueError("请先添加图片")
        if len(requests) > 500:
            raise ValueError("单次最多处理 500 张图片")
        directory = Path(output_dir).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            raise ValueError("保存路径不是文件夹")
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise ValueError("保存目录不可写")
        with self._lock:
            if self._active_job is not None:
                raise ValueError("已有任务正在运行")
            seen: set[str] = set()
            tasks: list[dict[str, Any]] = []
            for request in requests:
                image_id = request.get("id")
                if not isinstance(image_id, str) or image_id not in self._items or image_id in seen:
                    raise ValueError("图片列表已变化，请重新导入")
                seen.add(image_id)
                selection = request.get("selection")
                if not isinstance(selection, dict):
                    raise ValueError("缺少去除区域")
                item = self._items[image_id]
                engine.build_mask(item["width"], item["height"], selection)
                tasks.append({"id": image_id, "source": item["path"], "stem": item.get("outputStem"), "selection": copy.deepcopy(selection)})
            job_id = uuid4().hex
            job = {
                "id": job_id,
                "state": "running",
                "total": len(tasks),
                "completed": 0,
                "results": {task["id"]: {"state": "queued"} for task in tasks},
            }
            self._jobs[job_id] = job
            self._cancellations[job_id] = threading.Event()
            self._active_job = job_id
        self._output_dir = str(directory)
        self._save_output_dir(self._output_dir)
        worker = threading.Thread(target=self._run_job, args=(job_id, tasks, directory), daemon=True)
        worker.start()
        return job_id

    def _run_job(self, job_id: str, tasks: list[dict[str, Any]], directory: Path) -> None:
        try:
            for task in tasks:
                with self._lock:
                    if self._cancellations[job_id].is_set():
                        self._jobs[job_id]["results"][task["id"]] = {"state": "cancelled"}
                        continue
                    self._jobs[job_id]["results"][task["id"]] = {"state": "processing"}
                try:
                    output = engine.repair_to_file(Path(task["source"]), directory, task["selection"], task["stem"])
                    result = {"state": "success", **output}
                except Exception as exc:  # An individual file must not abort the batch.
                    result = {"state": "failed", "error": str(exc)}
                with self._lock:
                    self._jobs[job_id]["results"][task["id"]] = result
                    self._jobs[job_id]["completed"] += 1
        finally:
            with self._lock:
                self._jobs[job_id]["state"] = "cancelled" if self._cancellations[job_id].is_set() else "finished"
                self._active_job = None

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError("任务不存在")
            return copy.deepcopy(self._jobs[job_id])

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id != self._active_job:
                return False
            self._cancellations[job_id].set()
            return True

    def open_result(self, job_id: str, image_id: str, reveal: bool = False) -> bool:
        with self._lock:
            result = self._jobs.get(job_id, {}).get("results", {}).get(image_id, {})
            if result.get("state") != "success":
                raise ValueError("结果文件不可用")
            path = Path(result["path"])
        if not path.is_file():
            raise ValueError("结果文件已被移动")
        if reveal:
            import subprocess

            subprocess.Popen(["explorer.exe", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
        return True

    def open_output_dir(self) -> bool:
        directory = Path(self._output_dir)
        if not directory.is_dir():
            raise ValueError("保存目录还不存在")
        os.startfile(str(directory))  # type: ignore[attr-defined]
        return True
