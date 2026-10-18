#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按卡密归属的用户素材索引。"""

from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MaterialStore:
    """素材元数据索引；实际图片文件仍由服务层保存到 uploads/。"""

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, Any]] = {}
        self.load()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        for raw in data.get("items") or []:
            if not isinstance(raw, dict):
                continue
            mid = str(raw.get("id") or "").strip()
            owner = str(raw.get("owner") or "").strip()
            file_name = Path(str(raw.get("file") or "")).name
            if not mid or not owner or not file_name:
                continue
            self._items[mid] = {
                "id": mid,
                "owner": owner,
                "file": file_name,
                "name": str(raw.get("name") or file_name),
                "size": max(0, int(raw.get("size") or 0)),
                "content_type": str(raw.get("content_type") or "application/octet-stream"),
                "created": str(raw.get("created") or ""),
            }

    def _save_locked(self) -> None:
        payload = {
            "version": self.VERSION,
            "updated": self._now(),
            "items": sorted(self._items.values(), key=lambda x: x.get("created", ""), reverse=True),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(
        self,
        *,
        owner: str,
        file_name: str,
        original_name: str,
        size: int,
        content_type: str,
    ) -> dict[str, Any]:
        owner = str(owner or "").strip()
        file_name = Path(str(file_name or "")).name
        if not owner:
            raise ValueError("素材必须有所属用户")
        if not file_name:
            raise ValueError("素材必须有存储文件名")
        record = {
            "id": "mat-" + secrets.token_hex(10),
            "owner": owner,
            "file": file_name,
            "name": Path(str(original_name or file_name)).name or file_name,
            "size": max(0, int(size)),
            "content_type": str(content_type or "application/octet-stream"),
            "created": self._now(),
        }
        with self._lock:
            self._items[record["id"]] = record
            self._save_locked()
        return dict(record)

    def get(self, material_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(str(material_id or ""))
            return dict(item) if item else None

    def list(self, owner: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._items.values())
        if owner is not None:
            items = [item for item in items if item.get("owner") == owner]
        items.sort(key=lambda x: x.get("created", ""), reverse=True)
        return [dict(item) for item in items]

    def delete(self, material_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(str(material_id or ""))
            if item is None or (owner is not None and item.get("owner") != owner):
                return None
            removed = self._items.pop(item["id"])
            self._save_locked()
            return dict(removed)

    @staticmethod
    def public(item: dict[str, Any], *, image_url: str) -> dict[str, Any]:
        return {
            "id": item["id"],
            "name": item["name"],
            "size": item["size"],
            "content_type": item["content_type"],
            "created": item["created"],
            "image_url": image_url,
        }
