#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""渠道（Space）管理
==================

一个「渠道」就是一个 Hugging Face Space。每个 Space 是**独立队列**，
所以多挂几个能绕开排队；控制台里可以随时开关，开关立即生效不用重启。

状态存 ``channels.json``::

    {"version": 1, "channels": [{"space": "...", "enabled": true, "note": "主站"}]}

探测复用同一套逻辑（``probe_space``），``probe_spaces.py`` 也调它。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

#: 本地生成接口的签名，用来判断某个 Space 能不能直接替换
REF_PARAMS = ["prompt", "mode", "reference", "aspect_ratio", "steps", "seed", "randomize_seed"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def space_url(space_id: str) -> str:
    """``owner/name`` → ``https://owner-name.hf.space``（点、下划线、斜杠都转横线，全小写）。"""
    slug = (
        space_id.strip()
        .replace("/", "-")
        .replace("_", "-")
        .replace(".", "-")
        .lower()
    )
    return f"https://{slug}.hf.space"


# ==========================================================================
# 探测
# ==========================================================================


def probe_space(space: str, *, proxy: str | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """探测一个 Space：运行状态、是否暴露 /generate、参数与画幅值是否兼容。

    只读操作，不会触发生成、不消耗配额。
    """
    out: dict[str, Any] = {"space": space, "url": space_url(space)}
    try:
        with httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True) as c:
            meta = c.get(f"https://huggingface.co/api/spaces/{space}").json()
            rt = meta.get("runtime") or {}
            out["stage"] = rt.get("stage", "?")
            out["likes"] = meta.get("likes", 0)
            out["sdk"] = meta.get("sdk")
            sub = meta.get("subdomain")
            if not sub:
                out["error"] = "无 subdomain（Space 可能不存在）"
                out["compatible"] = False
                return out
            base = f"https://{sub}.hf.space"
            out["url"] = base

            cfg_r = c.get(base + "/config")
            out["config"] = cfg_r.status_code
            if cfg_r.status_code != 200:
                out["compatible"] = False
                out["error"] = f"config 返回 {cfg_r.status_code}"
                return out

            info = c.get(base + "/gradio_api/info").json()
            eps = info.get("named_endpoints") or {}
            endpoint = ""
            for name in ("/generate", "/infer", "/run", "/predict"):
                if name in eps:
                    endpoint = name
                    out["endpoint"] = name
                    out["params"] = [
                        p.get("parameter_name") for p in (eps[name].get("parameters") or [])
                    ]
                    break

            cfg = cfg_r.json()
            comps = {comp.get("id"): comp for comp in (cfg.get("components") or [])}
            for dep in cfg.get("dependencies") or []:
                if dep.get("api_name") not in ("generate", "/generate"):
                    continue
                for cid in dep.get("inputs") or []:
                    ch = ((comps.get(cid) or {}).get("props") or {}).get("choices")
                    if ch:
                        vals = [str(x[0]) if isinstance(x, (list, tuple)) else str(x) for x in ch]
                        if any("\u00b7" in v for v in vals):
                            out["aspects"] = vals
                            break
                break

            params = out.get("params") or []
            if not endpoint:
                out["compatible"] = False
                out["reason"] = "没有 /generate 端点"
            elif params != REF_PARAMS:
                out["compatible"] = False
                out["reason"] = f"参数不同（{len(params)} 个）"
            else:
                out["compatible"] = True
                out["reason"] = "可直接用"
    except Exception as exc:  # noqa: BLE001
        out["compatible"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:110]}"
    return out


def resolve_probe_proxy() -> str | None:
    """探测用的出口。优先 QWEN_PROBE_PROXY，其次系统 env，最后交给调用方。"""
    for key in ("QWEN_PROBE_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.getenv(key, "").strip()
        if v:
            return v
    return None


# ==========================================================================
# 渠道
# ==========================================================================


@dataclass
class Channel:
    space: str
    enabled: bool = True
    note: str = ""
    created: str = ""

    # 运行时探测结果（不落盘，重启后重新测）
    stage: str = ""
    likes: int = 0
    compatible: bool | None = None
    reason: str = ""
    error: str = ""
    checked_at: float = 0.0

    @property
    def url(self) -> str:
        return space_url(self.space)

    def public(self) -> dict[str, Any]:
        return {
            "space": self.space,
            "url": self.url,
            "enabled": self.enabled,
            "note": self.note,
            "created": self.created,
            "stage": self.stage,
            "likes": self.likes,
            "compatible": self.compatible,
            "reason": self.reason,
            "error": self.error,
            "checked_at": (
                datetime.fromtimestamp(self.checked_at, timezone.utc).isoformat()
                if self.checked_at
                else None
            ),
        }


class ChannelStore:
    """渠道表。默认从 ``SPACE_IDS`` 初始化，之后以 ``channels.json`` 为准。"""

    VERSION = 1

    def __init__(self, path: str | Path, defaults: list[str] | None = None) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._items: dict[str, Channel] = {}
        if not self.load() and defaults:
            for sid in defaults:
                self._items[sid] = Channel(space=sid, enabled=True, created=_now_iso())
            self.save()

    # ---------- 持久化 ----------

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False
        rows = data.get("channels") or []
        for raw in rows:
            space = (raw.get("space") or "").strip()
            if not space:
                continue
            self._items[space] = Channel(
                space=space,
                enabled=bool(raw.get("enabled", True)),
                note=raw.get("note", ""),
                created=raw.get("created", ""),
            )
        return bool(self._items)

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": self.VERSION,
                "updated": _now_iso(),
                "channels": [
                    {"space": c.space, "enabled": c.enabled, "note": c.note, "created": c.created}
                    for c in self._items.values()
                ],
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2))

    # ---------- 查询 ----------

    def list(self) -> list[Channel]:
        with self._lock:
            return list(self._items.values())

    def get(self, space: str) -> Channel | None:
        with self._lock:
            return self._items.get(space.strip())

    def enabled_spaces(self) -> list[str]:
        with self._lock:
            return [c.space for c in self._items.values() if c.enabled]

    def __len__(self) -> int:
        return len(self._items)

    # ---------- 变更 ----------

    def add(self, space: str, *, note: str = "", enabled: bool = True) -> Channel:
        sid = space.strip()
        with self._lock:
            ch = self._items.get(sid)
            if ch is None:
                ch = Channel(space=sid, enabled=enabled, note=note, created=_now_iso())
                self._items[sid] = ch
            else:
                ch.enabled = enabled
                if note:
                    ch.note = note
        self.save()
        return ch

    def toggle(self, space: str, enabled: bool) -> Channel | None:
        sid = space.strip()
        with self._lock:
            ch = self._items.get(sid)
            if ch is None:
                return None
            ch.enabled = bool(enabled)
        self.save()
        return ch

    def remove(self, space: str) -> bool:
        sid = space.strip()
        with self._lock:
            if self._items.pop(sid, None) is None:
                return False
        self.save()
        return True

    def set_probe(self, space: str, result: dict[str, Any]) -> None:
        """把一次探测结果记到渠道上（只在内存，不落盘）。"""
        with self._lock:
            ch = self._items.get(space.strip())
            if ch is None:
                return
            ch.stage = result.get("stage", "")
            ch.likes = int(result.get("likes") or 0)
            ch.compatible = result.get("compatible")
            ch.reason = result.get("reason", "")
            ch.error = result.get("error", "")
            ch.checked_at = time.time()
