#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Image 2.1 GGUF Studio —— Web 服务
======================================

把 qwen_client.py 包成 HTTP 服务 + 一个单页前端，附带 HF 账号池管理面板。

启动::

    python server.py                     # http://127.0.0.1:7860
    python server.py --port 8000 --host 0.0.0.0
    python server.py --concurrency 1     # 默认 1，和官方 Space 的 concurrency_limit=1 对齐

环境变量::

    QWEN_HF_TOKENS        账号池，格式 name:token,name2:token2
    QWEN_QUOTA_COOLDOWN_H 配额冷却小时数，默认到下一个 UTC 零点
    QWEN_CONCURRENCY      并发数，默认 1
    QWEN_DATA_DIR         运行时数据目录，默认与源码同目录
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from auth import (
    STATUS_CN,
    STATUS_VALID,
    AdminStore,
    LicenseStore,
    load_or_create_secret,
    normalize_key,
    sign_token,
    verify_token,
)
from channels import ChannelStore, probe_space, resolve_probe_proxy
from materials import MaterialStore
import fingerprint

from qwen_client import (
    API_NAME,
    DEFAULT_ASPECT,
    DEFAULT_MODE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STRATEGY,
    MODES,
    PROMPT_MAX,
    SIZES,
    STEPS_MAX,
    STEPS_MIN,
    STRATEGIES,
    MAX_REFS,
    MAX_SEED,
    SPACE_ID,
    SPACE_IDS,
    SPACE_SRC_MODE,
    SPACE_URL,
    AccountPool,
    Cancelled,
    norm_refs,
    PoolExhausted,
    ProxyPool,
    QwenError,
    QwenStudio,
    ValidationError,
    _parse_one_proxy,
    normalize_proxy,
    probe_proxy,
    proxy_scheme_ok,
    validate_params,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("qwen.server")

ROOT = Path(__file__).parent
DATA_ROOT = Path(os.getenv("QWEN_DATA_DIR", str(ROOT))).expanduser().resolve()
UPLOAD_DIR = DATA_ROOT / "uploads"
OUTPUT_DIR = DATA_ROOT / "outputs"
TOKEN_FILE = DATA_ROOT / "tokens.txt"
POOL_FILE = TOKEN_FILE  # 运行时可被 --tokens 覆盖，账号池的读写都走它
PROXY_FILE = DATA_ROOT / "proxies.txt"  # 运行时可被 --proxies 覆盖
HISTORY_MAX = 200
UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 官方 max_file_size="10mb"
GALLERY_FILE = DATA_ROOT / "gallery.json"
JOBS_FILE = DATA_ROOT / "jobs.json"
MATERIALS_FILE = DATA_ROOT / "materials.json"
CHANNELS_FILE = DATA_ROOT / "channels.json"
MATERIAL_ADMIN_OWNER = "__admin__"

# ---------- 鉴权相关 ----------
LICENSE_FILE = DATA_ROOT / "licenses.json"
ADMIN_FILE = DATA_ROOT / ".admin.json"
SECRET_FILE = DATA_ROOT / ".session.key"
ADMIN_PW_FILE = DATA_ROOT / "admin_password.txt"
STATIC_DIR = ROOT / "static"
SESSION_COOKIE = "qw_session"
SESSION_TTL_ADMIN = 12 * 3600          # 管理员会话 12 小时
SESSION_TTL_USER = 7 * 24 * 3600       # 卡密用户会话 7 天
#: 开发用：完全关掉鉴权（不要在有公网入口时开）
NO_AUTH = os.getenv("QWEN_NO_AUTH", "").strip().lower() in ("1", "true", "yes", "on")

DATA_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ==========================================================================
# 任务模型
# ==========================================================================


def _job_refs(params: dict[str, Any]) -> list[str]:
    """任务里存的参考图列表。兼容老任务（只有单个 reference 字段）。"""
    refs = params.get("references")
    if refs:
        return list(refs)
    single = params.get("reference")
    return [single] if single else []


@dataclass
class Job:
    id: str
    params: dict[str, Any]
    status: str = "queued"          # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    account: str | None = None
    license_key: str | None = None      # 这次生成花的是哪张卡密
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created": datetime.fromtimestamp(self.created, timezone.utc).isoformat(),
            "started": (
                datetime.fromtimestamp(self.started, timezone.utc).isoformat() if self.started else None
            ),
            "finished": (
                datetime.fromtimestamp(self.finished, timezone.utc).isoformat() if self.finished else None
            ),
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 2),
            "params": {k: v for k, v in self.params.items() if k not in ("reference", "references")},
            "has_reference": bool(_job_refs(self.params)),
            "reference_count": len(_job_refs(self.params)),
            "progress": self.progress,
            "account": self.account,
            "license": self.license_key,
            "error": self.error,
            "result": (
                {
                    "seed": self.result["seed"],
                    "details": self.result["details"],
                    "elapsed": self.result["elapsed"],
                    "width": self.result["width"],
                    "height": self.result["height"],
                    "account": self.result["account"],
                    "route": self.result.get("route"),
                    "proxy": self.result.get("proxy"),
                    "image_url": f"/api/image/{self.id}",
                    "download_name": Path(self.result["image_path"]).name,
                }
                if self.result
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """落盘用：保留完整 result（含本地路径），恢复后还能看图。"""
        return {
            "id": self.id,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "params": self.params,
            "progress": self.progress,
            "account": self.account,
            "license_key": self.license_key,
            "error": self.error,
            "result": self.result,
        }


class JobStore:
    """任务表。可选落盘到 jobs.json —— 服务重启后仍能看到「上次那个任务怎么了」。"""

    def __init__(self, cap: int = HISTORY_MAX, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._cap = cap
        self.path = path
        self.load()

    # ---------- 持久化 ----------

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("jobs.json 读不出来，忽略（不影响新任务）")
            return
        restored = 0
        for raw in data.get("jobs") or []:
            try:
                job = Job(id=raw["id"], params=raw.get("params") or {})
                job.status = raw.get("status", "error")
                job.created = float(raw.get("created") or time.time())
                job.started = raw.get("started")
                job.finished = raw.get("finished")
                job.progress = raw.get("progress") or {}
                job.account = raw.get("account")
                job.license_key = raw.get("license_key")
                job.error = raw.get("error")
                job.result = raw.get("result")
            except (KeyError, TypeError, ValueError):
                continue
            # 重启前还在跑的，gradio 那边的句柄已经没了，标成中断而不是假装还在跑
            if job.status in ("running", "queued"):
                job.status = "interrupted"
                job.error = "服务重启，任务已中断，请重新生成"
                job.finished = job.finished or time.time()
            self._jobs[job.id] = job
            self._order.append(job.id)
            restored += 1
        self._order = self._order[-self._cap:]
        if restored:
            log.info("任务表已恢复 %d 条（进行中的已标记为中断）", restored)

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            items = [self._jobs[i].to_dict() for i in self._order if i in self._jobs]
        payload = {
            "version": 1,
            "updated": datetime.now(timezone.utc).isoformat(),
            "jobs": items[-100:],
        }
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            log.warning("写入 jobs.json 失败（不影响运行）")

    # ---------- 增删查 ----------

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._cap:
                old = self._order.pop(0)
                self._jobs.pop(old, None)
        self.save()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            ids = list(reversed(self._order[-limit:]))
            return [self._jobs[i] for i in ids if i in self._jobs]

    def clear(self) -> int:
        with self._lock:
            n = len(self._jobs)
            self._jobs.clear()
            self._order.clear()
        self.save()
        return n

    def stats(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {"total": len(self._jobs)}
            for j in self._jobs.values():
                out[j.status] = out.get(j.status, 0) + 1
            return out


def _aspect_from_size(size: tuple[int, int]) -> str:
    """反查官方 SIZES，得到与接口一致的画幅键。"""
    for key, wh in SIZES.items():
        if tuple(wh) == tuple(size):
            return key
    return ""


def _details_from_meta(meta: dict[str, Any], size: tuple[int, int]) -> str:
    """按官方 details 的格式，从 PNG 内嵌元数据拼一句摘要。"""
    bits = [f"{size[0]} \u00d7 {size[1]}"]
    if meta.get("steps"):
        bits.append(f"{meta['steps']} steps")
    if meta.get("seconds"):
        bits.append(f"{meta['seconds']}s")
    if meta.get("seed") is not None:
        bits.append(f"seed {meta['seed']}")
    if meta.get("checkpoint"):
        bits.append(str(meta["checkpoint"]))
    return " \u00b7 ".join(bits)


def _gallery_record(job: Job, res: dict[str, Any]) -> dict[str, Any]:
    """把一次成功的生成写成画廊条目。"""
    return {
        "id": job.id,
        "created": datetime.now(timezone.utc).isoformat(),
        "prompt": job.params["prompt"],
        "mode": job.params["mode"],
        "aspect_ratio": job.params["aspect_ratio"],
        "steps": job.params["steps"],
        "seed": res["seed"],
        "requested_seed": job.params["seed"],
        "randomize_seed": job.params["randomize_seed"],
        "width": res["width"],
        "height": res["height"],
        "account": res["account"],
        "route": res.get("route", "token"),
        "proxy": res.get("proxy"),
        "license": job.license_key,
        "elapsed": res["elapsed"],
        "details": res["details"],
        "file": Path(res["image_path"]).name,
        "has_reference": bool(_job_refs(job.params)),
        "reference_count": len(_job_refs(job.params)),
        "source": "generate",
    }


def _gallery_public(it: dict[str, Any]) -> dict[str, Any]:
    out = dict(it)
    out["image_url"] = f"/api/gallery/{it['id']}/image"
    out["download_url"] = f"/api/gallery/{it['id']}/download"
    out["file_exists"] = bool(it.get("file")) and (OUTPUT_DIR / it["file"]).exists()
    return out


class Gallery:
    """生成画廊：索引持久化到 gallery.json，记录提示词与全部参数。

    条目里的 file 是 outputs/ 下的文件名，图片本身留在磁盘上；
    索引只是元数据，坏了或删了都不会丢图。
    """

    VERSION = 1

    def __init__(self, index_file: Path, images_dir: Path) -> None:
        self._lock = threading.Lock()
        self.index_file = index_file
        self.images_dir = images_dir
        self._items: dict[str, dict[str, Any]] = {}
        self.load()

    # ---------- 持久化 ----------

    def load(self) -> None:
        if not self.index_file.exists():
            return
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
            for it in data.get("items", []):
                if it.get("id"):
                    self._items[it["id"]] = it
            log.info("画廊已加载 %d 条记录", len(self._items))
        except Exception:  # noqa: BLE001
            log.exception("画廊索引损坏，已忽略（图片文件不受影响）")

    def _save_locked(self) -> None:
        payload = {
            "version": self.VERSION,
            "updated": datetime.now(timezone.utc).isoformat(),
            "items": sorted(
                self._items.values(), key=lambda x: x.get("created", ""), reverse=True
            ),
        }
        tmp = self.index_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_file)

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    # ---------- 增删查 ----------

    def add(self, rec: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._items[rec["id"]] = rec
            self._save_locked()
        return rec

    def get(self, gid: str) -> dict[str, Any] | None:
        with self._lock:
            it = self._items.get(gid)
            return dict(it) if it else None

    def list(
        self, *, q: str = "", mode: str = "", license: str = "", limit: int = 200, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            items = sorted(
                self._items.values(), key=lambda x: x.get("created", ""), reverse=True
            )
        if mode:
            items = [x for x in items if x.get("mode") == mode]
        if license:
            items = [x for x in items if x.get("license") == license]
        if q:
            needle = q.strip().lower()
            items = [
                x
                for x in items
                if needle in (x.get("prompt") or "").lower()
                or needle in (x.get("details") or "").lower()
                or needle == str(x.get("seed"))
                or needle in (x.get("account") or "").lower()
            ]
        total = len(items)
        return items[offset : offset + max(0, limit)], total

    def delete(self, gid: str, *, remove_file: bool = False) -> bool:
        with self._lock:
            it = self._items.pop(gid, None)
            if it is None:
                return False
            self._save_locked()
        if remove_file and it.get("file"):
            try:
                (self.images_dir / it["file"]).unlink(missing_ok=True)
            except OSError:
                log.warning("删除图片文件失败：%s", it.get("file"))
        return True

    def clear(self, *, remove_files: bool = False) -> int:
        with self._lock:
            n = len(self._items)
            files = [it.get("file") for it in self._items.values()]
            self._items.clear()
            self._save_locked()
        if remove_files:
            for f in files:
                if f:
                    try:
                        (self.images_dir / f).unlink(missing_ok=True)
                    except OSError:
                        pass
        return n

    # ---------- 从 outputs 目录补录 ----------

    def scan(self) -> dict[str, int]:
        """扫描 outputs/*.png，把内嵌的 generation 元数据读回来补录。

        官方生成的 PNG 里本来就带完整参数，所以画廊建立之前的老图也能补进来。
        """
        added = skipped = failed = 0
        known = {it.get("file") for it in self._items.values()}
        new_records: list[dict[str, Any]] = []
        for p in sorted(self.images_dir.glob("*.png")):
            if p.name in known:
                skipped += 1
                continue
            try:
                with Image.open(p) as im:
                    raw = im.info.get("generation")
                    size = (im.size[0], im.size[1])
                meta = json.loads(raw) if raw else {}
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            gid = "scan-" + hashlib.sha1(p.name.encode("utf-8")).hexdigest()[:12]
            new_records.append(
                {
                    "id": gid,
                    "created": datetime.fromtimestamp(
                        p.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "prompt": meta.get("prompt", ""),
                    "mode": meta.get("mode", ""),
                    "aspect_ratio": _aspect_from_size(size),
                    "steps": meta.get("steps"),
                    "seed": meta.get("seed"),
                    "randomize_seed": None,
                    "width": size[0],
                    "height": size[1],
                    "account": "(历史文件)",
                    "elapsed": meta.get("seconds"),
                    "details": _details_from_meta(meta, size),
                    "file": p.name,
                    "has_reference": False,
                    "source": "scan",
                }
            )
            added += 1
        if new_records:
            with self._lock:
                for r in new_records:
                    self._items[r["id"]] = r
                self._save_locked()
        return {"added": added, "skipped": skipped, "failed": failed}


# ==========================================================================
# 应用
# ==========================================================================

app = FastAPI(title="Qwen Image 2.1 GGUF Studio", version="2.0.0")
# 前后端同源，不需要放开跨域；开着 "*" 只会给 CSRF 留口子
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

proxy_pool = ProxyPool.from_env(proxy_file=PROXY_FILE)
pool = AccountPool.from_env(token_file=TOKEN_FILE, proxy_pool=proxy_pool)
# 渠道表：默认用 SPACE_IDS 初始化，之后以 channels.json 为准（控制台里开关）
channel_store = ChannelStore(CHANNELS_FILE, defaults=SPACE_IDS)
studio = QwenStudio(pool, out_dir=OUTPUT_DIR, spaces=channel_store.enabled_spaces())
store = JobStore(path=JOBS_FILE)
gallery = Gallery(GALLERY_FILE, OUTPUT_DIR)
materials = MaterialStore(MATERIALS_FILE)

# 鉴权三件套
secret = load_or_create_secret(SECRET_FILE)
admin_store = AdminStore(ADMIN_FILE)
licenses = LicenseStore(LICENSE_FILE)

CONCURRENCY = int(os.getenv("QWEN_CONCURRENCY", "1"))
executor = ThreadPoolExecutor(max_workers=max(1, CONCURRENCY), thread_name_prefix="qwen")
# Space 端 concurrency_limit=1，本地再串行化一层，避免把请求堆到服务端队列里
gpu_lock = threading.Semaphore(1)


class GenerateReq(BaseModel):
    prompt: str
    mode: str = DEFAULT_MODE
    #: 旧字段：单张参考图。新前端传 references，两个都收，references 优先。
    reference: str | None = None
    #: 多张参考图（最多 MAX_REFS=10），顺序即 prompt 里的「图1/图2…」
    references: list[str] | None = None
    #: 素材库中的参考图 ID，服务端会按当前用户权限解析为本地路径
    material_ids: list[str] | None = None
    aspect_ratio: str = DEFAULT_ASPECT
    steps: float = DEFAULT_STEPS
    seed: int = DEFAULT_SEED
    randomize_seed: bool = True


class PoolAddReq(BaseModel):
    name: str = ""
    token: str
    note: str = ""


class PoolNameReq(BaseModel):
    name: str | None = None


class StrategyReq(BaseModel):
    strategy: str


class ProxyAddReq(BaseModel):
    url: str
    label: str = ""
    note: str = ""
    days: int | None = None       # 到期天数，None = 永久


class ProxyImportReq(BaseModel):
    """批量导入：一次粘贴多行，格式见 README。文件走 /api/proxy/import-files。"""

    text: str = ""
    days: int | None = None       # 这一批的到期天数，None = 永久


class ProxyRefReq(BaseModel):
    """代理定位：url 或 label 都行，都为空表示「全部」。"""

    url: str | None = None
    label: str | None = None
    limit: int = 50       # 批量测试时最多测几个（代理池可能上千，全测要跑很久）


# ---------- 登录 / 卡密 请求模型 ----------


class LoginReq(BaseModel):
    mode: str = "license"     # license | admin
    key: str = ""
    password: str = ""


class LicenseCreateReq(BaseModel):
    count: int = 1
    quota: int | None = 10    # None = 不限次数
    days: int | None = 30     # None = 永不过期
    note: str = ""


class LicenseUpdateReq(BaseModel):
    key: str
    note: str | None = None
    quota: int | None = None
    set_quota: bool = False
    days: int | None = None
    extend_days: int | None = None


class LicenseKeyReq(BaseModel):
    key: str
    disabled: bool | None = None


class PasswordReq(BaseModel):
    current: str = ""
    password: str


# ---------- 鉴权依赖 ----------


def _session(request: Request) -> dict[str, Any] | None:
    """从 cookie 或 Authorization: Bearer 里取会话，验签并查过期。"""
    tok = request.cookies.get(SESSION_COOKIE) or ""
    if not tok:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            tok = auth[7:].strip()
    return verify_token(secret, tok) if tok else None


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def current_user(request: Request) -> dict[str, Any]:
    """任意已登录身份（管理员或卡密用户）。"""
    if NO_AUTH:
        return {"role": "admin", "key": None, "no_auth": True}
    sess = _session(request)
    if not sess:
        raise HTTPException(401, "请先登录")
    return sess


def require_admin(request: Request) -> dict[str, Any]:
    sess = current_user(request)
    if sess.get("role") != "admin":
        raise HTTPException(403, "需要管理员权限")
    return sess


def require_license(request: Request) -> dict[str, Any]:
    """卡密用户或管理员。卡密每次请求都复核状态 —— 停用/过期能立刻生效。"""
    sess = current_user(request)
    if sess.get("role") == "admin":
        return sess
    lic = licenses.get(sess.get("key") or "")
    if lic is None:
        raise HTTPException(401, "卡密不存在或已被删除")
    st = lic.status()
    if st != STATUS_VALID:
        raise HTTPException(403, f"卡密{STATUS_CN.get(st, st)}")
    return {**sess, "license": lic}


def _set_session_cookie(resp: Response, token: str, ttl: int) -> None:
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=ttl,
        httponly=True,
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(SESSION_COOKIE, path="/")


# ---------- 任务执行 ----------


def _run_job(job: Job) -> None:
    job.status = "running"
    job.started = time.time()
    job.progress = {"state": "starting", "elapsed": 0}

    def on_progress(ev: dict[str, Any]) -> None:
        job.progress = ev

    try:
        with gpu_lock:
            res = studio.generate(
                prompt=job.params["prompt"],
                mode=job.params["mode"],
                reference=_job_refs(job.params),
                aspect_ratio=job.params["aspect_ratio"],
                steps=job.params["steps"],
                seed=job.params["seed"],
                randomize_seed=job.params["randomize_seed"],
                on_progress=on_progress,
                should_cancel=job.cancel.is_set,
            )
        job.result = res
        job.account = res["account"]
        try:
            gallery.add(_gallery_record(job, res))
        except Exception:  # noqa: BLE001
            log.exception("写入画廊失败（不影响出图）")
        job.status = "done"
    except Cancelled as exc:
        job.status = "cancelled"
        job.error = str(exc)
    except ValidationError as exc:
        job.status = "error"
        job.error = str(exc)
    except PoolExhausted as exc:
        job.status = "error"
        job.error = str(exc)
    except QwenError as exc:
        job.status = "error"
        job.error = str(exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("任务 %s 异常", job.id)
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished = time.time()
        job.progress = {**job.progress, "state": job.status}
        # 失败/取消要把次数退回去 —— 预扣是为了防并发超发，不是真要收钱
        if job.status in ("error", "cancelled") and job.license_key:
            if licenses.refund(job.license_key):
                log.info("任务 %s 未成功，已退回 1 次（%s）", job.id, job.license_key)
        store.save()   # 落盘，页面刷新/服务重启后还能看到这次任务的下场


# ---------- 页面 ----------


def _page(name: str) -> FileResponse:
    p = ROOT / name
    if not p.exists():
        raise HTTPException(404, f"{name} 不存在")
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


@app.get("/")
def root(request: Request):
    if NO_AUTH:
        return RedirectResponse("/admin", 302)
    sess = _session(request)
    if not sess:
        return RedirectResponse("/login", 302)
    return RedirectResponse("/admin" if sess.get("role") == "admin" else "/app", 302)


@app.get("/login")
def page_login() -> FileResponse:
    return _page("login.html")


@app.get("/app")
def page_app(request: Request):
    if not NO_AUTH and not _session(request):
        return RedirectResponse("/login", 302)
    return _page("index.html")


@app.get("/admin")
def page_admin(request: Request):
    sess = _session(request)
    if not NO_AUTH and (not sess or sess.get("role") != "admin"):
        return RedirectResponse("/login", 302)
    return _page("admin.html")


# ---------- 登录 / 登出 / 我是谁 ----------


@app.post("/api/auth/login")
def api_login(req: LoginReq, request: Request, response: Response) -> dict[str, Any]:
    ip = _client_ip(request)

    if req.mode == "admin":
        if not admin_store.check(req.password or ""):
            log.warning("管理员登录失败 ip=%s", ip)
            raise HTTPException(401, "口令不正确")
        token = sign_token(secret, {"role": "admin"}, SESSION_TTL_ADMIN)
        _set_session_cookie(response, token, SESSION_TTL_ADMIN)
        log.info("管理员登录成功 ip=%s", ip)
        return {"ok": True, "role": "admin", "expires_in": SESSION_TTL_ADMIN}

    key = normalize_key(req.key or "")
    if not key:
        raise HTTPException(400, "请输入卡密")
    lic = licenses.get(key)
    if lic is None:
        raise HTTPException(404, "卡密不存在")
    st = lic.status()
    if st != STATUS_VALID:
        raise HTTPException(403, f"卡密{STATUS_CN.get(st, st)}")

    token = sign_token(secret, {"role": "user", "key": lic.key}, SESSION_TTL_USER)
    _set_session_cookie(response, token, SESSION_TTL_USER)
    log.info("卡密登录 %s ip=%s 剩余=%s", lic.key, ip, lic.remaining())
    return {
        "ok": True,
        "role": "user",
        "key": lic.key,
        "note": lic.note,
        "remaining": lic.remaining(),
        "quota": lic.quota,
        "used": lic.used,
        "expires": lic.expires,
        "expires_in": SESSION_TTL_USER,
    }


@app.post("/api/auth/logout")
def api_logout(response: Response) -> dict[str, Any]:
    _clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/auth/me")
def api_me(request: Request) -> dict[str, Any]:
    """前端每次加载都问一次：我是不是登录了、还剩多少次、什么时候过期。"""
    if NO_AUTH:
        return {"authenticated": True, "role": "admin", "no_auth": True}
    sess = _session(request)
    if not sess:
        return {"authenticated": False}
    if sess.get("role") == "admin":
        return {"authenticated": True, "role": "admin"}
    lic = licenses.get(sess.get("key") or "")
    if lic is None:
        return {"authenticated": False, "reason": "卡密已删除"}
    st = lic.status()
    return {
        "authenticated": True,
        "role": "user",
        "key": lic.key,
        "note": lic.note,
        "remaining": lic.remaining(),
        "quota": lic.quota,
        "used": lic.used,
        "expires": lic.expires,
        "status": st,
        "status_cn": STATUS_CN.get(st, st),
        "usable": st == STATUS_VALID,
    }


@app.post("/api/auth/password")
def api_change_password(req: PasswordReq, sess: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    if not NO_AUTH and not admin_store.check(req.current or ""):
        raise HTTPException(401, "当前口令不正确")
    try:
        admin_store.set_password(req.password)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        ADMIN_PW_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    log.info("管理员口令已修改")
    return {"ok": True}


# ---------- 基础接口 ----------


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return {
        "space": SPACE_ID,
        "space_url": SPACE_URL,
        "space_src_mode": SPACE_SRC_MODE,
        "api_name": API_NAME,
        "modes": MODES,
        "sizes": {k: list(v) for k, v in SIZES.items()},
        "steps_min": STEPS_MIN,
        "steps_max": STEPS_MAX,
        "prompt_max": PROMPT_MAX,
        "max_seed": MAX_SEED,
        "defaults": {
            "mode": DEFAULT_MODE,
            "aspect_ratio": DEFAULT_ASPECT,
            "steps": DEFAULT_STEPS,
            "seed": DEFAULT_SEED,
            "randomize_seed": True,
        },
        "concurrency": CONCURRENCY,
        "upload_max_mb": UPLOAD_MAX_BYTES // (1024 * 1024),
        "max_refs": MAX_REFS,
        "strategy": pool.strategy,
        "strategies": list(STRATEGIES),
        "proxy_total": len(proxy_pool),
        "examples": [
            {
                "label": "霓虹招牌",
                "prompt": 'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
                "mode": "Create an image",
            },
            {
                "label": "浮岛天文台",
                "prompt": "A tiny observatory on a floating island above clouds, warm sunset, intricate storybook illustration",
                "mode": "Create an image",
            },
            {
                "label": "卡通龙贴纸（透明）",
                "prompt": "A cute cartoon dragon sticker, jade green scales, a friendly expression, clean edges",
                "mode": "Transparent PNG",
            },
        ],
    }


# ---------- 账号池 ----------


@app.get("/api/pool", dependencies=[Depends(require_admin)])
def api_pool() -> dict[str, Any]:
    return pool.snapshot()


@app.post("/api/pool/add", dependencies=[Depends(require_admin)])
def api_pool_add(req: PoolAddReq) -> dict[str, Any]:
    token = (req.token or "").strip()
    if not token:
        raise HTTPException(400, "token 不能为空")
    if any(a.token == token for a in pool.accounts):
        raise HTTPException(409, "该 token 已在池中")
    acc = pool.add(req.name.strip(), token, req.note.strip())
    _persist_pool()
    log.info("账号池新增：%s", acc.name)
    return {"ok": True, "name": acc.name, "pool": pool.snapshot()}


@app.post("/api/pool/remove", dependencies=[Depends(require_admin)])
def api_pool_remove(req: PoolNameReq) -> dict[str, Any]:
    if not req.name:
        raise HTTPException(400, "name 不能为空")
    if not pool.remove(req.name):
        raise HTTPException(404, f"账号不存在：{req.name}")
    _persist_pool()
    return {"ok": True, "pool": pool.snapshot()}


@app.post("/api/pool/clear_cooldown", dependencies=[Depends(require_admin)])
def api_pool_clear(req: PoolNameReq) -> dict[str, Any]:
    n = pool.clear_cooldown(req.name)
    return {"ok": True, "cleared": n, "pool": pool.snapshot()}


@app.post("/api/pool/reload", dependencies=[Depends(require_admin)])
def api_pool_reload() -> dict[str, Any]:
    global pool, studio
    fresh = AccountPool.from_env(
        token_file=POOL_FILE, proxy_pool=proxy_pool, strategy=pool.strategy
    )
    # 保留运行时状态：同 token 的计数与冷却沿用
    old = {a.token: a for a in pool.accounts if a.token}
    for a in fresh.accounts:
        prev = old.get(a.token)
        if prev:
            a.exhausted_until = prev.exhausted_until
            a.success, a.failure = prev.success, prev.failure
            a.last_error, a.last_used = prev.last_error, prev.last_used
    pool.replace_accounts(fresh.accounts)
    studio.pool = pool
    return {"ok": True, "pool": pool.snapshot()}


@app.post("/api/pool/strategy", dependencies=[Depends(require_admin)])
def api_pool_strategy(req: StrategyReq) -> dict[str, Any]:
    """切换调用路线：proxy_first 先吃匿名额，token_first 先花账号配额。"""
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"未知策略：{req.strategy}")
    pool.attach_proxy_pool(proxy_pool, req.strategy)
    log.info("调用策略切换为 %s", pool.strategy)
    return {"ok": True, "strategy": pool.strategy, "pool": pool.snapshot()}


@app.post("/api/pool/save", dependencies=[Depends(require_admin)])
def api_pool_save() -> dict[str, Any]:
    _persist_pool()
    return {"ok": True, "file": str(POOL_FILE)}


def _persist_pool() -> None:
    try:
        pool.save(POOL_FILE)
    except Exception:  # noqa: BLE001
        log.exception("写入 tokens.txt 失败")


# ---------- 代理池 ----------


@app.get("/api/proxy", dependencies=[Depends(require_admin)])
def api_proxy(limit: int = 200, offset: int = 0, q: str = "") -> dict[str, Any]:
    snap = proxy_pool.snapshot(limit=min(max(limit, 1), 1000), offset=max(offset, 0), q=q)
    return {
        **snap,
        "strategy": pool.strategy,
        "strategies": list(STRATEGIES),
        "routes_available": pool.routes_available(),
    }


@app.post("/api/proxy/add", dependencies=[Depends(require_admin)])
def api_proxy_add(req: ProxyAddReq) -> dict[str, Any]:
    parsed = _parse_one_proxy(req.url)
    if not parsed:
        raise HTTPException(400, f"代理格式不支持：{req.url}")
    auto_label, url, auto_note, auto_exp = parsed
    if any(p.url == url for p in proxy_pool.proxies):
        raise HTTPException(409, "该代理已在池中")
    p = proxy_pool.add(
        url,
        req.label.strip() or auto_label,
        req.note.strip() or auto_note,
        days=req.days,
        expires=auto_exp or None,
    )
    _persist_proxies()
    log.info("代理池新增：%s（到期 %s）", p.label, p.expires or "永久")
    return {"ok": True, "label": p.label, "proxy": proxy_pool.snapshot()}


#: 一次导入最多读多少个文件、单个文件多大
MAX_IMPORT_FILES = 500
MAX_IMPORT_BYTES = 8 * 1024 * 1024
def _decode_text(raw: bytes) -> str:
    """代理列表可能是 UTF-8 / 带 BOM / GBK 存的，按顺序试，都不行就替换坏字节。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _import_proxies(
    text: str,
    days: int | None,
    sources: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """批量导入的公共收尾：写池子、落盘、记日志、回明细。"""
    r = proxy_pool.add_many(text, days=days)
    if r["added"]:
        _persist_proxies()
    log.info(
        "代理池批量导入：新增 %d，重复 %d，无效 %d，来源 %d 个文件，到期 %s",
        len(r["added"]), len(r["skipped"]), len(r["invalid"]),
        len(sources or []), f"{days} 天后" if days else "永久",
    )
    return {
        "ok": True,
        "added": r["added"],
        "added_count": len(r["added"]),
        "skipped": r["skipped"],
        "invalid": r["invalid"],
        "days": days,
        "sources": sources or [],
        "notes": notes or [],
        "proxy": proxy_pool.snapshot(),
    }


@app.post("/api/proxy/import", dependencies=[Depends(require_admin)])
def api_proxy_import(req: ProxyImportReq) -> dict[str, Any]:
    """批量导入代理（粘贴的文本）。可给这一批统一设到期天数；行内第 4 段自带的以行内为准。"""
    return _import_proxies(req.text or "", req.days)


@app.post("/api/proxy/import-files", dependencies=[Depends(require_admin)])
async def api_proxy_import_files(
    files: list[UploadFile] = File(...),
    days: int | None = Form(None),
) -> dict[str, Any]:
    """浏览器多选文件导入：一次可以选多个 txt / csv（字段名 files）。"""
    if not files:
        raise HTTPException(400, "没有选中文件")
    if len(files) > MAX_IMPORT_FILES:
        raise HTTPException(400, f"一次最多 {MAX_IMPORT_FILES} 个文件")
    chunks: list[str] = []
    sources: list[dict[str, Any]] = []
    notes: list[str] = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_IMPORT_BYTES:
            notes.append(f"{f.filename}：超过 {MAX_IMPORT_BYTES // 1024 // 1024}MB，已跳过")
            continue
        chunk = _decode_text(raw)
        chunks.append(chunk)
        sources.append({"path": f.filename or "upload", "bytes": len(raw), "lines": len(chunk.splitlines())})
    return _import_proxies("\n".join(chunks), days, sources, notes)


@app.post("/api/proxy/purge_expired", dependencies=[Depends(require_admin)])
def api_proxy_purge_expired() -> dict[str, Any]:
    """一键清理已到期的代理（整批作废的那种）。"""
    r = proxy_pool.purge_expired()
    if r["removed"]:
        _persist_proxies()
    log.info("清理过期代理 %d 个，剩余 %d 个", r["removed"], r["remaining"])
    return {"ok": True, **r, "proxy": proxy_pool.snapshot()}


@app.post("/api/proxy/remove", dependencies=[Depends(require_admin)])
def api_proxy_remove(req: ProxyRefReq) -> dict[str, Any]:
    ref = req.url or req.label or ""
    if not ref:
        raise HTTPException(400, "url 或 label 至少给一个")
    if not proxy_pool.remove(ref):
        raise HTTPException(404, f"代理不存在：{ref}")
    _persist_proxies()
    return {"ok": True, "proxy": proxy_pool.snapshot()}


@app.post("/api/proxy/clear_cooldown", dependencies=[Depends(require_admin)])
def api_proxy_clear(req: ProxyRefReq) -> dict[str, Any]:
    ref = req.url or req.label or None
    n = proxy_pool.clear_cooldown(ref)
    return {"ok": True, "cleared": n, "proxy": proxy_pool.snapshot()}


@app.post("/api/proxy/reload", dependencies=[Depends(require_admin)])
def api_proxy_reload() -> dict[str, Any]:
    global proxy_pool
    fresh = ProxyPool.from_env(proxy_file=PROXY_FILE)
    # 保留运行时状态：同 url 的冷却、计数、出口 IP 沿用
    old = {p.url: p for p in proxy_pool.proxies}
    for p in fresh.proxies:
        prev = old.get(p.url)
        if prev:
            p.cooldown_until = prev.cooldown_until
            p.ok, p.fail = prev.ok, prev.fail
            p.last_error, p.last_used = prev.last_error, prev.last_used
            p.egress_ip, p.latency_ms = prev.egress_ip, prev.latency_ms
    proxy_pool = fresh
    pool.attach_proxy_pool(proxy_pool)
    studio.pool = pool
    return {"ok": True, "proxy": proxy_pool.snapshot()}


@app.post("/api/proxy/save", dependencies=[Depends(require_admin)])
def api_proxy_save() -> dict[str, Any]:
    _persist_proxies()
    return {"ok": True, "file": str(PROXY_FILE)}


@app.post("/api/proxy/test", dependencies=[Depends(require_admin)])
def api_proxy_test(req: ProxyRefReq) -> dict[str, Any]:
    """探测代理能否出网 + 出口 IP。给界面的「测试」按钮用，不影响调用逻辑。"""
    targets = list(proxy_pool.proxies)
    if req.url or req.label:
        ref = req.url or req.label
        want = normalize_proxy(ref or "")
        targets = [p for p in targets if p.url == want or p.label == ref]
        if not targets:
            raise HTTPException(404, f"代理不存在：{ref}")
    else:
        # 代理池可能有几千个，默认只测前 N 个，避免一次跑几小时
        targets = targets[: max(1, min(req.limit, 500))]
    if not targets:
        return {"ok": True, "results": []}

    def one(p: Any) -> dict[str, Any]:
        r = probe_proxy(p.url)
        proxy_pool.mark_probe(p, ok=r["ok"], egress_ip=r["egress_ip"],
                             latency_ms=r["latency_ms"], error=r["error"])
        return {"label": p.label, "url": p.url, **r}

    workers = min(8, len(targets))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pxprobe") as ex:
        results = list(ex.map(one, targets))
    return {"ok": True, "results": results, "tested": len(results),
            "pool_total": len(proxy_pool)}


def _persist_proxies() -> None:
    try:
        proxy_pool.save(PROXY_FILE)
    except Exception:  # noqa: BLE001
        log.exception("写入 proxies.txt 失败")


# ---------- 渠道（Space） ----------


class ChannelToggleReq(BaseModel):
    space: str
    enabled: bool


class ChannelAddReq(BaseModel):
    space: str
    note: str = ""
    enabled: bool = True


class ChannelSpaceReq(BaseModel):
    space: str | None = None      # None = 全部


@app.get("/api/channels", dependencies=[Depends(require_admin)])
def api_channels() -> dict[str, Any]:
    return {
        "items": [c.public() for c in channel_store.list()],
        "enabled": channel_store.enabled_spaces(),
        "active": studio.spaces,
    }


@app.post("/api/channels/toggle", dependencies=[Depends(require_admin)])
def api_channel_toggle(req: ChannelToggleReq) -> dict[str, Any]:
    ch = channel_store.toggle(req.space, req.enabled)
    if ch is None:
        raise HTTPException(404, f"渠道不存在：{req.space}")
    active = studio.set_spaces(channel_store.enabled_spaces())
    log.info("渠道 %s 已%s，当前启用：%s", ch.space, "启用" if ch.enabled else "关闭", active or "（无）")
    return {"ok": True, "channel": ch.public(), "active": active}


@app.post("/api/channels/add", dependencies=[Depends(require_admin)])
def api_channel_add(req: ChannelAddReq) -> dict[str, Any]:
    space = (req.space or "").strip()
    if "/" not in space:
        raise HTTPException(400, "格式应为 owner/name")
    ch = channel_store.add(space, note=req.note, enabled=req.enabled)
    active = studio.set_spaces(channel_store.enabled_spaces())
    log.info("渠道新增：%s（%s）", ch.space, "启用" if ch.enabled else "关闭")
    return {"ok": True, "channel": ch.public(), "active": active}


@app.post("/api/channels/remove", dependencies=[Depends(require_admin)])
def api_channel_remove(req: ChannelSpaceReq) -> dict[str, Any]:
    if not req.space:
        raise HTTPException(400, "缺少 space")
    if not channel_store.remove(req.space):
        raise HTTPException(404, f"渠道不存在：{req.space}")
    active = studio.set_spaces(channel_store.enabled_spaces())
    log.info("渠道已移除：%s，当前启用：%s", req.space, active or "（无）")
    return {"ok": True, "active": active}


@app.post("/api/channels/probe", dependencies=[Depends(require_admin)])
def api_channel_probe(req: ChannelSpaceReq) -> dict[str, Any]:
    """探测渠道可用性（只读，不消耗配额）。不传 space 就测全部。"""
    targets = [req.space] if req.space else [c.space for c in channel_store.list()]
    if not targets:
        return {"ok": True, "results": [], "proxy": None}

    proxy = resolve_probe_proxy()
    if not proxy and len(proxy_pool):
        for p in proxy_pool.proxies:
            if p.available():
                proxy = p.url
                break

    def one(space: str) -> dict[str, Any]:
        r = probe_space(space, proxy=proxy)
        channel_store.set_probe(space, r)
        return r

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="chprobe") as ex:
        results = list(ex.map(one, targets))
    log.info("渠道探测完成：%d 个，经代理=%s", len(results), bool(proxy))
    return {
        "ok": True,
        "results": results,
        "proxy": bool(proxy),
        "items": [c.public() for c in channel_store.list()],
    }


# ---------- 卡密（管理员） ----------


@app.get("/api/licenses", dependencies=[Depends(require_admin)])
def api_licenses(
    q: str = "", status: str = "", limit: int = 500, offset: int = 0
) -> dict[str, Any]:
    items, total = licenses.list(
        q=q, status=status, limit=min(max(limit, 1), 2000), offset=max(offset, 0)
    )
    return {
        "items": [x.public() for x in items],
        "total": total,
        "stats": licenses.stats(),
    }


@app.post("/api/licenses/create", dependencies=[Depends(require_admin)])
def api_license_create(req: LicenseCreateReq) -> dict[str, Any]:
    n = min(max(int(req.count or 1), 1), 500)
    quota = None if req.quota is None else max(0, int(req.quota))
    days = None if req.days is None else max(0, int(req.days))
    if n == 1:
        created = [licenses.create(note=req.note, quota=quota, days=days)]
    else:
        created = licenses.create_many(n, note=req.note, quota=quota, days=days)
    log.info("生成卡密 %d 张（次数=%s，有效期=%s 天）", n, quota, days)
    return {
        "ok": True,
        "count": len(created),
        "created": [x.public() for x in created],
        "stats": licenses.stats(),
    }


@app.post("/api/licenses/update", dependencies=[Depends(require_admin)])
def api_license_update(req: LicenseUpdateReq) -> dict[str, Any]:
    lic = licenses.update(
        req.key,
        note=req.note,
        quota=req.quota,
        set_quota=req.set_quota,
        days=req.days,
        extend_days=req.extend_days,
    )
    if lic is None:
        raise HTTPException(404, "卡密不存在")
    return {"ok": True, "license": lic.public()}


@app.post("/api/licenses/disable", dependencies=[Depends(require_admin)])
def api_license_disable(req: LicenseKeyReq) -> dict[str, Any]:
    if req.disabled is None:
        raise HTTPException(400, "缺少 disabled 字段")
    lic = licenses.set_disabled(req.key, req.disabled)
    if lic is None:
        raise HTTPException(404, "卡密不存在")
    log.info("卡密 %s 已%s", lic.key, "停用" if lic.disabled else "启用")
    return {"ok": True, "license": lic.public()}


@app.post("/api/licenses/reset", dependencies=[Depends(require_admin)])
def api_license_reset(req: LicenseKeyReq) -> dict[str, Any]:
    lic = licenses.reset_usage(req.key)
    if lic is None:
        raise HTTPException(404, "卡密不存在")
    return {"ok": True, "license": lic.public()}


@app.delete("/api/licenses/{key}", dependencies=[Depends(require_admin)])
def api_license_delete(key: str) -> dict[str, Any]:
    if not licenses.delete(key):
        raise HTTPException(404, "卡密不存在")
    return {"ok": True, "removed": normalize_key(key)}


@app.post("/api/licenses/clear", dependencies=[Depends(require_admin)])
def api_license_clear(only_invalid: bool = True) -> dict[str, Any]:
    return {"ok": True, "removed": licenses.clear(only_invalid=only_invalid)}


# ---------- 控制台概览 ----------


@app.get("/api/overview", dependencies=[Depends(require_admin)])
def api_overview() -> dict[str, Any]:
    pool_snap = pool.snapshot()
    proxy_snap = proxy_pool.snapshot()
    return {
        "licenses": licenses.stats(),
        "pool": {
            "total": pool_snap["total"],
            "usable": pool_snap["usable"],
            "cooling": pool_snap["cooling"],
            "strategy": pool_snap["strategy"],
            "routes_available": pool_snap["routes_available"],
        },
        "proxy": {
            "total": proxy_snap["total"],
            "usable": proxy_snap["usable"],
            "cooling": proxy_snap["cooling"],
            "expired": proxy_snap.get("expired", 0),
        },
        "gallery": {"total": gallery.list(limit=0)[1]},
        "jobs": store.stats(),
        "space": {
            "id": SPACE_ID,
            "url": SPACE_URL,
            "src_mode": SPACE_SRC_MODE,
            "channels": len(channel_store),
            "enabled": channel_store.enabled_spaces(),
            "active": studio.spaces,
        },
        "auth": {"no_auth": NO_AUTH},
        "fingerprint": fingerprint.summary(),
    }


# ---------- 上传 ----------


async def _save_upload(file: UploadFile, sess: dict[str, Any]) -> dict[str, Any]:
    """落一张参考图，并把它归属到当前卡密用户。"""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    if len(raw) > UPLOAD_MAX_BYTES:
        raise HTTPException(413, f"文件超过 {UPLOAD_MAX_BYTES // (1024 * 1024)}MB")
    ext = Path(file.filename or "").suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise HTTPException(400, f"不支持的图片格式：{ext}")
    name = f"{uuid.uuid4().hex}{ext}"
    dst = UPLOAD_DIR / name
    dst.write_bytes(raw)
    owner = sess.get("key") or MATERIAL_ADMIN_OWNER
    material = materials.add(
        owner=owner,
        file_name=name,
        original_name=file.filename or name,
        size=len(raw),
        content_type=file.content_type or "application/octet-stream",
    )
    public = MaterialStore.public(
        material,
        image_url=f"/api/materials/{material['id']}/image",
    )
    return {
        "ok": True,
        "path": str(dst),
        "name": name,
        "url": public["image_url"],
        "material_id": material["id"],
        "material": public,
        "size": len(raw),
    }


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    sess: dict[str, Any] = Depends(require_license),
) -> dict[str, Any]:
    return await _save_upload(file, sess)


@app.post("/api/upload-many")
async def api_upload_many(
    files: list[UploadFile] = File(...),
    sess: dict[str, Any] = Depends(require_license),
) -> dict[str, Any]:
    """一次传多张参考图（前端多选 / 多文件拖拽）。超上限直接拒，省得白传。"""
    if len(files) > MAX_REFS:
        raise HTTPException(400, f"最多 {MAX_REFS} 张参考图")
    saved = [await _save_upload(f, sess) for f in files]
    return {"ok": True, "files": saved, "count": len(saved)}


@app.get("/api/upload/{name}")
def api_upload_get(
    name: str,
    sess: dict[str, Any] = Depends(require_license),
) -> FileResponse:
    safe_name = Path(name).name
    owned = next((item for item in materials.list() if item.get("file") == safe_name), None)
    if owned:
        _own_material(owned["id"], sess)
    p = (UPLOAD_DIR / safe_name).resolve()
    if not str(p).startswith(str(UPLOAD_DIR.resolve())) or not p.exists():
        raise HTTPException(404, "不存在")
    return FileResponse(p)


# ---------- 用户素材库 ----------


def _own_material(material_id: str, sess: dict[str, Any]) -> dict[str, Any]:
    """普通用户只能访问自己卡密下的素材，管理员可访问全部。"""
    item = materials.get(material_id)
    if not item:
        raise HTTPException(404, "素材不存在")
    if sess.get("role") != "admin" and item.get("owner") != sess.get("key"):
        raise HTTPException(404, "素材不存在")
    return item


@app.get("/api/materials")
def api_materials(sess: dict[str, Any] = Depends(require_license)) -> dict[str, Any]:
    owner = None if sess.get("role") == "admin" else (sess.get("key") or "")
    items = materials.list(owner=owner)
    return {
        "items": [
            MaterialStore.public(item, image_url=f"/api/materials/{item['id']}/image")
            for item in items
        ],
        "total": len(items),
    }


@app.delete("/api/materials/{material_id}")
def api_material_delete(
    material_id: str,
    sess: dict[str, Any] = Depends(require_license),
) -> dict[str, Any]:
    item = _own_material(material_id, sess)
    owner = None if sess.get("role") == "admin" else (sess.get("key") or "")
    removed = materials.delete(material_id, owner=owner)
    if removed is None:
        raise HTTPException(404, "素材不存在")
    path = (UPLOAD_DIR / removed["file"]).resolve()
    if UPLOAD_DIR.resolve() in path.parents:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("删除素材文件失败：%s", path)
    return {"ok": True, "removed": item["id"]}


@app.get("/api/materials/{material_id}/image")
def api_material_image(
    material_id: str,
    sess: dict[str, Any] = Depends(require_license),
) -> FileResponse:
    item = _own_material(material_id, sess)
    path = (UPLOAD_DIR / item["file"]).resolve()
    if UPLOAD_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "素材文件不存在")
    return FileResponse(path, media_type=item.get("content_type") or "application/octet-stream")


# ---------- 生成 ----------


@app.post("/api/generate")
def api_generate(
    req: GenerateReq,
    request: Request,
    sess: dict[str, Any] = Depends(require_license),
) -> dict[str, Any]:
    refs = norm_refs(req.references or req.reference)
    for material_id in req.material_ids or []:
        item = _own_material(material_id, sess)
        material_path = (UPLOAD_DIR / item["file"]).resolve()
        if UPLOAD_DIR.resolve() not in material_path.parents or not material_path.exists():
            raise HTTPException(404, "素材文件不存在")
        path = str(material_path)
        if path not in refs:
            refs.append(path)
    if len(refs) > MAX_REFS:
        raise HTTPException(400, f"最多 {MAX_REFS} 张参考图")
    missing = [r for r in refs if not Path(r).exists()]
    if missing:
        raise HTTPException(400, f"参考图不存在：{missing[0]}")

    try:
        validate_params(req.prompt, req.mode, refs, req.aspect_ratio, req.steps, req.seed, req.randomize_seed)
    except ValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    # 先扣次再提交：并发下不会超发；任务没成功会在 _run_job 里退回去
    lic = sess.get("license")
    lic_key = lic.key if lic is not None else None
    if lic is not None:
        ok, why, _ = licenses.consume(lic.key, ip=_client_ip(request))
        if not ok:
            raise HTTPException(403, f"卡密{why}")

    job = Job(
        id=secrets.token_hex(8),
        params={
            "prompt": req.prompt.strip(),
            "mode": req.mode,
            "references": refs,
            "reference": refs[0] if refs else None,   # 兼容旧前端 / 旧任务
            "aspect_ratio": req.aspect_ratio,
            "steps": int(req.steps),
            "seed": int(req.seed),
            "randomize_seed": bool(req.randomize_seed),
        },
        license_key=lic_key,
    )
    store.add(job)
    executor.submit(_run_job, job)
    return {
        "ok": True,
        "job_id": job.id,
        "job": job.public(),
        "remaining": licenses.get(lic_key).remaining() if lic_key else None,
    }


def _own_job(job_id: str, sess: dict[str, Any]) -> Job:
    """取任务并校验归属。别人的任务一律报 404，不暴露存在性。"""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if sess.get("role") != "admin" and job.license_key != sess.get("key"):
        raise HTTPException(404, "任务不存在")
    return job


@app.get("/api/job/active")
def api_job_active(sess: dict[str, Any] = Depends(require_license)) -> dict[str, Any]:
    """当前这个身份「最近一个值得恢复的任务」。

    页面刷新、切到控制台再回来、关掉浏览器重进，前端都靠这个把任务状态接回来。
    优先返回**进行中**的；没有的话返回 24 小时内最近一个已完成的，
    这样用户回来能看到上次的结果而不是一片空白。
    """
    jobs = store.list(60)
    if sess.get("role") != "admin":
        jobs = [j for j in jobs if j.license_key == sess.get("key")]

    for j in jobs:
        if j.status in ("running", "queued"):
            return {"job": j.public(), "resumable": True}

    now = time.time()
    for j in jobs:
        if now - j.created < 86400:
            return {"job": j.public(), "resumable": False}
    return {"job": None, "resumable": False}


@app.get("/api/job/{job_id}")
def api_job(job_id: str, sess: dict[str, Any] = Depends(require_license)) -> dict[str, Any]:
    return _own_job(job_id, sess).public()


@app.post("/api/job/{job_id}/cancel")
def api_job_cancel(job_id: str, sess: dict[str, Any] = Depends(require_license)) -> dict[str, Any]:
    job = _own_job(job_id, sess)
    job.cancel.set()
    return {"ok": True, "job": job.public()}


@app.get("/api/image/{job_id}")
def api_image(job_id: str, sess: dict[str, Any] = Depends(require_license)) -> FileResponse:
    job = _own_job(job_id, sess)
    if not job.result:
        raise HTTPException(404, "结果不存在")
    p = Path(job.result["image_path"])
    if not p.exists():
        raise HTTPException(404, "文件已被清理")
    return FileResponse(p, media_type="image/png", filename=p.name)


# ---------- 历史 ----------


@app.get("/api/history")
def api_history(
    limit: int = 40,
    sess: dict[str, Any] = Depends(require_license),
) -> dict[str, Any]:
    jobs = store.list(min(max(limit, 1), 200))
    if sess.get("role") != "admin":
        jobs = [j for j in jobs if j.license_key == sess.get("key")]
    return {"jobs": [j.public() for j in jobs]}


@app.delete("/api/history", dependencies=[Depends(require_admin)])
def api_history_clear() -> dict[str, Any]:
    return {"ok": True, "removed": store.clear()}



# ---------- 画廊 ----------


@app.get("/api/gallery")
def api_gallery(
    q: str = "",
    mode: str = "",
    limit: int = 200,
    offset: int = 0,
    sess: dict[str, Any] = Depends(require_license),
) -> dict[str, Any]:
    limit = min(max(limit, 1), 500)
    offset = max(offset, 0)
    lic_filter = "" if sess.get("role") == "admin" else (sess.get("key") or "-")
    items, total = gallery.list(q=q, mode=mode, license=lic_filter, limit=limit, offset=offset)
    return {
        "items": [_gallery_public(i) for i in items],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.post("/api/gallery/scan", dependencies=[Depends(require_admin)])
def api_gallery_scan() -> dict[str, Any]:
    """扫描 outputs 目录，把 PNG 内嵌的元数据补录进画廊。"""
    return {"ok": True, **gallery.scan()}


@app.delete("/api/gallery", dependencies=[Depends(require_admin)])
def api_gallery_clear(remove_files: bool = False) -> dict[str, Any]:
    return {"ok": True, "removed": gallery.clear(remove_files=remove_files)}


def _own_gallery(gid: str, sess: dict[str, Any]) -> dict[str, Any]:
    """取画廊记录并校验归属。别人的记录一律 404。"""
    it = gallery.get(gid)
    if not it:
        raise HTTPException(404, "画廊里没有这条记录")
    if sess.get("role") != "admin" and it.get("license") != sess.get("key"):
        raise HTTPException(404, "画廊里没有这条记录")
    return it


@app.get("/api/gallery/{gid}")
def api_gallery_item(gid: str, sess: dict[str, Any] = Depends(require_license)) -> dict[str, Any]:
    return _gallery_public(_own_gallery(gid, sess))


@app.delete("/api/gallery/{gid}", dependencies=[Depends(require_admin)])
def api_gallery_delete(gid: str, remove_file: bool = False) -> dict[str, Any]:
    if not gallery.delete(gid, remove_file=remove_file):
        raise HTTPException(404, "画廊里没有这条记录")
    return {"ok": True, "removed": gid, "file_removed": remove_file}


@app.get("/api/gallery/{gid}/image")
def api_gallery_image(gid: str, sess: dict[str, Any] = Depends(require_license)) -> FileResponse:
    return _gallery_file(gid, as_attachment=False, sess=sess)


@app.get("/api/gallery/{gid}/download")
def api_gallery_download(gid: str, sess: dict[str, Any] = Depends(require_license)) -> FileResponse:
    return _gallery_file(gid, as_attachment=True, sess=sess)


def _gallery_file(gid: str, *, as_attachment: bool, sess: dict[str, Any]) -> FileResponse:
    it = _own_gallery(gid, sess)
    if not it.get("file"):
        raise HTTPException(404, "记录不存在")
    p = (OUTPUT_DIR / it["file"]).resolve()
    if OUTPUT_DIR.resolve() not in p.parents:
        raise HTTPException(403, "非法路径")
    if not p.exists():
        raise HTTPException(404, "图片文件已被删除")
    if as_attachment:
        return FileResponse(p, media_type="image/png", filename=p.name)
    return FileResponse(p, media_type="image/png")
@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "space": SPACE_ID,
        "pool": pool.snapshot()["total"],
        "proxy": len(proxy_pool),
        "strategy": pool.strategy,
        "routes_available": pool.routes_available(),
    }


# ==========================================================================
# 入口
# ==========================================================================


def main() -> None:
    global pool, studio, POOL_FILE, PROXY_FILE, proxy_pool, executor, CONCURRENCY

    import uvicorn
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


    ap = argparse.ArgumentParser(description="Qwen Image 2.1 GGUF Studio Web 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--tokens", help="账号池文件，默认 ./tokens.txt")
    ap.add_argument("--proxies", help="代理池文件，默认 ./proxies.txt")
    ap.add_argument(
        "--default-proxy",
        help="给没写第四段的 token 账号兜一个出口（本机不能直连 HF 时必需）",
    )
    ap.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=DEFAULT_STRATEGY,
        help="调用路线：proxy_first=先吃匿名额(默认) / token_first / proxy_only / token_only",
    )
    ap.add_argument("--reload", action="store_true")
    ap.add_argument(
        "--no-auth",
        action="store_true",
        help="关掉登录与卡密校验（仅本机开发用，别带公网入口）",
    )
    ap.add_argument("--admin-password", help="直接设置管理员口令（也可用 QWEN_ADMIN_PASSWORD）")
    args = ap.parse_args()

    global NO_AUTH, admin_store
    if args.no_auth:
        NO_AUTH = True
    if args.admin_password:
        admin_store.set_password(args.admin_password)
        try:
            ADMIN_PW_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    POOL_FILE = Path(args.tokens) if args.tokens else TOKEN_FILE
    PROXY_FILE = Path(args.proxies) if args.proxies else PROXY_FILE
    # --concurrency 之前是个死参数（executor 在导入时就建好了），这里补上重建
    if args.concurrency and args.concurrency != CONCURRENCY:
        CONCURRENCY = max(1, args.concurrency)
        executor = ThreadPoolExecutor(max_workers=CONCURRENCY, thread_name_prefix="qwen")
    proxy_pool = ProxyPool.from_env(proxy_file=PROXY_FILE)
    pool = AccountPool.from_env(
        token_file=POOL_FILE,
        proxy_pool=proxy_pool,
        strategy=args.strategy,
        default_proxy=args.default_proxy,
    )
    studio = QwenStudio(pool, out_dir=OUTPUT_DIR)

    snap = pool.snapshot()
    print("=" * 74)
    print("  Qwen Image 2.1 GGUF Studio")
    print(f"  Space       : {SPACE_ID}")
    print(
        f"  调用策略    : {pool.strategy}  "
        f"(proxy_first=先吃匿名额, token_first=先花账号配额)"
    )
    print(f"  账号池      : 共 {snap['total']} 个，带 token {snap['usable']} 个，冷却中 {snap['cooling']} 个")
    px = snap.get("proxy") or {}
    print(
        f"  代理池      : 共 {px.get('total', 0)} 个，可用 {px.get('usable', 0)} 个，"
        f"冷却中 {px.get('cooling', 0)} 个"
    )
    if snap["routes_available"] == 0:
        print("  [!] 账号池和代理池里都没有可用路线 —— 无法生成，请先配置。")
        print(f"    账号：把 HF token 写进 {POOL_FILE}（每行一个，或 名字|token）")
        print(f"    代理：把代理写进 {PROXY_FILE}（每行一个，如 127.0.0.1:7897）")
    print(f"  输出目录    : {OUTPUT_DIR}")
    print(f"  卡密        : 共 {len(licenses)} 张（可用 {licenses.stats()['valid']} 张）")
    if NO_AUTH:
        print("  访问控制    : 已关闭（--no-auth）—— 任何人都能直接进控制台")
        print(f"  控制台      : http://{args.host}:{args.port}/admin")
    else:
        fresh = admin_store.take_fresh_password()
        if fresh:
            try:
                ADMIN_PW_FILE.write_text(fresh + "\n", encoding="utf-8")
            except OSError:
                pass
            print("=" * 74)
            print(f"  管理员口令  : {fresh}")
            print(f"  已写入 {ADMIN_PW_FILE.name}，登录后请立刻改掉，然后删掉这个文件")
        else:
            print(f"  管理员口令  : 已设置（忘了就删 {ADMIN_FILE.name} 重新生成）")
        print(f"  登录页      : http://{args.host}:{args.port}/login")
        print(f"  控制台      : http://{args.host}:{args.port}/admin")
    print("=" * 74)

    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
