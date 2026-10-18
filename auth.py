#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""卡密 / 会话 / 管理员口令
==========================

三块互不依赖的能力，给 server.py 用：

1. **卡密** —— 次数 + 有效期，存 ``licenses.json``（原子写）。扣减是原子的，
   失败能退回，所以「提交任务扣 1 次、任务失败退 1 次」不会漏。
2. **会话** —— HMAC-SHA256 签名的无状态 token，放 HttpOnly cookie。
   服务端不存 session，重启不踢人；改 ``secret.key`` 才会全体失效。
3. **管理员口令** —— PBKDF2-HMAC-SHA256（20 万轮）存 ``admin.json``，
   首次启动自动生成随机口令，写 ``admin_password.txt`` 并打印到控制台。

刻意不引入任何第三方依赖：只用标准库。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import string
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

# ==========================================================================
# 1. 小工具
# ==========================================================================

#: 卡密字符集 —— 去掉 0/O/1/I 这些手抄容易错的
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
KEY_PREFIX = "QWEN"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def gen_key(groups: int = 4, size: int = 4) -> str:
    """生成卡密，形如 ``QWEN-A7K2-M9XP-4TRB-QW3D``。"""
    body = "-".join(
        "".join(secrets.choice(ALPHABET) for _ in range(size)) for _ in range(groups)
    )
    return f"{KEY_PREFIX}-{body}"


def normalize_key(raw: str) -> str:
    """用户输入容错：去空格、转大写、补前缀、去掉多余分隔符。"""
    s = "".join(ch for ch in (raw or "").strip().upper() if ch.isalnum() or ch == "-")
    if not s:
        return ""
    if not s.startswith(KEY_PREFIX):
        s = f"{KEY_PREFIX}-{s}"
    body = s[len(KEY_PREFIX):].lstrip("-").replace("-", "")
    if not body:
        return ""
    return f"{KEY_PREFIX}-" + "-".join(body[i : i + 4] for i in range(0, len(body), 4))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ==========================================================================
# 2. 会话 token（HMAC 签名，无状态）
# ==========================================================================


def sign_token(secret: str, payload: dict[str, Any], ttl: int) -> str:
    body = {**payload, "exp": int(time.time()) + int(ttl)}
    raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).rstrip(b"=")
    sig = hmac.new(secret.encode("utf-8"), b64, hashlib.sha256).digest()
    return b64.decode() + "." + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


def verify_token(secret: str, token: str) -> dict[str, Any] | None:
    """校验签名与过期时间。任何异常都当成无效，绝不抛给调用方。"""
    if not token or "." not in token:
        return None
    try:
        b64, sig_b64 = token.split(".", 1)
    except ValueError:
        return None
    expected = hmac.new(secret.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).digest()
    try:
        got = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
    except Exception:  # noqa: BLE001
        return None
    if not hmac.compare_digest(got, expected):
        return None
    try:
        raw = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload


def load_or_create_secret(path: str | Path) -> str:
    p = Path(path)
    if p.exists():
        s = p.read_text(encoding="utf-8").strip()
        if s:
            return s
    s = secrets.token_urlsafe(48)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, s + "\n")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return s


# ==========================================================================
# 3. 管理员口令
# ==========================================================================

PBKDF2_ROUNDS = 200_000


def hash_password(password: str, salt: str | None = None, rounds: int = PBKDF2_ROUNDS) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds)
    return dk.hex(), salt


def verify_password(password: str, stored: str, salt: str, rounds: int = PBKDF2_ROUNDS) -> bool:
    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt.encode("utf-8"), rounds)
    return hmac.compare_digest(dk.hex(), stored or "")


class AdminStore:
    """管理员口令。文件里只存哈希和盐，口令本身不落盘（除了首次的提示文件）。"""

    def __init__(self, path: str | Path, *, password: str | None = None) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.hash = ""
        self.salt = ""
        self.rounds = PBKDF2_ROUNDS
        self.updated = ""
        self._fresh_password: str | None = None
        self.load(password=password)

    # ---------- 读写 ----------

    def load(self, *, password: str | None = None) -> None:
        env_pw = password or os.getenv("QWEN_ADMIN_PASSWORD", "").strip()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.hash = data.get("hash", "")
                self.salt = data.get("salt", "")
                self.rounds = int(data.get("rounds", PBKDF2_ROUNDS))
                self.updated = data.get("updated", "")
            except Exception:  # noqa: BLE001
                self.hash = ""
        if env_pw:
            # 环境变量优先：方便容器化部署时直接指定
            h, s = hash_password(env_pw)
            if h != self.hash or s != self.salt:
                self.set_password(env_pw, persist=False)
            return
        if not self.hash:
            self._bootstrap()

    def _bootstrap(self) -> None:
        """首次启动：生成一个足够强的随机口令。"""
        pw = secrets.token_urlsafe(12)
        self.set_password(pw, persist=True)
        self._fresh_password = pw

    def set_password(self, password: str, *, persist: bool = True) -> None:
        if len(password or "") < 6:
            raise ValueError("口令至少 6 位")
        h, s = hash_password(password)
        with self._lock:
            self.hash, self.salt = h, s
            self.updated = _iso(_now()) or ""
            if persist:
                self._save_locked()

    def _save_locked(self) -> None:
        payload = {
            "hash": self.hash,
            "salt": self.salt,
            "rounds": self.rounds,
            "updated": self.updated,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2))
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # ---------- 校验 ----------

    def check(self, password: str) -> bool:
        if not self.hash:
            return False
        return verify_password(password or "", self.hash, self.salt, self.rounds)

    def take_fresh_password(self) -> str | None:
        """首次启动生成的口令只交出去一次，之后返回 None。"""
        pw, self._fresh_password = self._fresh_password, None
        return pw


# ==========================================================================
# 4. 卡密
# ==========================================================================

STATUS_VALID = "valid"
STATUS_EXPIRED = "expired"
STATUS_EXHAUSTED = "exhausted"
STATUS_DISABLED = "disabled"

STATUS_CN = {
    STATUS_VALID: "可用",
    STATUS_EXPIRED: "已过期",
    STATUS_EXHAUSTED: "次数用尽",
    STATUS_DISABLED: "已停用",
}


@dataclass
class License:
    key: str
    note: str = ""
    quota: int | None = None          # None = 不限次数
    used: int = 0
    expires: str | None = None        # ISO8601 UTC，None = 永不过期
    created: str = ""
    last_used: str | None = None
    last_ip: str = ""
    disabled: bool = False

    # ---------- 判定 ----------

    def status(self, now: datetime | None = None) -> str:
        if self.disabled:
            return STATUS_DISABLED
        exp = _parse_iso(self.expires)
        if exp and (_now() if now is None else now) >= exp:
            return STATUS_EXPIRED
        if self.quota is not None and self.used >= self.quota:
            return STATUS_EXHAUSTED
        return STATUS_VALID

    def remaining(self) -> int | None:
        if self.quota is None:
            return None
        return max(0, self.quota - self.used)

    def expired_at(self) -> datetime | None:
        return _parse_iso(self.expires)

    def public(self, now: datetime | None = None) -> dict[str, Any]:
        st = self.status(now)
        return {
            "key": self.key,
            "note": self.note,
            "quota": self.quota,
            "used": self.used,
            "remaining": self.remaining(),
            "expires": self.expires,
            "status": st,
            "status_cn": STATUS_CN.get(st, st),
            "created": self.created,
            "last_used": self.last_used,
            "last_ip": self.last_ip,
            "disabled": self.disabled,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LicenseStore:
    """卡密库。索引存 ``licenses.json``，原子写，多线程安全。"""

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._items: dict[str, License] = {}
        self.load()

    # ---------- 持久化 ----------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        for raw in data.get("items", []):
            try:
                lic = License(
                    key=raw["key"],
                    note=raw.get("note", ""),
                    quota=raw.get("quota"),
                    used=int(raw.get("used", 0)),
                    expires=raw.get("expires"),
                    created=raw.get("created", ""),
                    last_used=raw.get("last_used"),
                    last_ip=raw.get("last_ip", ""),
                    disabled=bool(raw.get("disabled", False)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self._items[lic.key] = lic

    def _save_locked(self) -> None:
        payload = {
            "version": self.VERSION,
            "updated": _iso(_now()),
            "items": [x.to_dict() for x in sorted(self._items.values(), key=lambda i: i.created, reverse=True)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2))

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    # ---------- 生成 ----------

    def create(
        self,
        *,
        note: str = "",
        quota: int | None = 10,
        days: int | None = 30,
        key: str | None = None,
    ) -> License:
        with self._lock:
            k = normalize_key(key) if key else ""
            while not k or k in self._items:
                k = gen_key()
            lic = License(
                key=k,
                note=note,
                quota=(None if quota is None else max(0, int(quota))),
                used=0,
                expires=_iso(_now() + timedelta(days=days)) if days else None,
                created=_iso(_now()) or "",
            )
            self._items[k] = lic
            self._save_locked()
            return lic

    def create_many(
        self,
        count: int,
        *,
        note: str = "",
        quota: int | None = 10,
        days: int | None = 30,
    ) -> list[License]:
        out: list[License] = []
        for _ in range(max(0, int(count))):
            out.append(self.create(note=note, quota=quota, days=days))
        return out

    # ---------- 查询 ----------

    def get(self, key: str) -> License | None:
        k = normalize_key(key)
        with self._lock:
            return self._items.get(k)

    def list(
        self,
        *,
        q: str = "",
        status: str = "",
        limit: int = 500,
        offset: int = 0,
    ) -> tuple[list[License], int]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda i: i.created, reverse=True)
        if status:
            items = [x for x in items if x.status() == status]
        if q:
            needle = q.strip().upper()
            items = [
                x
                for x in items
                if needle in x.key or needle in (x.note or "").upper() or needle in (x.last_ip or "")
            ]
        total = len(items)
        return items[offset : offset + max(0, limit)], total

    # ---------- 变更 ----------

    def consume(self, key: str, *, ip: str = "") -> tuple[bool, str, License | None]:
        """原子扣 1 次。返回 (是否成功, 原因, 卡密)。"""
        k = normalize_key(key)
        with self._lock:
            lic = self._items.get(k)
            if lic is None:
                return False, "卡密不存在", None
            st = lic.status()
            if st != STATUS_VALID:
                return False, STATUS_CN.get(st, st), lic
            if lic.quota is not None:
                if lic.used >= lic.quota:
                    return False, "次数用尽", lic
                lic.used += 1
            lic.last_used = _iso(_now())
            lic.last_ip = ip or lic.last_ip
            self._save_locked()
            return True, "", lic

    def refund(self, key: str) -> bool:
        """任务失败时把次数退回去。不限次数的卡密不用退。"""
        k = normalize_key(key)
        with self._lock:
            lic = self._items.get(k)
            if lic is None or lic.quota is None or lic.used <= 0:
                return False
            lic.used -= 1
            self._save_locked()
            return True

    def set_disabled(self, key: str, disabled: bool) -> License | None:
        k = normalize_key(key)
        with self._lock:
            lic = self._items.get(k)
            if lic is None:
                return None
            lic.disabled = bool(disabled)
            self._save_locked()
            return lic

    def update(
        self,
        key: str,
        *,
        note: str | None = None,
        quota: int | None = None,
        set_quota: bool = False,
        days: int | None = None,
        extend_days: int | None = None,
    ) -> License | None:
        """改备注 / 改总次数 / 重设有效期 / 在原有效期上顺延。"""
        k = normalize_key(key)
        with self._lock:
            lic = self._items.get(k)
            if lic is None:
                return None
            if note is not None:
                lic.note = note
            if set_quota:
                lic.quota = None if quota is None else max(0, int(quota))
            if days is not None:
                lic.expires = _iso(_now() + timedelta(days=days)) if days else None
            if extend_days:
                base = lic.expired_at() or _now()
                lic.expires = _iso(base + timedelta(days=int(extend_days)))
            self._save_locked()
            return lic

    def reset_usage(self, key: str) -> License | None:
        k = normalize_key(key)
        with self._lock:
            lic = self._items.get(k)
            if lic is None:
                return None
            lic.used = 0
            self._save_locked()
            return lic

    def delete(self, key: str) -> bool:
        k = normalize_key(key)
        with self._lock:
            if self._items.pop(k, None) is None:
                return False
            self._save_locked()
            return True

    def clear(self, *, only_invalid: bool = False) -> int:
        with self._lock:
            if only_invalid:
                dead = [k for k, v in self._items.items() if v.status() != STATUS_VALID]
                for k in dead:
                    self._items.pop(k, None)
                n = len(dead)
            else:
                n = len(self._items)
                self._items.clear()
            if n:
                self._save_locked()
            return n

    # ---------- 统计 ----------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._items.values())
        now = _now()
        by = {s: 0 for s in (STATUS_VALID, STATUS_EXPIRED, STATUS_EXHAUSTED, STATUS_DISABLED)}
        for x in items:
            by[x.status(now)] = by.get(x.status(now), 0) + 1
        unlimited = sum(1 for x in items if x.quota is None)
        return {
            "total": len(items),
            "valid": by[STATUS_VALID],
            "expired": by[STATUS_EXPIRED],
            "exhausted": by[STATUS_EXHAUSTED],
            "disabled": by[STATUS_DISABLED],
            "unlimited": unlimited,
            "used_total": sum(x.used for x in items),
            "quota_total": sum((x.quota or 0) for x in items),
        }

    def __len__(self) -> int:
        return len(self._items)
