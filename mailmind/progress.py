from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


def _now() -> int:
    return int(time.time())


@dataclass
class ProgressConfig:
    mode: str = "auto"  # auto|bar|json|none
    root: Optional[Path] = None  # for status file updates
    key: str = "task"
    total: Optional[int] = None
    desc: str = ""


class ProgressReporter:
    def __init__(self, cfg: ProgressConfig):
        self.cfg = cfg
        self._count = 0
        self._start = _now()
        self._tqdm = None
        mode = cfg.mode
        if mode == "auto":
            # If stdout is a tty, prefer bar; otherwise json
            mode = "bar" if sys.stderr.isatty() else "json"
        self._mode = mode
        if mode == "bar":
            try:
                from tqdm import tqdm  # type: ignore

                self._tqdm = tqdm(total=cfg.total, desc=cfg.desc or cfg.key, unit="it")
            except Exception:
                self._mode = "json"
        # Initialize status file if needed
        self._status_path: Optional[Path] = None
        if cfg.root:
            state_dir = cfg.root / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            self._status_path = state_dir / "progress.json"
            self._write_status({
                "key": cfg.key,
                "desc": cfg.desc,
                "total": cfg.total,
                "done": 0,
                "start_ts": self._start,
                "end_ts": None,
                "mode": self._mode,
            })

    def _write_status(self, data: dict) -> None:
        if not self._status_path:
            return
        try:
            # Merge with existing JSON keyed by task key
            if self._status_path.exists():
                try:
                    all_data = json.loads(self._status_path.read_text())
                except Exception:
                    all_data = {}
            else:
                all_data = {}
            all_data[self.cfg.key] = data
            self._status_path.write_text(json.dumps(all_data, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def update(self, n: int = 1) -> None:
        self._count += n
        if self._tqdm is not None:
            try:
                self._tqdm.update(n)
            except Exception:
                pass
        elif self._mode == "json":
            evt = {
                "type": "progress",
                "key": self.cfg.key,
                "done": self._count,
                "total": self.cfg.total,
                "ts": _now(),
            }
            try:
                sys.stderr.write(json.dumps(evt) + "\n")
                sys.stderr.flush()
            except Exception:
                pass
        # Write status file
        self._write_status({
            "key": self.cfg.key,
            "desc": self.cfg.desc,
            "total": self.cfg.total,
            "done": self._count,
            "start_ts": self._start,
            "end_ts": None,
            "mode": self._mode,
        })

    def close(self) -> None:
        if self._tqdm is not None:
            try:
                self._tqdm.close()
            except Exception:
                pass
        # Mark end_ts
        self._write_status({
            "key": self.cfg.key,
            "desc": self.cfg.desc,
            "total": self.cfg.total,
            "done": self._count,
            "start_ts": self._start,
            "end_ts": _now(),
            "mode": self._mode,
        })

    def wrap(self, it: Iterable) -> Iterator:
        for x in it:
            yield x
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def resolve_progress_mode(arg_mode: Optional[str] = None) -> str:
    env = os.getenv("MAILMIND_PROGRESS", "auto").lower()
    if arg_mode and arg_mode != "auto":
        return arg_mode
    return env

