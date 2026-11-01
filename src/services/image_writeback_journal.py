from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import json
import sqlite3
import time
import traceback


def image_error_details(error: BaseException) -> Dict[str, Any]:
    """保留异常链、组件备注和 ExceptionGroup 子异常，不能只存分组标题。"""
    detail: Dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }
    if isinstance(error, BaseExceptionGroup):
        detail["children"] = [image_error_details(child) for child in error.exceptions]
    return detail


class ImageWritebackJournal:
    """持久化消息及组件结果；原图仍从聊天记录读取，失败历史只追加不清空。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS image_writeback_jobs (
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                retry_at REAL NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS image_writeback_components (
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                component_path TEXT NOT NULL,
                status TEXT NOT NULL,
                error_json TEXT NOT NULL DEFAULT '{}',
                attempts INTEGER NOT NULL DEFAULT 1,
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, message_id, component_path)
            );
            CREATE TABLE IF NOT EXISTS image_writeback_component_history (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                component_path TEXT NOT NULL,
                status TEXT NOT NULL,
                error_json TEXT NOT NULL,
                recorded_at REAL NOT NULL
            );
        """)
        self.connection.commit()

    def enqueue(self, session_id: str, message_id: str) -> None:
        if not session_id or not message_id:
            raise ValueError("图片入库任务缺少真实聊天流或消息 ID")
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO image_writeback_jobs(session_id,message_id,updated_at) VALUES (?,?,?)",
                (session_id, message_id, time.time()),
            )

    def next_job(self) -> Optional[Dict[str, Any]]:
        # 成功前始终保持 pending，进程退出不会丢失正在处理的任务。
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM image_writeback_jobs WHERE status='pending' AND retry_at<=? "
                "ORDER BY retry_at,updated_at LIMIT 1", (time.time(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def components(self, session_id: str, message_id: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM image_writeback_components WHERE session_id=? AND message_id=? "
                "ORDER BY component_path", (session_id, message_id),
            ).fetchall()
        result = {}
        for row in rows:
            item = dict(row)
            item["error"] = json.loads(item.pop("error_json"))
            result[item["component_path"]] = item
        return result

    def record_component(
        self, session_id: str, message_id: str, component_path: str, status: str,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        if status not in {"done", "permanent_failure", "retryable_failure"}:
            raise ValueError(f"非法图片组件状态: {status}")
        encoded = json.dumps(error if error is not None else {}, ensure_ascii=False)
        now = time.time()
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO image_writeback_components "
                "(session_id,message_id,component_path,status,error_json,updated_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(session_id,message_id,component_path) DO UPDATE SET "
                "status=excluded.status,error_json=excluded.error_json,updated_at=excluded.updated_at,"
                "attempts=image_writeback_components.attempts+1",
                (session_id, message_id, component_path, status, encoded, now),
            )
            self.connection.execute(
                "INSERT INTO image_writeback_component_history "
                "(session_id,message_id,component_path,status,error_json,recorded_at) VALUES (?,?,?,?,?,?)",
                (session_id, message_id, component_path, status, encoded, now),
            )

    def complete(self, session_id: str, message_id: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE image_writeback_jobs SET status='done',last_error='',updated_at=? "
                "WHERE session_id=? AND message_id=?", (time.time(), session_id, message_id),
            )

    def fail(self, job: Dict[str, Any], error: str, max_attempts: int, *, retryable: bool = True) -> None:
        attempts = int(job["attempts"]) + 1
        now = time.time()
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE image_writeback_jobs SET status=?,attempts=?,retry_at=?,last_error=?,updated_at=? "
                "WHERE session_id=? AND message_id=?",
                ("failed" if not retryable or attempts >= max_attempts else "pending", attempts,
                 now + min(300, 2 ** min(attempts, 9)), error, now, job["session_id"], job["message_id"]),
            )

    def list_jobs(self, status: str, limit: int, offset: int) -> Dict[str, Any]:
        where = "WHERE status=?" if status else ""
        parameters = (status,) if status else ()
        with self._lock:
            total = self.connection.execute(
                f"SELECT COUNT(*) FROM image_writeback_jobs {where}", parameters,
            ).fetchone()[0]
            rows = self.connection.execute(
                f"SELECT * FROM image_writeback_jobs {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
            items: List[Dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["components"] = list(self.components(item["session_id"], item["message_id"]).values())
                items.append(item)
        return {"success": True, "items": items, "total": total}

    def retry_failed(self) -> int:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE image_writeback_jobs AS jobs SET status='pending',attempts=0,retry_at=0,updated_at=? "
                "WHERE status='failed' AND (NOT EXISTS (SELECT 1 FROM image_writeback_components AS c "
                "WHERE c.session_id=jobs.session_id AND c.message_id=jobs.message_id) OR EXISTS "
                "(SELECT 1 FROM image_writeback_components AS c WHERE c.session_id=jobs.session_id "
                "AND c.message_id=jobs.message_id AND c.status='retryable_failure'))", (time.time(),),
            )
        return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self.connection.close()
