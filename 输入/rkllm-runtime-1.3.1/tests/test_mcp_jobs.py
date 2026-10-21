"""jobs.py 单测：状态机、TTL、容量、取消幂等。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.mcp.jobs import JobTable, estimate_seconds


def test_lifecycle():
    table = JobTable(ttl_s=600, max_jobs=4)
    job = table.create()
    assert job.state == "working"
    job.complete({"text": "done"})
    got = table.get(job.id)
    assert got.state == "completed"
    assert got.result == {"text": "done"}
    assert got.summary() == {"job_id": job.id, "status": "completed"}


def test_failure_state():
    table = JobTable()
    job = table.create()
    job.fail("boom")
    assert table.get(job.id).state == "failed"
    assert table.get(job.id).error == "boom"


def test_cancel_idempotent():
    table = JobTable()
    job = table.create()
    assert job.cancel() is True
    assert job.cancel() is False
    assert job.cancel_requested
    # 完成后取消不再转移状态
    job2 = table.create()
    job2.complete(None)
    assert job2.cancel() is False


def test_ttl_eviction():
    table = JobTable(ttl_s=0, max_jobs=4)
    job = table.create()
    job.complete(None)
    time.sleep(0.02)
    assert table.get(job.id) is None          # TTL=0：完成即逐出


def test_max_jobs_drops_oldest():
    table = JobTable(ttl_s=600, max_jobs=2)
    a, b, c = table.create(), table.create(), table.create()
    ids = {j.id for j in table.all()}
    assert a.id not in ids and b.id in ids and c.id in ids


def test_estimate_seconds():
    assert estimate_seconds("small", 0, 512) == 64.0     # 512/8
    assert estimate_seconds("large", 0, 512) == 256.0    # 512/2
    assert estimate_seconds("small", 50, 0) == 1.0       # prefill 50/50
