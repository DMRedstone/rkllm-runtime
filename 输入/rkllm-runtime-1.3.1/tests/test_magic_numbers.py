"""N11：代码中的数值常量必须登记（见 工具/脚本/magic_numbers.py）。"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "工具" / "脚本" / "magic_numbers.py"


def _load():
    spec = importlib.util.spec_from_file_location("magic_numbers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["magic_numbers"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_exists():
    assert SCRIPT.is_file(), "魔法数扫描脚本缺失：%s" % SCRIPT


def test_no_unregistered_numbers():
    mod = _load()
    known = mod.load_baseline()
    assert known, "基线为空，先运行 magic_numbers.py --update 并审阅"
    unknown = [h for h in mod.scan() if h["key"] not in known]
    assert unknown == [], (
        "出现未登记的数值常量（要么改成配置键，要么写入基线并说明理由）：\n"
        + "\n".join("  %s:%d  %r" % (h["file"], h["line"], h["value"])
                    for h in unknown[:20]))
