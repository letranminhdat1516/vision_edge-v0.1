# tools/entity_generator.py
"""
Tự động tạo dataclass entity từ các SQLAlchemy models.
Chạy:
    python tools/entity_generator.py --models-dir models/generated --outdir domain/entities/generated
"""

import argparse, importlib.util, inspect, os, re, sys
from pathlib import Path

HEADER = "# Auto-generated from SQLAlchemy models. Do NOT edit by hand.\nfrom dataclasses import dataclass\nfrom datetime import datetime\nfrom typing import Optional, Any\n\n"

PY_TYPE_MAP = {
    "INTEGER": "int",
    "BIGINT": "int",
    "SMALLINT": "int",
    "VARCHAR": "str",
    "TEXT": "str",
    "BOOLEAN": "bool",
    "TIMESTAMP": "datetime",
    "TIMESTAMP WITH TIME ZONE": "datetime",
    "DATE": "datetime",
    "JSON": "dict[str, Any]",
    "JSONB": "dict[str, Any]",
    "FLOAT": "float",
    "DOUBLE PRECISION": "float",
    "NUMERIC": "float",
    "UUID": "str",
}

def strip_len(t: str) -> str:
    return re.sub(r"\(.*\)", "", t).strip()

def guess_py_type(col) -> str:
    t = strip_len(str(col.type).upper())
    if "ENUM" in t:
        return "str"
    return PY_TYPE_MAP.get(t, "str")

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

def snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\-]", name))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "__init__.py").write_text("", encoding="utf-8")

    for py in models_dir.glob("*.py"):
        if py.name == "__init__.py":
            continue
        mod = load_module(py)
        code = HEADER
        wrote = False
        for _, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and hasattr(obj, "__table__"):
                table = getattr(obj, "__table__")
                fields = []
                for col in table.columns:
                    py_t = guess_py_type(col)
                    is_optional = col.nullable or col.primary_key
                    ftype = f"Optional[{py_t}]" if is_optional else py_t
                    default = " = None" if is_optional else ""
                    fields.append((col.name, ftype, default))
                entity_name = snake_to_camel(table.name)
                code += f"@dataclass(slots=True)\nclass {entity_name}:\n"
                for name, ftype, default in fields:
                    code += f"    {name}: {ftype}{default}\n"
                code += "\n"
                wrote = True
        if wrote:
            (outdir / py.name).write_text(code, encoding="utf-8")

if __name__ == "__main__":
    main()
