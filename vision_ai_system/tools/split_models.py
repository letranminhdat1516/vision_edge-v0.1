import os, re, pathlib

src = pathlib.Path("models/generated_all.py")
out = pathlib.Path("models/generated")
out.mkdir(parents=True, exist_ok=True)
(open(out/"__init__.py","w")).close()

text = src.read_text(encoding="utf-8")

# Lấy phần header + các class ORM (class X(Base): ...)
m = re.split(r'(\nclass\s+\w+\(Base\):)', text)
header = m[0]
chunks = text[len(header):]

classes = re.findall(r'(\nclass\s+\w+\(Base\):[\s\S]*?)(?=\nclass\s+\w+\(Base\):|\Z)', chunks)
for c in classes:
    name = re.search(r'class\s+(\w+)\(Base\):', c).group(1)
    (out/f"{name.lower()}.py").write_text(header + c.strip() + "\n", encoding="utf-8")

print(f"OK: Split {len(classes)} model file(s) into {out}")