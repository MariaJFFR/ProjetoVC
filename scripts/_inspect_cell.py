import nbformat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.read(ROOT / "notebooks" / "02_autoencoder.ipynb", as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == "code"]
last = code_cells[-1]
print("=== SOURCE DA ÚLTIMA CÉLULA ===")
print(last.source)
print("\n=== OUTPUTS ===")
for o in last.get("outputs", []):
    t = o.get("output_type")
    if t == "error":
        print("ERRO:", o.get("ename"), "-", o.get("evalue"))
        print("\n".join(o.get("traceback", []))[-1500:])
    elif t == "stream":
        print("STREAM:", o.get("text", "")[:500])
    else:
        print("OUT:", t, list(o.get("data", {}).keys()))
