"""
Fixes SadTalker (2023 code) so it runs on today's Python / NumPy 2 / new basicsr.
Run once after cloning:  python patch_sadtalker.py SadTalker
Safe to run more than once.
"""
import pathlib
import re
import sys
import site
import sysconfig

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "SadTalker")


def patch(path, subs):
    p = pathlib.Path(path)
    if not p.is_absolute() and not str(p).startswith(str(root)):
        p = root / p
    if not p.exists():
        return
    text = p.read_text()
    new = text
    for old, rep in subs:
        new = re.sub(old, rep, new)
    if new != text:
        p.write_text(new)
        print("patched", p)


# 1) NumPy removed np.float / np.int and VisibleDeprecationWarning
for py in root.rglob("*.py"):
    patch(py, [
        (r"np\.float\b(?!\d|_)", "np.float64"),
        (r"np\.int\b(?!\d|_)", "np.int64"),
        (r"np\.VisibleDeprecationWarning", "DeprecationWarning"),
    ])

# 2) NumPy 2 refuses to mix arrays and numbers in one np.array([...])
patch(pathlib.Path("src/face3d/util/preprocess.py"), [
    (r"np\.array\(\[w0, h0, s, t\[0\], t\[1\]\]\)",
     "np.array([float(w0), float(h0), float(s), float(t[0]), float(t[1])])"),
    (r"(\n    s = rescale_factor/s\n)(?!    t = np\.asarray)",
     r"\1    t = np.asarray(t, dtype=np.float64).reshape(-1)\n"
     r"    s = np.float64(np.asarray(s, dtype=np.float64).reshape(-1)[0])\n"),
])

# 2b) same kind of numpy 2 fix in the main preprocess step
patch(pathlib.Path("src/utils/preprocess.py"), [
    (r"np\.array\(\[float\(item\) for item in np\.hsplit\(trans_params, 5\)\]\)",
     "np.asarray(trans_params, dtype=np.float64).reshape(-1)[:5]"),
])
# 3) basicsr imports a torchvision module that no longer exists
dirs = set(site.getsitepackages() + [site.getusersitepackages(), sysconfig.get_paths()["purelib"]])
for d in dirs:
    f = pathlib.Path(d) / "basicsr" / "data" / "degradations.py"
    patch(f, [(r"torchvision\.transforms\.functional_tensor", "torchvision.transforms.functional")])

print("SadTalker patch done.")
