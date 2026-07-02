"""Geometry and prediction-mesh support.

Two related concepts:

* **Geometry** - the propagation environment used for ray tracing (a Mitsuba
  ``.xml`` scene, or a mesh).  The Sionna engine loads it; the analytical engine
  ignores geometry but still uses its 2D footprint to auto-fit the view.
* **Prediction mesh** - the surface on which coverage is evaluated.  Its XY
  bounding box and height define the measurement region.

This module is pure Python / NumPy (no Sionna): it parses ASCII **OBJ** and
**PLY** meshes, computes bounding boxes, extracts a 2D footprint (edges
projected to the XY plane) for drawing, and derives a :class:`GridSpec` that
fits the geometry.  For Mitsuba ``.xml`` scenes it discovers referenced mesh
files and unions their bounds (transforms are approximated as identity, which
is sufficient for auto-fitting the view).
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import GridSpec


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Mesh parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray          # (N, 3)
    faces: List[Tuple[int, ...]]  # vertex-index tuples
    sha: str
    path: str

    @property
    def bounds(self) -> Dict[str, float]:
        v = self.vertices
        return {
            "x_min": float(v[:, 0].min()), "x_max": float(v[:, 0].max()),
            "y_min": float(v[:, 1].min()), "y_max": float(v[:, 1].max()),
            "z_min": float(v[:, 2].min()), "z_max": float(v[:, 2].max()),
        }


def parse_obj(path: str):
    verts, faces = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # OBJ is 1-indexed; negatives are relative to current count
                idx = [(i - 1) if i > 0 else (len(verts) + i) for i in idx]
                faces.append(tuple(idx))
    return np.asarray(verts, dtype=float), faces


_PLY_STRUCT = {  # ply type -> (struct char, numpy dtype, size)
    "char": ("b", "i1"), "int8": ("b", "i1"),
    "uchar": ("B", "u1"), "uint8": ("B", "u1"),
    "short": ("h", "i2"), "int16": ("h", "i2"),
    "ushort": ("H", "u2"), "uint16": ("H", "u2"),
    "int": ("i", "i4"), "int32": ("i", "i4"),
    "uint": ("I", "u4"), "uint32": ("I", "u4"),
    "float": ("f", "f4"), "float32": ("f", "f4"),
    "double": ("d", "f8"), "float64": ("d", "f8"),
}


def parse_ply(path: str, max_faces: int = 400_000):
    """Parse an ASCII or binary (little/big-endian) PLY mesh.

    Only the vertex ``x/y/z`` and the face vertex-index lists are extracted;
    extra per-vertex properties (normals, colours, ...) are skipped correctly.
    """
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("Not a PLY file (no end_header)")
    nl = raw.find(b"\n", end)
    header = raw[:nl].decode("ascii", "replace")
    body = raw[nl + 1:]

    fmt = "ascii"
    elements = []          # {name, count, props:[('scalar',type,name) | ('list',cnt_t,idx_t,name)]}
    cur = None
    for line in header.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            cur = {"name": t[1], "count": int(t[2]), "props": []}
            elements.append(cur)
        elif t[0] == "property" and cur is not None:
            if t[1] == "list":
                cur["props"].append(("list", t[2], t[3], t[4]))
            else:
                cur["props"].append(("scalar", t[1], t[2]))

    vtx = next((e for e in elements if e["name"] == "vertex"), None)
    fac = next((e for e in elements if e["name"] == "face"), None)
    if vtx is None:
        raise ValueError("PLY has no vertex element")
    vprops = vtx["props"]
    names = [p[2] for p in vprops]
    xi = names.index("x") if "x" in names else 0
    yi = names.index("y") if "y" in names else 1
    zi = names.index("z") if "z" in names else 2

    if fmt == "ascii":
        toks = body.split()
        stride = len(vprops)
        nv = vtx["count"]
        vals = toks[:nv * stride]
        arr = np.array(vals, dtype=float).reshape(nv, stride)
        verts = arr[:, [xi, yi, zi]]
        faces = []
        if fac is not None:
            pos = nv * stride
            for _ in range(min(fac["count"], max_faces)):
                if pos >= len(toks):
                    break
                k = int(float(toks[pos])); pos += 1
                faces.append(tuple(int(float(toks[pos + j])) for j in range(k)))
                pos += k
        return verts.astype(float), faces

    # binary
    endian = "<" if "little" in fmt else ">"
    npend = "<" if "little" in fmt else ">"
    vdt = np.dtype({"names": [f"f{i}" for i in range(len(vprops))],
                    "formats": [npend + _PLY_STRUCT[p[1]][1] for p in vprops]})
    nv = vtx["count"]
    varr = np.frombuffer(body, dtype=vdt, count=nv)
    verts = np.stack([varr[f"f{xi}"], varr[f"f{yi}"], varr[f"f{zi}"]], axis=1).astype(float)
    faces = []
    if fac is not None and fac["count"]:
        import struct
        off = nv * vdt.itemsize
        cnt_c = _PLY_STRUCT[fac["props"][0][1]][0]
        idx_c = _PLY_STRUCT[fac["props"][0][2]][0]
        cnt_sz = struct.calcsize(endian + cnt_c)
        for _ in range(min(fac["count"], max_faces)):
            if off + cnt_sz > len(body):
                break
            k = struct.unpack_from(endian + cnt_c, body, off)[0]; off += cnt_sz
            idx = struct.unpack_from(endian + idx_c * k, body, off)
            off += struct.calcsize(endian + idx_c * k)
            faces.append(tuple(int(i) for i in idx))
    return verts, faces


_MESH_CACHE: Dict[Tuple[str, str], Mesh] = {}


def load_mesh(path: str) -> Mesh:
    """Load (and memoise) an OBJ/PLY mesh by extension."""
    path = os.path.abspath(path)
    sha = file_sha(path)
    key = (path, sha)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        verts, faces = parse_obj(path)
    elif ext == ".ply":
        verts, faces = parse_ply(path)
    else:
        raise ValueError(f"Unsupported mesh type {ext!r} (use .obj or .ply)")
    if verts.size == 0:
        raise ValueError(f"No vertices found in {path}")
    mesh = Mesh(vertices=verts, faces=faces, sha=sha, path=path)
    _MESH_CACHE[key] = mesh
    return mesh


# --------------------------------------------------------------------------- #
# Footprint (XY-projected edges) for drawing
# --------------------------------------------------------------------------- #
def footprint_segments(mesh: Mesh, max_segments: int = 1500) -> List[List[float]]:
    """Unique mesh edges projected to the XY plane, as ``[x1,y1,x2,y2]``.

    Down-sampled to ``max_segments`` for responsive drawing.
    """
    v = mesh.vertices
    edges = set()
    for face in mesh.faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            if 0 <= a < len(v) and 0 <= b < len(v):
                edges.add((a, b) if a < b else (b, a))
    edges = list(edges)
    if len(edges) > max_segments:
        step = len(edges) / max_segments
        edges = [edges[int(i * step)] for i in range(max_segments)]
    segs = []
    for a, b in edges:
        segs.append([float(v[a, 0]), float(v[a, 1]), float(v[b, 0]), float(v[b, 1])])
    return segs


# --------------------------------------------------------------------------- #
# Mitsuba XML geometry: discover referenced meshes -> combined bounds/footprint
# --------------------------------------------------------------------------- #
def xml_mesh_paths(xml_path: str) -> List[str]:
    """Resolve mesh files referenced by a Mitsuba XML scene (best effort)."""
    base = os.path.dirname(os.path.abspath(xml_path))
    out = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return out
    for shape in root.iter("shape"):
        for s in shape.iter("string"):
            if s.get("name") == "filename":
                fn = s.get("value", "")
                if fn:
                    p = fn if os.path.isabs(fn) else os.path.join(base, fn)
                    if os.path.exists(p) and os.path.splitext(p)[1].lower() in (".ply", ".obj"):
                        out.append(p)
    return out


@dataclass(frozen=True)
class GeometryInfo:
    bounds: Dict[str, float]
    segments: List[List[float]]
    n_meshes: int


def geometry_info(path: str, max_segments: int = 1500) -> GeometryInfo:
    """Bounds + footprint for a geometry file (mesh or Mitsuba .xml)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        mesh_paths = xml_mesh_paths(path)
        if not mesh_paths:
            raise ValueError("No referenced .ply/.obj meshes found in this XML "
                             "(inline shapes are not parsed for the footprint; "
                             "geometry still ray-traces in Sionna).")
        allv = []
        segs: List[List[float]] = []
        per = max(50, max_segments // len(mesh_paths))
        errors = []
        for mp in mesh_paths:
            try:
                m = load_mesh(mp)
                allv.append(m.vertices)
                segs.extend(footprint_segments(m, per))
            except Exception as exc:  # skip a mesh we cannot read, keep the rest
                errors.append(f"{os.path.basename(mp)}: {exc}")
        if not allv:
            raise ValueError("Could not read any referenced mesh (" +
                             "; ".join(errors[:3]) + ")")
        v = np.vstack(allv)
        bounds = {
            "x_min": float(v[:, 0].min()), "x_max": float(v[:, 0].max()),
            "y_min": float(v[:, 1].min()), "y_max": float(v[:, 1].max()),
            "z_min": float(v[:, 2].min()), "z_max": float(v[:, 2].max()),
        }
        return GeometryInfo(bounds=bounds, segments=segs[:max_segments], n_meshes=len(allv))
    # plain mesh
    m = load_mesh(path)
    return GeometryInfo(bounds=m.bounds, segments=footprint_segments(m, max_segments), n_meshes=1)


# --------------------------------------------------------------------------- #
# Grid auto-fit
# --------------------------------------------------------------------------- #
def union_bounds(*bounds: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    bs = [b for b in bounds if b]
    if not bs:
        return None
    return {
        "x_min": min(b["x_min"] for b in bs), "x_max": max(b["x_max"] for b in bs),
        "y_min": min(b["y_min"] for b in bs), "y_max": max(b["y_max"] for b in bs),
        "z_min": min(b["z_min"] for b in bs), "z_max": max(b["z_max"] for b in bs),
    }


def grid_from_bounds(bounds: Dict[str, float], cell_size: float = 1.0,
                     z: Optional[float] = None, pad_frac: float = 0.03) -> GridSpec:
    """Build a :class:`GridSpec` that fits ``bounds`` (with a small margin)."""
    dx = bounds["x_max"] - bounds["x_min"]
    dy = bounds["y_max"] - bounds["y_min"]
    px, py = dx * pad_frac, dy * pad_frac
    if z is None:
        z = bounds["z_min"] + 0.1 * (bounds["z_max"] - bounds["z_min"])  # near floor
    return GridSpec(
        x_min=bounds["x_min"] - px, x_max=bounds["x_max"] + px,
        y_min=bounds["y_min"] - py, y_max=bounds["y_max"] + py,
        z=float(z), cell_size=cell_size,
    )
