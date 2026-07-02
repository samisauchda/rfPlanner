"""Triangle-mesh extraction for the web planner's 3D view.

Parses geometry files into plain ``(vertices, triangles)`` arrays that the
browser can render with three.js:

* ``.ply``  -- ASCII and binary (little/big endian); extra vertex properties
  (normals, colours, ...) are skipped, polygon faces are fan-triangulated.
* ``.obj``  -- ASCII; ``v`` / ``f`` records, ``v/vt/vn`` face tokens and
  negative indices supported, polygons fan-triangulated.
* ``.xml``  -- Mitsuba scenes: every ``<shape>`` with a ``filename`` string
  parameter referencing a ``.ply``/``.obj`` (relative to the XML) is loaded.

This module is deliberately independent of :mod:`wifisim.geometry` (which
only needs bounds/footprints) so the 3D payload does not depend on that
module's internals.
"""
from __future__ import annotations

import os
import struct
from typing import List, Tuple
from xml.etree import ElementTree

import numpy as np

__all__ = ["load_meshes", "mesh_payload"]

Mesh = Tuple[np.ndarray, np.ndarray]        # (verts (N,3) f32, tris (M,3) i32)

_PLY_SIZES = {
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
    "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    "float": 4, "float32": 4, "double": 8, "float64": 8,
}
_PLY_STRUCT = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def _fan(indices: List[int], out: List[List[int]]) -> None:
    for k in range(1, len(indices) - 1):
        out.append([indices[0], indices[k], indices[k + 1]])


# --------------------------------------------------------------------------- #
# PLY
# --------------------------------------------------------------------------- #
def parse_ply(path: str) -> Mesh:
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no PLY end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    body = raw[end:]

    fmt = "ascii"
    elements: List[dict] = []                # {name, count, props:[(kind, ...)]}
    for line in header:
        toks = line.strip().split()
        if not toks:
            continue
        if toks[0] == "format":
            fmt = {"ascii": "ascii", "binary_little_endian": "<",
                   "binary_big_endian": ">"}.get(toks[1], "ascii")
        elif toks[0] == "element":
            elements.append({"name": toks[1], "count": int(toks[2]), "props": []})
        elif toks[0] == "property" and elements:
            if toks[1] == "list":
                elements[-1]["props"].append(("list", toks[2], toks[3], toks[4]))
            else:
                elements[-1]["props"].append(("scalar", toks[1], toks[2]))

    verts: List[List[float]] = []
    tris: List[List[int]] = []

    if fmt == "ascii":
        lines = body.decode("ascii", errors="replace").split("\n")
        li = 0
        for elem in elements:
            names = [p[2] for p in elem["props"] if p[0] == "scalar"]
            for _ in range(elem["count"]):
                while li < len(lines) and not lines[li].strip():
                    li += 1
                toks = lines[li].split(); li += 1
                if elem["name"] == "vertex":
                    vals = dict(zip(names, toks))
                    verts.append([float(vals.get("x", 0)), float(vals.get("y", 0)),
                                  float(vals.get("z", 0))])
                elif elem["name"] == "face":
                    n = int(float(toks[0]))
                    _fan([int(float(t)) for t in toks[1:1 + n]], tris)
    else:
        off = 0
        for elem in elements:
            if elem["name"] == "vertex":
                # byte offsets of x/y/z inside one vertex record
                stride, pos = 0, {}
                for p in elem["props"]:
                    if p[0] == "list":
                        raise ValueError(f"{path}: list property in vertex element")
                    if p[2] in ("x", "y", "z"):
                        pos[p[2]] = (stride, p[1])
                    stride += _PLY_SIZES[p[1]]
                for i in range(elem["count"]):
                    base = off + i * stride
                    v = []
                    for ax in ("x", "y", "z"):
                        o, typ = pos[ax]
                        v.append(struct.unpack_from(fmt + _PLY_STRUCT[typ],
                                                    body, base + o)[0])
                    verts.append(v)
                off += elem["count"] * stride
            else:
                for _ in range(elem["count"]):
                    for p in elem["props"]:
                        if p[0] == "list":
                            n = struct.unpack_from(fmt + _PLY_STRUCT[p[1]], body, off)[0]
                            off += _PLY_SIZES[p[1]]
                            idx = struct.unpack_from(fmt + str(n) + _PLY_STRUCT[p[2]],
                                                     body, off)
                            off += n * _PLY_SIZES[p[2]]
                            if elem["name"] == "face" and p[3] in (
                                    "vertex_indices", "vertex_index"):
                                _fan(list(idx), tris)
                        else:
                            off += _PLY_SIZES[p[1]]

    return (np.asarray(verts, dtype=np.float32),
            np.asarray(tris, dtype=np.int32).reshape(-1, 3))


# --------------------------------------------------------------------------- #
# OBJ
# --------------------------------------------------------------------------- #
def parse_obj(path: str) -> Mesh:
    verts: List[List[float]] = []
    tris: List[List[int]] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            if toks[0] == "v" and len(toks) >= 4:
                verts.append([float(toks[1]), float(toks[2]), float(toks[3])])
            elif toks[0] == "f" and len(toks) >= 4:
                idx = []
                for t in toks[1:]:
                    i = int(t.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                _fan(idx, tris)
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(tris, dtype=np.int32).reshape(-1, 3))


# --------------------------------------------------------------------------- #
# Files -> meshes
# --------------------------------------------------------------------------- #
def load_meshes(path: str) -> List[Mesh]:
    """All triangle meshes contained in / referenced by ``path``."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return [parse_ply(path)]
    if ext == ".obj":
        return [parse_obj(path)]
    if ext == ".xml":
        base = os.path.dirname(os.path.abspath(path))
        meshes: List[Mesh] = []
        root = ElementTree.parse(path).getroot()
        for shape in root.iter("shape"):
            for s in shape.iter("string"):
                if s.get("name") != "filename":
                    continue
                ref = s.get("value") or ""
                p = ref if os.path.isabs(ref) else os.path.join(base, ref)
                sub = os.path.splitext(p)[1].lower()
                if os.path.exists(p) and sub in (".ply", ".obj"):
                    try:
                        meshes.append(parse_ply(p) if sub == ".ply" else parse_obj(p))
                    except Exception:
                        pass                 # skip unreadable shapes, keep the rest
        return meshes
    raise ValueError(f"unsupported geometry file: {path}")


def mesh_payload(path: str, max_tris: int = 120_000) -> List[dict]:
    """JSON-friendly meshes: flat vertex/index arrays, decimated if huge.

    Decimation keeps every k-th triangle so very large scenes still render
    (approximately) without megabytes of JSON.
    """
    out = []
    for verts, tris in load_meshes(path):
        n = len(tris)
        if n > max_tris:
            tris = tris[:: int(np.ceil(n / max_tris))]
        out.append({
            "vertices": np.round(verts, 3).ravel().tolist(),
            "faces": tris.ravel().tolist(),
            "n_vertices": int(len(verts)),
            "n_faces": int(len(tris)),
            "decimated": bool(n > max_tris),
        })
    return out
