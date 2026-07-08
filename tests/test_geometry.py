"""Tests for geometry/mesh loading and grid auto-fit (pure Python)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wifisim import geometry as geo
from wifisim.models import GridSpec


def _write_obj(path: Path):
    # a unit box from (0,0,0) to (10,6,3)
    verts = [(0, 0, 0), (10, 0, 0), (10, 6, 0), (0, 6, 0),
             (0, 0, 3), (10, 0, 3), (10, 6, 3), (0, 6, 3)]
    faces = [(1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"f {' '.join(map(str, fa))}" for fa in faces]
    path.write_text("\n".join(lines) + "\n")


def _write_ply(path: Path):
    verts = [(-5, -5, 1), (5, -5, 1), (5, 5, 1), (-5, 5, 1)]
    faces = [(0, 1, 2), (0, 2, 3)]
    hdr = ["ply", "format ascii 1.0", f"element vertex {len(verts)}",
           "property float x", "property float y", "property float z",
           f"element face {len(faces)}", "property list uchar int vertex_indices",
           "end_header"]
    body = [f"{x} {y} {z}" for x, y, z in verts] + [f"3 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(hdr + body) + "\n")


def test_parse_obj_bounds():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "box.obj"
        _write_obj(f)
        m = geo.load_mesh(str(f))
        b = m.bounds
        assert (b["x_min"], b["x_max"]) == (0.0, 10.0)
        assert (b["y_min"], b["y_max"]) == (0.0, 6.0)
        assert (b["z_min"], b["z_max"]) == (0.0, 3.0)


def test_parse_ply_bounds_and_footprint():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "plane.ply"
        _write_ply(f)
        m = geo.load_mesh(str(f))
        assert m.bounds["x_min"] == -5.0 and m.bounds["x_max"] == 5.0
        segs = geo.footprint_segments(m)
        assert len(segs) >= 4 and all(len(s) == 4 for s in segs)


def test_grid_from_bounds_fits_and_aspect():
    bounds = {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 6, "z_min": 0, "z_max": 3}
    g = geo.grid_from_bounds(bounds, cell_size=0.5, z=1.5)
    assert isinstance(g, GridSpec)
    # grid spans the bounds (plus small pad)
    assert g.x_min <= 0 and g.x_max >= 10 and g.y_min <= 0 and g.y_max >= 6
    assert g.z == 1.5


def test_grid_from_bounds_uses_whole_metre_axis_limits():
    # Fractional bounds (as produced by e.g. a mesh with odd-shaped geometry)
    # must still snap to whole-metre axis limits with an exact-multiple span,
    # so the engine's own size/cell_size cell count can never drift by one
    # from GridSpec.nx/ny (see wifisim/combine.py ValueError regression).
    bounds = {"x_min": -12.37, "x_max": 44.86, "y_min": 3.14, "y_max": 58.02,
              "z_min": 0, "z_max": 3}
    for cell_size in (0.5, 0.3, 1.0, 0.25):
        g = geo.grid_from_bounds(bounds, cell_size=cell_size, z=1.5)
        assert g.x_min == int(g.x_min) and g.x_max == int(g.x_max)
        assert g.y_min == int(g.y_min) and g.y_max == int(g.y_max)
        assert g.x_min <= bounds["x_min"] and g.x_max >= bounds["x_max"]
        assert g.y_min <= bounds["y_min"] and g.y_max >= bounds["y_max"]
        nx = (g.x_max - g.x_min) / cell_size
        ny = (g.y_max - g.y_min) / cell_size
        assert abs(nx - round(nx)) < 1e-6            # exact multiple, no off-by-one
        assert abs(ny - round(ny)) < 1e-6


def test_union_bounds():
    a = {"x_min": 0, "x_max": 5, "y_min": 0, "y_max": 5, "z_min": 0, "z_max": 2}
    b = {"x_min": -3, "x_max": 2, "y_min": 1, "y_max": 9, "z_min": 1, "z_max": 4}
    u = geo.union_bounds(a, b, None)
    assert (u["x_min"], u["x_max"], u["y_min"], u["y_max"]) == (-3, 5, 0, 9)


def test_geometry_info_mesh():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "box.obj"
        _write_obj(f)
        info = geo.geometry_info(str(f))
        assert info.n_meshes == 1 and info.bounds["x_max"] == 10.0 and info.segments


def test_sha_changes_with_content():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "m.obj"
        _write_obj(f)
        s1 = geo.file_sha(str(f))
        f.write_text(f.read_text() + "v 1 1 1\n")
        assert geo.file_sha(str(f)) != s1


def _write_binary_ply(path, endian="<"):
    import struct
    verts = [(-5, -3, 0, 0.1), (7, -3, 0, 0.2), (7, 4, 2, 0.3), (-5, 4, 2, 0.4)]
    faces = [(0, 1, 2), (0, 2, 3)]
    tag = "little" if endian == "<" else "big"
    hdr = (f"ply\nformat binary_{tag}_endian 1.0\nelement vertex {len(verts)}\n"
           "property float x\nproperty float y\nproperty float z\nproperty float nx\n"
           f"element face {len(faces)}\n"
           "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode())
        for x, y, z, nx in verts:
            f.write(struct.pack(endian + "ffff", x, y, z, nx))
        for fa in faces:
            f.write(struct.pack(endian + "B", len(fa)) +
                    struct.pack(endian + f"{len(fa)}i", *fa))


def test_binary_ply_little_and_big_endian():
    with tempfile.TemporaryDirectory() as d:
        for endian, nm in (("<", "le.ply"), (">", "be.ply")):
            p = Path(d) / nm
            _write_binary_ply(p, endian)
            m = geo.load_mesh(str(p))          # extra 'nx' property must be skipped
            assert m.bounds["x_min"] == -5 and m.bounds["x_max"] == 7
            assert m.bounds["z_max"] == 2
            assert len(geo.footprint_segments(m)) > 0


def test_xml_references_binary_ply():
    with tempfile.TemporaryDirectory() as d:
        _write_binary_ply(Path(d) / "mesh.ply", "<")
        (Path(d) / "scene.xml").write_text(
            '<scene version="2.1.0"><shape type="ply">'
            '<string name="filename" value="mesh.ply"/></shape></scene>')
        info = geo.geometry_info(str(Path(d) / "scene.xml"))
        assert info.n_meshes == 1 and info.bounds["x_max"] == 7 and info.segments


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} geometry tests passed")
