"""Export the built piano scene to a Rerun ``.rrd`` you can open anywhere.

Isaac renders pixels; Rerun wants *geometry*. So we read each link's visual
prims straight off the USD stage (points/indices, or the analytic gprims the
piano keys are built from) and express them in that link's OWN body frame --
which is rigid, so the geometry is logged once as static and only a per-link
transform changes over time. Link world poses come from PhysX via the Isaac Lab
``Articulation`` views, so what you see in Rerun is the simulated state, not the
authored rest layout.

Used by ``render_server.py``'s ``rerun`` job (see ``scripts/render/render.py``).
"""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

from dexsim.piano import key_is_black

# link geometry is rigid in its own frame -> resolve mesh<-link once, at rest.
_TIME = Usd.TimeCode.Default()

# skip authored collision/guide geometry; we only want what a camera would see.
_SKIP_PURPOSES = ("guide", "proxy")

WHITE_KEY_RGB = (235, 235, 232)
BLACK_KEY_RGB = (28, 28, 32)
LEFT_ARM_RGB = (168, 172, 180, 255)
RIGHT_ARM_RGB = (168, 172, 180, 255)
PIANO_BODY_RGB = (90, 70, 60)


# --------------------------------------------------------------------------- #
# USD geometry -> triangles
# --------------------------------------------------------------------------- #
def _fan_triangulate(counts, indices):
    """Fan-triangulate arbitrary polygons (USD meshes are often quads)."""
    tris, off = [], 0
    for c in counts:
        c = int(c)
        for k in range(1, c - 1):
            tris.append((indices[off], indices[off + k], indices[off + k + 1]))
        off += c
    return np.asarray(tris, dtype=np.uint32).reshape(-1, 3)


def _box(hx, hy, hz):
    v = np.array([(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
                  (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
                 dtype=np.float64)
    f = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                  (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
                  (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)], dtype=np.uint32)
    return v, f


def _cylinder(radius, height, axis="Z", seg=16):
    a = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    c, s, h = np.cos(a) * radius, np.sin(a) * radius, height / 2.0
    ring = np.stack([np.concatenate([c, c]), np.stack([s, s]).reshape(-1),
                     np.concatenate([np.full(seg, -h), np.full(seg, h)])], axis=1)
    verts = np.vstack([ring, [(0, 0, -h), (0, 0, h)]])
    f = []
    for i in range(seg):
        j = (i + 1) % seg
        f += [(i, j, seg + j), (i, seg + j, seg + i)]           # side
        f += [(2 * seg, j, i), (2 * seg + 1, seg + i, seg + j)]  # caps
    verts = _reaxis(verts, axis)
    return verts, np.asarray(f, dtype=np.uint32)


def _sphere(radius, lat=8, lon=16):
    v, f = [], []
    for i in range(lat + 1):
        th = np.pi * i / lat
        for j in range(lon):
            ph = 2.0 * np.pi * j / lon
            v.append((radius * np.sin(th) * np.cos(ph),
                      radius * np.sin(th) * np.sin(ph), radius * np.cos(th)))
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            f += [(a, b, a + lon), (b, b + lon, a + lon)]
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.uint32)


def _reaxis(verts, axis):
    """USD cylinders/capsules carry an axis attr; our builder emits Z."""
    if axis == "X":
        return verts[:, [2, 1, 0]]
    if axis == "Y":
        return verts[:, [0, 2, 1]]
    return verts


def _prim_geometry(prim):
    """(verts, tris) in the prim's own local frame, or None if not renderable."""
    if UsdGeom.Imageable(prim).ComputePurpose() in _SKIP_PURPOSES:
        return None
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            return None
        return np.asarray(pts, dtype=np.float64), _fan_triangulate(counts, np.asarray(idx))
    if prim.IsA(UsdGeom.Cube):
        h = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
        return _box(h, h, h)
    if prim.IsA(UsdGeom.Sphere):
        return _sphere(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0))
    if prim.IsA(UsdGeom.Cylinder):
        cy = UsdGeom.Cylinder(prim)
        return _cylinder(float(cy.GetRadiusAttr().Get() or 1.0),
                         float(cy.GetHeightAttr().Get() or 2.0),
                         str(cy.GetAxisAttr().Get() or "Z"))
    if prim.IsA(UsdGeom.Capsule):                       # body only -- close enough
        ca = UsdGeom.Capsule(prim)
        r = float(ca.GetRadiusAttr().Get() or 0.5)
        return _cylinder(r, float(ca.GetHeightAttr().Get() or 1.0) + 2 * r,
                         str(ca.GetAxisAttr().Get() or "Z"))
    return None


def _material_color(prim):
    """Best-effort USD PreviewSurface/display color as an 8-bit RGBA value."""
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if mat:
        output = mat.ComputeSurfaceSource()
        shader = output[0] if output else None
        if shader:
            for name in ("diffuseColor", "diffuse_color", "base_color"):
                inp = shader.GetInput(name)
                value = inp.Get() if inp else None
                if value is not None:
                    rgb = np.clip(np.asarray(value, dtype=float)[:3], 0.0, 1.0)
                    opacity = shader.GetInput("opacity")
                    alpha = float(opacity.Get()) if opacity and opacity.Get() is not None else 1.0
                    return tuple(np.rint(np.r_[rgb, np.clip(alpha, 0, 1)] * 255).astype(np.uint8))
    imageable = UsdGeom.Gprim(prim)
    colors = imageable.GetDisplayColorAttr().Get() if imageable else None
    if colors:
        rgb = np.clip(np.asarray(colors[0], dtype=float)[:3], 0.0, 1.0)
        return tuple(np.rint(np.r_[rgb, 1.0] * 255).astype(np.uint8))
    return None


# --------------------------------------------------------------------------- #
# stage -> per-link triangle soup
# --------------------------------------------------------------------------- #
def _link_prims(stage, root_path, body_names):
    """{body_name: prim} for the articulation's links, matched by prim name."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {}
    wanted, found = set(body_names), {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        n = prim.GetName()
        if n in wanted and n not in found:
            found[n] = prim
    return found


def _gather(prim, link_prims_set, xf_cache, link_world_inv):
    """Per-material triangle soups under a link, expressed in its body frame."""
    groups = {}
    stack = [prim]
    while stack:
        p = stack.pop()
        geo = _prim_geometry(p)
        if geo is not None:
            v, f = geo
            # USD is row-vector: v_world = v_local * M. Compose mesh->world->link.
            rel = xf_cache.GetLocalToWorldTransform(p) * link_world_inv
            m = np.asarray(rel, dtype=np.float64)
            v = (np.hstack([v, np.ones((len(v), 1))]) @ m)[:, :3]
            color = _material_color(p)
            material = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
            material_path = str(material.GetPath()) if material else "displayColor"
            group = groups.setdefault((material_path, color), [[], [], 0])
            group[0].append(v)
            group[1].append(f + group[2])
            group[2] += len(v)
        for c in p.GetFilteredChildren(Usd.TraverseInstanceProxies()):
            if c not in link_prims_set:                 # don't steal a child link's geometry
                stack.append(c)
    if not groups:
        return None
    return [(np.vstack(v).astype(np.float32), np.vstack(f).astype(np.uint32), color)
            for (_, color), (v, f, _) in groups.items()]


def collect_link_meshes(stage, art, root_path):
    """{body_name: [(verts, tris, RGBA), ...]} in each link's frame."""
    names = list(art.body_names)
    link_prims = _link_prims(stage, root_path, names)
    prim_set = set(link_prims.values())
    xf_cache = UsdGeom.XformCache(_TIME)
    out = {}
    for name, prim in link_prims.items():
        inv = xf_cache.GetLocalToWorldTransform(prim).GetInverse()
        got = _gather(prim, prim_set, xf_cache, inv)
        if got is not None:
            out[name] = got
    return out


# --------------------------------------------------------------------------- #
# scene dump  (Isaac venv writes this; the rerun venv turns it into a .rrd)
# --------------------------------------------------------------------------- #
# Isaac Sim 4.5 pins numpy<2 and rerun-sdk>=0.36 demands numpy>=2, so the two
# CANNOT live in one venv. We hand off through a plain npz instead: this side
# (pxr + Isaac) writes geometry/poses, scripts/render/rrd_from_dump.py (rerun
# venv, no Isaac import) writes the .rrd.
def _key_color(body_name):
    """Piano links are key_<i> with i the 0..87 key index; else it's the case."""
    if not body_name.startswith("key_"):
        return PIANO_BODY_RGB
    try:
        idx = int(body_name.split("_")[1])
    except (IndexError, ValueError):
        return PIANO_BODY_RGB
    return BLACK_KEY_RGB if key_is_black(idx) else WHITE_KEY_RGB


def capture_poses(art, meshes, env=0):
    """{body_name: (pos(3,), quat_wxyz(4,))} for one env, straight from PhysX."""
    pos = art.data.body_pos_w[env].cpu().numpy()
    quat = art.data.body_quat_w[env].cpu().numpy()        # Isaac is wxyz
    names = list(art.body_names)
    return {n: (pos[names.index(n)].astype(np.float32),
                quat[names.index(n)].astype(np.float32)) for n in meshes}


class SceneDump:
    """Accumulates static geometry + one pose row per captured frame."""

    def __init__(self):
        self.arrays = {}          # npz payload
        self.links = []           # "<root>/<link>" in log order
        self.frames = {}          # key -> list of (pos, quat), one per frame
        self.n_frames = 0

    def add_geometry(self, root, meshes, color=None):
        for name, parts in meshes.items():
            link = f"{root}/{name}"
            self.links.append(link)
            self.frames[link] = []
            self.arrays[f"{link}|parts"] = np.asarray(len(parts), dtype=np.int32)
            for i, (verts, tris, material_color) in enumerate(parts):
                key = f"{link}/visual_{i}"
                self.arrays[f"{key}|verts"] = verts
                self.arrays[f"{key}|tris"] = tris
                # Preserve authored USD materials. The explicit color argument is
                # retained only as a fallback for old/procedural unmaterialed scenes.
                fallback = color or _key_color(name)
                self.arrays[f"{key}|color"] = np.asarray(material_color or fallback,
                                                         dtype=np.uint8)

    def add_frame(self, root, art, meshes, env=0):
        for name, (pos, quat) in capture_poses(art, meshes, env).items():
            self.frames[f"{root}/{name}"].append((pos, quat))

    def end_frame(self):
        self.n_frames += 1

    def save(self, path, control_dt=0.0):
        for key, rows in self.frames.items():
            self.arrays[f"{key}|pos"] = np.stack([r[0] for r in rows])
            self.arrays[f"{key}|quat"] = np.stack([r[1] for r in rows])
        self.arrays["__links__"] = np.asarray(self.links)
        self.arrays["__meta__"] = np.asarray(
            [str(self.n_frames), str(control_dt)])
        np.savez_compressed(path, **self.arrays)
        return {"links": len(self.links), "frames": self.n_frames,
                "tris": int(sum(len(self.arrays[f'{k}/visual_{i}|tris'])
                                for k in self.links
                                for i in range(int(self.arrays[f'{k}|parts']))))}


ROOTS = {"piano": "world/piano", "left": "world/left_hand", "right": "world/right_hand"}
COLORS = {"piano": None, "left": LEFT_ARM_RGB, "right": RIGHT_ARM_RGB}
