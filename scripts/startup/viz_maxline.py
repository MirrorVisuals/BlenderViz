# SPDX-License-Identifier: GPL-3.0-or-later
"""
Max Line - 3ds Max style direct spline drawing for Blender.

Click in the viewport to place corner (vector) points, click-drag to pull
bezier handles. Points land on geometry under the mouse, or on a fallback
construction plane when there is nothing there. Hold Ctrl to snap using the
scene's own snap element settings.

Built into BlenderViz (it started as the "Max Line" extension). It is loaded at
startup like Blender's own tools, so it is always available: Add > Curve, the
Tool tab of the sidebar, and Shift+Alt+L in Object Mode (in the default keymap).
"""


import math

import bpy
import blf
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from mathutils.geometry import interpolate_bezier, intersect_line_plane

NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
    'WHEELINMOUSE', 'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM',
    'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
    'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9',
    'NUMPAD_PERIOD', 'NUMPAD_PLUS', 'NUMPAD_MINUS', 'NUMPAD_SLASH',
}

COL_CURVE = (0.10, 0.85, 1.00, 1.0)
COL_PREVIEW = (1.00, 0.70, 0.15, 0.9)
COL_HANDLE = (0.75, 0.45, 0.95, 0.9)
COL_KNOT = (0.95, 0.95, 0.95, 1.0)
COL_KNOT_BEZ = (0.35, 1.00, 0.45, 1.0)
COL_FIRST = (1.00, 0.35, 0.35, 1.0)
COL_SNAP = (1.00, 0.90, 0.20, 1.0)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def tag_view3d_redraw(context):
    wm = getattr(context, "window_manager", None) or bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _touch_preview(self, context):
    tag_view3d_redraw(context)


def _touch_fillet(self, context):
    # Chamfer stays selected once used, so its preview keeps showing.
    if self.preview_mode != 'CHAMFER':
        self.preview_mode = 'FILLET'
    tag_view3d_redraw(context)


def _touch_outline(self, context):
    self.preview_mode = 'OUTLINE'
    tag_view3d_redraw(context)


MEASURE_ITEMS = [
    ('RADIUS', "Radius", "Amount is the radius of the fillet arc"),
    ('DISTANCE', "Distance", "Amount is the distance trimmed back along each "
                             "adjacent segment, as 3ds Max does"),
]


class MAXLINE_PG_settings(bpy.types.PropertyGroup):
    fallback_plane: bpy.props.EnumProperty(
        name="Empty Space",
        description="Where points land when there is no geometry under the mouse",
        items=[
            ('GROUND', "Ground Plane", "World XY plane through the origin (Z = 0)"),
            ('CURSOR_Z', "Cursor Height", "World XY plane through the 3D cursor"),
            ('VIEW', "View Plane", "Plane facing the viewer, through the 3D cursor"),
        ],
        default='GROUND',
    )
    surface_offset: bpy.props.FloatProperty(
        name="Surface Offset",
        description="Lift points off the surface along its normal, to avoid z-fighting",
        default=0.0, min=0.0, soft_max=0.5, step=1, precision=4, unit='LENGTH',
    )
    continuous: bpy.props.BoolProperty(
        name="Keep Tool Active",
        description="After finishing a spline, immediately start a new one. Esc exits",
        default=False,
    )
    attach_to_active: bpy.props.BoolProperty(
        name="Add to Active Curve",
        description="Append the spline to the active curve object instead of creating a new "
                    "one. Equivalent to unchecking 'Start New Shape' in 3ds Max",
        default=False,
    )
    align_mode: bpy.props.EnumProperty(
        name="Align",
        description="Which axes the new object gets",
        items=[
            ('WORLD', "World",
             "Keep the world axes, whatever was drawn on"),
            ('OBJECT', "Surface Object",
             "Match the axes and scale of the object drawn on. With an Origin "
             "of Surface Object this gives local coordinates identical to that "
             "object's"),
            ('NORMAL', "Surface Normal",
             "Point Z along the surface normal where it was drawn, so the "
             "curve lies flat on that face. Rotation only, no inherited scale"),
        ],
        default='OBJECT',
    )
    parent_to_surface: bpy.props.BoolProperty(
        name="Parent to Surface Object",
        description="Parent the spline to the object it was drawn on, so it follows when "
                    "that object is moved",
        default=False,
    )
    origin_mode: bpy.props.EnumProperty(
        name="Origin",
        description="Where a drawn line's origin is placed. Circle, Rectangle "
                    "and NGon always use their own centre",
        items=[
            ('SURFACE', "Surface Object",
             "At the origin of the object drawn on. With Align to Surface Object this "
             "gives the spline that object's exact local coordinates. Falls back to "
             "First Point when the spline touches no geometry"),
            ('FIRST', "First Point", "At the first point you clicked, as 3ds Max does"),
            ('BBOX', "Bounding Box Center", "Center of the spline's bounding box"),
            ('MEDIAN', "Median Point", "Average of all the knots"),
            ('CURSOR', "3D Cursor", "At the 3D cursor"),
            ('WORLD', "World Origin", "At (0, 0, 0)"),
        ],
        default='SURFACE',
    )
    snap_radius: bpy.props.IntProperty(
        name="Snap Radius",
        description="Search radius in pixels for snapping",
        default=14, min=2, max=80, subtype='PIXEL',
    )
    close_radius: bpy.props.IntProperty(
        name="Close Radius",
        description="Clicking within this many pixels of the first point closes the spline",
        default=12, min=2, max=80, subtype='PIXEL',
    )
    drag_threshold: bpy.props.IntProperty(
        name="Drag Threshold",
        description="Mouse travel in pixels before a click becomes a bezier drag",
        default=4, min=1, max=40, subtype='PIXEL',
    )
    grid_increment: bpy.props.FloatProperty(
        name="Increment",
        description="Step size used for Increment / Grid snapping",
        default=1.0, min=0.0001, soft_max=100.0, unit='LENGTH',
    )
    use_occlusion: bpy.props.BoolProperty(
        name="Ignore Hidden Geometry",
        description="Skip snap candidates that sit behind other surfaces. Automatically "
                    "bypassed in X-ray mode",
        default=True,
    )
    curve_resolution: bpy.props.IntProperty(
        name="Resolution",
        description="Resolution U of the created curve",
        default=12, min=1, max=64,
    )
    fillet_amount: bpy.props.FloatProperty(
        name="Amount",
        description="Fillet radius, or trim distance in Distance mode. Returns "
                    "to zero after each use, as the 3ds Max spinner does",
        default=0.0, min=0.0, soft_max=100.0, subtype='DISTANCE',
        update=_touch_fillet,
    )
    fillet_measure: bpy.props.EnumProperty(
        name="Measure", description="How the amount is interpreted",
        items=MEASURE_ITEMS, default='RADIUS', update=_touch_fillet,
    )
    outline_distance: bpy.props.FloatProperty(
        name="Distance",
        description="Offset distance in the object's local XY plane. Returns to "
                    "zero after each use",
        default=0.0, soft_min=-100.0, soft_max=100.0, subtype='DISTANCE',
        update=_touch_outline,
    )
    outline_center: bpy.props.BoolProperty(
        name="Center", description="Offset both ways by half, keeping the "
                                   "original centred", default=False,
        update=_touch_outline,
    )
    outline_close_ends: bpy.props.BoolProperty(
        name="Close Ends", description="On open splines, join the two ends into "
                                       "one closed shape", default=True,
        update=_touch_outline,
    )
    ngon_sides: bpy.props.IntProperty(
        name="Sides", description="Number of corners for a new NGon",
        default=6, min=3, max=64,
    )
    ngon_inscribed: bpy.props.BoolProperty(
        name="Inscribed",
        description="Corners sit on the radius. Off puts the edges there "
                    "instead, as 3ds Max's Circumscribed does",
        default=True,
    )
    shape_corner_radius: bpy.props.FloatProperty(
        name="Corner Radius",
        description="Round the corners of a new Rectangle or NGon",
        default=0.0, min=0.0, soft_max=100.0, subtype='DISTANCE',
    )
    arc_method: bpy.props.EnumProperty(
        name="Arc Method",
        description="How a new arc is dragged out",
        items=[
            ('ENDS', "End-End-Middle", "Drag the two ends, then the bulge"),
            ('CENTER', "Center-End-End", "Centre, then radius, then sweep"),
        ],
        default='ENDS',
    )
    arc_pie: bpy.props.BoolProperty(
        name="Pie Slice",
        description="Close a new arc back through its centre",
        default=False,
    )
    rect_from_center: bpy.props.BoolProperty(
        name="From Center",
        description="Drag a rectangle out from its middle instead of a corner",
        default=False,
    )
    show_preview: bpy.props.BoolProperty(
        name="Live Preview",
        description="Show the result of the values above in the viewport before "
                    "you apply them",
        default=True, update=_touch_preview,
    )
    preview_mode: bpy.props.EnumProperty(
        name="Preview Mode",
        items=[('FILLET', "Fillet", ""), ('CHAMFER', "Chamfer", ""),
               ('OUTLINE', "Outline", ""), ('NONE', "None", "")],
        default='FILLET', options={'HIDDEN'},
    )
    use_drag: bpy.props.BoolProperty(
        name="Drag to Set Amount",
        description="Make the buttons above start a viewport drag instead of "
                    "applying at once. Confirm the drag with Enter or a click "
                    "in the viewport",
        default=False,
    )
    max_snap_verts: bpy.props.IntProperty(
        name="Snap Vertex Limit",
        description="Stop collecting snap geometry after this many vertices, to keep heavy "
                    "scenes responsive",
        default=1500000, min=1000, soft_max=5000000,
    )


def settings(context):
    return context.scene.maxline_settings


# ---------------------------------------------------------------------------
# Spline point
# ---------------------------------------------------------------------------

class SplinePoint:
    """One knot of the spline being drawn, in world space."""

    __slots__ = ("co", "hl", "hr", "htype", "obj", "nor")

    def __init__(self, co, htype='VECTOR'):
        self.co = co.copy()
        self.hl = co.copy()
        self.hr = co.copy()
        self.htype = htype
        self.obj = None
        self.nor = None

    def set_drag(self, anchor, dragged):
        """Mirror the dragged position around the knot, like Max's bezier drag."""
        self.hr = dragged.copy()
        self.hl = anchor * 2.0 - dragged
        self.htype = 'ALIGNED'

    def clear_drag(self):
        self.hl = self.co.copy()
        self.hr = self.co.copy()
        self.htype = 'VECTOR'


def effective_handles(points, closed):
    """Resolve VECTOR handles the way Blender does, so the preview matches the result."""
    n = len(points)
    out = []
    for i, p in enumerate(points):
        if p.htype != 'VECTOR':
            out.append((p.hl.copy(), p.hr.copy()))
            continue
        prev = points[(i - 1) % n] if (closed or i > 0) else None
        nxt = points[(i + 1) % n] if (closed or i < n - 1) else None
        if prev is not None:
            hl = p.co + (prev.co - p.co) / 3.0
        elif nxt is not None:
            hl = p.co - (nxt.co - p.co) / 3.0
        else:
            hl = p.co.copy()
        if nxt is not None:
            hr = p.co + (nxt.co - p.co) / 3.0
        elif prev is not None:
            hr = p.co - (prev.co - p.co) / 3.0
        else:
            hr = p.co.copy()
        out.append((hl, hr))
    return out


def spline_segments(points, closed, resolution):
    """Return one polyline (list of Vectors) per curve segment."""
    n = len(points)
    if n < 2:
        return []
    handles = effective_handles(points, closed)
    count = n if closed else n - 1
    res = max(2, resolution)
    segments = []
    for i in range(count):
        a, b = points[i], points[(i + 1) % n]
        if a.htype == 'VECTOR' and b.htype == 'VECTOR':
            segments.append([a.co.copy(), b.co.copy()])
        else:
            segments.append(list(interpolate_bezier(
                a.co, handles[i][1], handles[(i + 1) % n][0], b.co, res)))
    return segments


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------

def snap_element_prop(tool_settings):
    """Blender has renamed this property over time; pick whichever exists."""
    if hasattr(tool_settings, "snap_elements_base"):
        return "snap_elements_base"
    return "snap_elements"


def snap_elements(context):
    """Read the scene's snap element set across the naming variants Blender has used."""
    ts = context.scene.tool_settings
    found = set()
    for attr in ("snap_elements_base", "snap_elements", "snap_elements_individual"):
        value = getattr(ts, attr, None)
        if value:
            found |= set(value)
    return found


def snapping_active(context, event):
    """Mirror Blender's convention: Ctrl inverts the snap toggle."""
    return bool(context.scene.tool_settings.use_snap) != bool(event.ctrl)


class SnapCache:
    """World-space vertex and edge soup of the visible scene, projected with numpy."""

    def __init__(self):
        self.verts = None      # (N, 3) float64
        self.edges = None      # (E, 2) int32
        self.midpoints = None  # (E, 3) float64, built lazily
        self.truncated = False
        # Built only when snapping is first used, then kept until the scene changes
        # (see `_snap_cache_invalidate`), so starting a shape costs nothing.
        self.dirty = True

    def ensure(self, context):
        if self.dirty:
            self.build(context)
            self.dirty = False

    def build(self, context):
        self.verts = None
        self.edges = None
        self.midpoints = None
        self.truncated = False

        ts = context.scene.tool_settings
        only_selectable = bool(getattr(ts, "use_snap_selectable", False))
        limit = settings(context).max_snap_verts

        depsgraph = context.evaluated_depsgraph_get()
        vert_chunks = []
        edge_chunks = []
        offset = 0

        for ob in context.visible_objects:
            if ob.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}:
                continue
            if only_selectable and ob.hide_select:
                continue
            ob_eval = ob.evaluated_get(depsgraph)
            try:
                mesh = ob_eval.to_mesh()
            except (RuntimeError, AttributeError):
                continue
            if mesh is None:
                continue
            try:
                nverts = len(mesh.vertices)
                if nverts:
                    co = np.empty(nverts * 3, dtype=np.float64)
                    mesh.vertices.foreach_get("co", co)
                    co.shape = (nverts, 3)
                    mw = np.array(ob.matrix_world, dtype=np.float64)
                    co = co @ mw[:3, :3].T + mw[:3, 3]
                    vert_chunks.append(co)

                    nedges = len(mesh.edges)
                    if nedges:
                        ev = np.empty(nedges * 2, dtype=np.int32)
                        mesh.edges.foreach_get("vertices", ev)
                        ev.shape = (nedges, 2)
                        edge_chunks.append(ev + offset)
                    offset += nverts
            finally:
                ob_eval.to_mesh_clear()

            if offset >= limit:
                self.truncated = True
                break

        if vert_chunks:
            self.verts = np.concatenate(vert_chunks)
        if edge_chunks:
            self.edges = np.concatenate(edge_chunks)

    @staticmethod
    def project(coords, pm, width, height):
        """Project (N, 3) world coords to region pixels. Returns (x, y, valid)."""
        n = len(coords)
        hom = np.empty((n, 4), dtype=np.float64)
        hom[:, :3] = coords
        hom[:, 3] = 1.0
        clip = hom @ pm.T
        w = clip[:, 3]
        valid = w > 1e-6
        safe = np.where(valid, w, 1.0)
        x = (clip[:, 0] / safe * 0.5 + 0.5) * width
        y = (clip[:, 1] / safe * 0.5 + 0.5) * height
        return x, y, valid

    def edge_midpoints(self):
        if self.midpoints is None and self.edges is not None and self.verts is not None:
            self.midpoints = (self.verts[self.edges[:, 0]]
                              + self.verts[self.edges[:, 1]]) * 0.5
        return self.midpoints


_SNAP_CACHE = SnapCache()


def shared_snap_cache():
    """One snap cache for all the Max Line tools, rebuilt only after the scene changed."""
    return _SNAP_CACHE


@bpy.app.handlers.persistent
def _snap_cache_invalidate(*_args):
    _SNAP_CACHE.dirty = True


def ranked_indices(d2, count=8):
    """Indices of the 'count' smallest values, in ascending order."""
    n = len(d2)
    take = min(count, n)
    if take < n:
        order = np.argpartition(d2, take - 1)[:take]
    else:
        order = np.arange(n)
    return order[np.argsort(d2[order])]


def occluded(context, region, rv3d, mouse, co):
    """True if something solid sits between the viewer and 'co'."""
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    delta = co - origin
    dist = delta.length
    if dist < 1e-6:
        return False
    direction = delta / dist
    tolerance = max(1e-4, dist * 2e-3)
    depsgraph = context.evaluated_depsgraph_get()
    hit = context.scene.ray_cast(
        depsgraph, origin, direction, distance=dist - tolerance)[0]
    return bool(hit)


def nearest_point(coords, pm, region, mouse, radius, context, rv3d, use_occlusion):
    """Nearest projected point within 'radius' pixels, or None."""
    if coords is None or not len(coords):
        return None
    x, y, valid = SnapCache.project(coords, pm, region.width, region.height)
    d2 = (x - mouse[0]) ** 2 + (y - mouse[1]) ** 2
    d2[~valid] = np.inf
    r2 = float(radius) ** 2

    for idx in ranked_indices(d2):
        if not np.isfinite(d2[idx]) or d2[idx] > r2:
            return None
        co = Vector(coords[idx])
        if use_occlusion and occluded(context, region, rv3d, mouse, co):
            continue
        return co
    return None


def nearest_edge_point(cache, pm, region, mouse, radius, context, rv3d, use_occlusion):
    """Closest point along any visible edge, within 'radius' pixels."""
    if cache.verts is None or cache.edges is None or not len(cache.edges):
        return None
    x, y, valid = SnapCache.project(cache.verts, pm, region.width, region.height)
    p2d = np.stack((x, y), axis=1)

    e0 = cache.edges[:, 0]
    e1 = cache.edges[:, 1]
    good = valid[e0] & valid[e1]
    if not np.any(good):
        return None

    a = p2d[e0]
    ab = p2d[e1] - a
    denom = np.einsum('ij,ij->i', ab, ab)
    denom = np.where(denom > 1e-12, denom, 1.0)
    m = np.array(mouse, dtype=np.float64)
    t = np.clip(np.einsum('ij,ij->i', m - a, ab) / denom, 0.0, 1.0)
    delta = (a + t[:, None] * ab) - m
    d2 = np.einsum('ij,ij->i', delta, delta)
    d2[~good] = np.inf

    r2 = float(radius) ** 2
    for idx in ranked_indices(d2):
        if not np.isfinite(d2[idx]) or d2[idx] > r2:
            return None
        va = Vector(cache.verts[e0[idx]])
        vb = Vector(cache.verts[e1[idx]])
        co = va.lerp(vb, float(t[idx]))
        if use_occlusion and occluded(context, region, rv3d, mouse, co):
            continue
        return co
    return None


# ---------------------------------------------------------------------------
# Point placement
# ---------------------------------------------------------------------------

def plane_point(context, region, rv3d, mouse, mode):
    """Intersect the mouse ray with the chosen construction plane."""
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    cursor = context.scene.cursor.location

    if mode == 'GROUND':
        plane_co, plane_no = Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))
    elif mode == 'CURSOR_Z':
        plane_co, plane_no = cursor.copy(), Vector((0.0, 0.0, 1.0))
    else:
        plane_co, plane_no = cursor.copy(), rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))

    hit = intersect_line_plane(origin, origin + direction * 1e6, plane_co, plane_no)
    if hit is None or (hit - origin).dot(direction) <= 0.0:
        # Plane is edge-on or behind the viewer: fall back to a view-facing plane.
        plane_no = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
        hit = intersect_line_plane(origin, origin + direction * 1e6, cursor, plane_no)
    return hit if hit is not None else cursor.copy()


def view_plane_point(context, region, rv3d, mouse, through):
    """Point on the viewer-facing plane that passes through 'through'."""
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    normal = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
    hit = intersect_line_plane(origin, origin + direction * 1e6, through, normal)
    return hit if hit is not None else through.copy()


def ray_to_plane(context, region, rv3d, mouse, plane_co, plane_no):
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    return intersect_line_plane(origin, origin + direction * 1e6,
                                plane_co, plane_no)


def scene_ray(context, region, rv3d, mouse):
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    depsgraph = context.evaluated_depsgraph_get()
    return context.scene.ray_cast(depsgraph, origin, direction)


def object_under_mouse(context, region, rv3d, mouse):
    return hit_object_and_normal(context, region, rv3d, mouse)[0]


def hit_object_and_normal(context, region, rv3d, mouse):
    """Object and surface normal under the mouse, in one ray cast."""
    hit, _loc, normal, _idx, ob, _mat = scene_ray(context, region, rv3d, mouse)
    if not hit:
        return None, None
    obj = ob.original if ob is not None else None
    nor = normal.normalized() if normal.length_squared > 1e-12 else None
    return obj, nor


def surface_normal_under_mouse(context, region, rv3d, mouse):
    hit, _loc, normal, _idx, _ob, _mat = scene_ray(context, region, rv3d, mouse)
    if hit and normal.length_squared > 1e-12:
        return normal.normalized()
    return None


def raw_point(context, region, rv3d, mouse):
    """Surface under the mouse, else the fallback construction plane."""
    cfg = settings(context)
    hit, loc, normal = scene_ray(context, region, rv3d, mouse)[:3]
    if hit:
        if cfg.surface_offset > 0.0 and normal.length_squared > 0.0:
            return loc + normal.normalized() * cfg.surface_offset
        return loc.copy()
    return plane_point(context, region, rv3d, mouse, cfg.fallback_plane)


def query_snap(context, region, rv3d, cache, mouse, extra_points=()):
    """Snap using the scene's own snap element settings."""
    cfg = settings(context)
    elements = snap_elements(context)
    if not elements:
        return None
    if elements & {'VERTEX', 'EDGE_MIDPOINT', 'EDGE'}:
        cache.ensure(context)

    pm = np.array(rv3d.perspective_matrix, dtype=np.float64)
    radius = cfg.snap_radius
    shading = getattr(context.space_data, "shading", None)
    occl = cfg.use_occlusion and not getattr(shading, "show_xray", False)

    if 'VERTEX' in elements:
        if len(extra_points):
            own = np.array([tuple(p) for p in extra_points], dtype=np.float64)
            found = nearest_point(own, pm, region, mouse, radius, context,
                                  rv3d, False)
            if found is not None:
                return "Point", found
        found = nearest_point(cache.verts, pm, region, mouse, radius,
                              context, rv3d, occl)
        if found is not None:
            return "Vertex", found

    if 'EDGE_MIDPOINT' in elements:
        found = nearest_point(cache.edge_midpoints(), pm, region, mouse,
                              radius, context, rv3d, occl)
        if found is not None:
            return "Midpoint", found

    if 'EDGE' in elements:
        found = nearest_edge_point(cache, pm, region, mouse, radius,
                                   context, rv3d, occl)
        if found is not None:
            return "Edge", found

    if elements & {'FACE', 'FACE_PROJECT', 'FACE_NEAREST'}:
        hit, loc, normal = scene_ray(context, region, rv3d, mouse)[:3]
        if hit:
            if cfg.surface_offset > 0.0 and normal.length_squared > 0.0:
                loc = loc + normal.normalized() * cfg.surface_offset
            return "Face", loc.copy()

    if elements & {'INCREMENT', 'GRID'}:
        base = plane_point(context, region, rv3d, mouse, cfg.fallback_plane)
        step = cfg.grid_increment
        return "Grid", Vector((round(base.x / step) * step,
                               round(base.y / step) * step,
                               round(base.z / step) * step))
    return None


def find_view3d(context):
    """
    Locate a 3D viewport to work in. These tools can be launched from the
    Properties editor's Tool tab, which mirrors VIEW_3D sidebar panels but has
    no 3D region of its own, so fall back to searching the screen.
    Returns (area, region, region_3d) or (None, None, None).
    """
    def unpack(area):
        if area is None or area.type != 'VIEW_3D':
            return None
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        space = area.spaces.active
        rv3d = getattr(space, "region_3d", None)
        if region is None or rv3d is None:
            return None
        return area, region, rv3d

    current = getattr(context, "area", None)
    found = unpack(current)
    if found is not None:
        # Prefer the caller's own region data, which is correct in quad view.
        rv3d = getattr(context, "region_data", None)
        return found[0], found[1], rv3d if rv3d is not None else found[2]

    window = getattr(context, "window", None)
    screen = getattr(window, "screen", None) if window is not None else None
    if screen is None:
        screen = getattr(context, "screen", None)
    if screen is not None:
        for area in screen.areas:
            found = unpack(area)
            if found is not None:
                return found
    return None, None, None


# ---------------------------------------------------------------------------
# Line: the drawing is done in C (curve.maxline_draw, editcurve_maxline.cc), with the snapping
# cursor of the Add tools. It hands the finished spline to this operator, which makes the curve
# object with the Max Line settings.
# ---------------------------------------------------------------------------

class MAXLINE_PG_commit_point(bpy.types.PropertyGroup):
    co: bpy.props.FloatVectorProperty(name="Location", size=3, subtype='XYZ')
    object: bpy.props.StringProperty(name="Object", description="Object the point was placed on")
    normal: bpy.props.FloatVectorProperty(name="Normal", size=3, subtype='XYZ')
    has_normal: bpy.props.BoolProperty(name="Has Normal")


class CURVE_OT_maxline_commit(bpy.types.Operator):
    """Make a curve object from the points of a Max Line (used by the Max Line tool)"""

    bl_idname = "curve.maxline_commit"
    bl_label = "Max Line"
    bl_options = {'UNDO', 'INTERNAL'}

    points: bpy.props.CollectionProperty(type=MAXLINE_PG_commit_point, options={'SKIP_SAVE'})
    cyclic: bpy.props.BoolProperty(name="Closed", options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        if len(self.points) < 2:
            return {'CANCELLED'}
        points = []
        for item in self.points:
            point = SplinePoint(Vector(item.co))
            point.obj = bpy.data.objects.get(item.object) if item.object else None
            point.nor = Vector(item.normal) if item.has_normal else None
            points.append(point)
        anchor = anchor_object(points)
        knots = [Knot(p.co, p.hl, p.hr, p.htype, p.htype, True) for p in points]
        obj = commit_knots(context, knots, self.cyclic, anchor, "MaxLine", push_undo=False,
                           align_normal=anchor_normal(points, anchor))
        return {'FINISHED'} if obj is not None else {'CANCELLED'}


def anchor_object(points):
    """The object a spline belongs to: whichever one most of it was drawn on."""
    counts = {}
    order = []
    for p in points:
        if p.obj is None:
            continue
        key = p.obj.name
        if key not in counts:
            counts[key] = 0
            order.append(p.obj)
        counts[key] += 1
    if not order:
        return None
    # max() keeps the earliest entry on a tie, so ties go to the first drawn on.
    return max(order, key=lambda ob: counts[ob.name])


def anchor_normal(points, anchor):
    """Average surface normal of the points that landed on the anchor."""
    normals = [p.nor for p in points if p.nor is not None and (anchor is None or p.obj is anchor)]
    if not normals:
        return None
    total = Vector((0.0, 0.0, 0.0))
    for n in normals:
        total = total + n
    return total.normalized() if total.length_squared > 1e-12 else normals[0]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_polyline(coords, color, width):
    """Wide lines need the POLYLINE shader variants from Blender 4.5 onwards."""
    if len(coords) < 2:
        return
    pairs = []
    for i in range(len(coords) - 1):
        pairs.append(tuple(coords[i]))
        pairs.append(tuple(coords[i + 1]))
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.bind()
    viewport = gpu.state.viewport_get()
    shader.uniform_float("viewportSize", (viewport[2], viewport[3]))
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch_for_shader(shader, 'LINES', {"pos": pairs}).draw(shader)


def draw_quad_2d(x, y, half, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", color)
    verts = [(x - half, y - half), (x + half, y - half),
             (x + half, y + half), (x - half, y + half)]
    batch_for_shader(shader, 'TRIS', {"pos": verts},
                     indices=[(0, 1, 2), (0, 2, 3)]).draw(shader)


def draw_frame_2d(x, y, half, color, thickness=2.0):
    corners = [(x - half, y - half, 0.0), (x + half, y - half, 0.0),
               (x + half, y + half, 0.0), (x - half, y + half, 0.0),
               (x - half, y - half, 0.0)]
    draw_polyline(corners, color, thickness)


# ---------------------------------------------------------------------------
# Bezier math for the edit mode tools
# ---------------------------------------------------------------------------

LINEAR_TOL = 1e-4


def bezier_at(p, t):
    s = 1.0 - t
    return (p[0] * (s * s * s) + p[1] * (3 * s * s * t)
            + p[2] * (3 * s * t * t) + p[3] * (t * t * t))


def bezier_tangent(p, t):
    s = 1.0 - t
    return ((p[1] - p[0]) * (3 * s * s) + (p[2] - p[1]) * (6 * s * t)
            + (p[3] - p[2]) * (3 * t * t))


def bezier_split(p, t):
    """De Casteljau split. Returns the two halves, exactly preserving the shape."""
    a = p[0].lerp(p[1], t)
    b = p[1].lerp(p[2], t)
    c = p[2].lerp(p[3], t)
    d = a.lerp(b, t)
    e = b.lerp(c, t)
    f = d.lerp(e, t)
    return [p[0], a, d, f], [f, e, c, p[3]]


def bezier_slice(p, u0, u1):
    """The sub-curve between two parameters, as its own cubic."""
    if u1 <= u0:
        point = bezier_at(p, u0)
        return [point.copy(), point.copy(), point.copy(), point.copy()]
    if u0 > 0.0:
        p = bezier_split(p, u0)[1]
        u1 = (u1 - u0) / (1.0 - u0)
    if u1 < 1.0:
        p = bezier_split(p, u1)[0]
    return [v.copy() for v in p]


def bezier_length(p, samples=64):
    total = 0.0
    prev = p[0]
    for i in range(1, samples + 1):
        cur = bezier_at(p, i / samples)
        total += (cur - prev).length
        prev = cur
    return total


def bezier_param_at_length(p, target, samples=64):
    """Parameter where arc length measured from the start reaches 'target'."""
    if target <= 0.0:
        return 0.0
    total = 0.0
    prev = p[0]
    for i in range(1, samples + 1):
        t = i / samples
        cur = bezier_at(p, t)
        step = (cur - prev).length
        if total + step >= target:
            frac = (target - total) / step if step > 1e-12 else 0.0
            return min(1.0, (i - 1 + frac) / samples)
        total += step
        prev = cur
    return 1.0


def is_linear(p, tol=LINEAR_TOL):
    a = p[1] - p[0]
    b = p[3] - p[2]
    if a.length_squared < 1e-16 or b.length_squared < 1e-16:
        return True
    return 1.0 - a.normalized().dot(b.normalized()) < tol


# ---------------------------------------------------------------------------
# Knot container, mirroring one Blender bezier point
# ---------------------------------------------------------------------------

class Knot:
    __slots__ = ("co", "hl", "hr", "hlt", "hrt", "sel")

    def __init__(self, co, hl=None, hr=None, hlt='FREE', hrt='FREE', sel=False):
        self.co = co.copy()
        self.hl = co.copy() if hl is None else hl.copy()
        self.hr = co.copy() if hr is None else hr.copy()
        self.hlt = hlt
        self.hrt = hrt
        self.sel = sel

    def copy(self):
        return Knot(self.co, self.hl, self.hr, self.hlt, self.hrt, self.sel)

    def reversed(self):
        """Same point with the direction of travel flipped."""
        return Knot(self.co, self.hr, self.hl, self.hrt, self.hlt, self.sel)


def read_spline(spline):
    knots = [Knot(bp.co, bp.handle_left, bp.handle_right,
                  bp.handle_left_type, bp.handle_right_type,
                  bp.select_control_point)
             for bp in spline.bezier_points]
    return knots, spline.use_cyclic_u


def resolve_handles(knots, cyclic):
    """VECTOR handles are recomputed by Blender, so resolve them for the math."""
    n = len(knots)
    out = []
    for i, k in enumerate(knots):
        prev = knots[(i - 1) % n] if (cyclic or i > 0) else None
        nxt = knots[(i + 1) % n] if (cyclic or i < n - 1) else None
        hl, hr = k.hl, k.hr
        if k.hlt == 'VECTOR':
            hl = k.co + (prev.co - k.co) / 3.0 if prev else k.co.copy()
        if k.hrt == 'VECTOR':
            hr = k.co + (nxt.co - k.co) / 3.0 if nxt else k.co.copy()
        out.append((hl, hr))
    return out


def knot_segments(knots, cyclic):
    """(start index, end index, cubic control points) for every segment."""
    n = len(knots)
    if n < 2:
        return []
    handles = resolve_handles(knots, cyclic)
    segs = []
    for i in range(n if cyclic else n - 1):
        j = (i + 1) % n
        segs.append((i, j, [knots[i].co.copy(), handles[i][1].copy(),
                            handles[j][0].copy(), knots[j].co.copy()]))
    return segs


def tessellate_knots(knots, cyclic, resolution):
    return [list(interpolate_bezier(p[0], p[1], p[2], p[3], max(2, resolution)))
            for _i, _j, p in knot_segments(knots, cyclic)]


def write_spline(obj, knots, cyclic, resolution_u=12, select=False, replace=None):
    """Add a bezier spline built from 'knots'. Works in edit mode."""
    curve = obj.data
    spline = curve.splines.new('BEZIER')
    spline.use_cyclic_u = cyclic
    spline.resolution_u = resolution_u
    spline.bezier_points.add(len(knots) - 1)

    # Positions first with free handles, so nothing gets auto-adjusted.
    for bp, k in zip(spline.bezier_points, knots):
        bp.handle_left_type = 'FREE'
        bp.handle_right_type = 'FREE'
        bp.handle_left = k.hl
        bp.co = k.co
        bp.handle_right = k.hr
    for bp, k in zip(spline.bezier_points, knots):
        bp.handle_left_type = k.hlt
        bp.handle_right_type = k.hrt
        chosen = k.sel if select else False
        bp.select_control_point = chosen
        bp.select_left_handle = chosen
        bp.select_right_handle = chosen

    if replace is not None:
        curve.splines.remove(replace)
    return spline


def selected_bezier_splines(obj, require_selection=True):
    out = []
    for spline in obj.data.splines:
        if spline.type != 'BEZIER':
            continue
        if not require_selection or any(
                bp.select_control_point for bp in spline.bezier_points):
            out.append(spline)
    return out


# ---------------------------------------------------------------------------
# Fillet and chamfer
# ---------------------------------------------------------------------------

def arc_knots(p_a, t_a, p_b, t_b, segments_max_angle=math.pi / 2.0):
    """
    Circular arc from p_a to p_b, tangent to t_a and t_b at the ends.
    Returned as a list of (position, unit tangent) pairs plus handle lengths,
    split into sub-arcs so no piece exceeds 90 degrees.
    """
    chord = p_b - p_a
    chord_len = chord.length
    if chord_len < 1e-9:
        return None

    dot = max(-1.0, min(1.0, t_a.dot(t_b)))
    alpha = math.acos(dot)
    if alpha < 1e-5:
        return None

    normal = t_a.cross(t_b)
    if normal.length_squared < 1e-16:
        return None
    normal.normalize()

    radius = chord_len / (2.0 * math.sin(alpha / 2.0))
    center = p_a + normal.cross(t_a).normalized() * radius
    start_vec = p_a - center

    count = max(1, int(math.ceil(alpha / segments_max_angle - 1e-9)))
    beta = alpha / count
    handle = (4.0 / 3.0) * math.tan(beta / 4.0) * radius

    samples = []
    for j in range(count + 1):
        vec = Matrix.Rotation(beta * j, 4, normal) @ start_vec
        pos = center + vec
        tangent = normal.cross(vec).normalized()
        samples.append((pos, tangent))
    return samples, handle, radius, alpha


FILLET_COUNTERS = ('selected', 'endpoints', 'straight', 'tiny', 'filleted',
                   'clamped')


def fillet_knots(knots, cyclic, amount, chamfer=False, use_radius=True,
                 limit=0.999, samples=64):
    """
    Round or bevel every selected corner.
    Returns (new knot list or None, dict describing what was skipped and why).
    """
    info = {key: 0 for key in FILLET_COUNTERS}
    info['selected'] = sum(1 for k in knots if k.sel)
    info['zero_amount'] = amount <= 0.0
    n = len(knots)
    if n < 3 or amount <= 0.0:
        return None, info

    segs = knot_segments(knots, cyclic)
    seg_pts = {i: p for i, _j, p in segs}
    seg_len = {i: bezier_length(p, samples) for i, _j, p in segs}
    if not seg_pts:
        return None, info

    # Decide the trim distance at each corner, then convert it to parameters.
    desired = {}
    for i, k in enumerate(knots):
        prev_i = (i - 1) % n
        if not k.sel:
            continue
        if not cyclic and (i == 0 or i == n - 1):
            info['endpoints'] += 1
            continue
        if prev_i not in seg_pts or i not in seg_pts:
            info['endpoints'] += 1
            continue

        p_in, p_out = seg_pts[prev_i], seg_pts[i]
        t_in = bezier_tangent(p_in, 1.0)
        t_out = bezier_tangent(p_out, 0.0)
        if t_in.length_squared < 1e-16 or t_out.length_squared < 1e-16:
            info['straight'] += 1
            continue
        t_in.normalize()
        t_out.normalize()
        dot = max(-1.0, min(1.0, t_in.dot(t_out)))
        alpha = math.acos(dot)
        if alpha < 1e-5 or alpha > math.pi - 1e-5:
            info['straight'] += 1
            continue

        distance = amount * math.tan(alpha / 2.0) if use_radius else amount
        if distance <= 1e-9:
            info['tiny'] += 1
            continue
        desired[i] = distance

    if not desired:
        return None, info

    # Each segment is shared between the fillets at its two ends. Only split it
    # when both ends actually want room; a lone corner may use nearly all of it.
    factor = {i: 1.0 for i in desired}
    for i in seg_pts:
        j = (i + 1) % n
        total = desired.get(i, 0.0) + desired.get(j, 0.0)
        room = seg_len[i] * limit
        if total > room and total > 1e-12:
            share = room / total
            if i in factor:
                factor[i] = min(factor[i], share)
            if j in factor:
                factor[j] = min(factor[j], share)

    trim_start = {i: 0.0 for i in seg_pts}   # param where segment i begins
    trim_end = {i: 1.0 for i in seg_pts}     # param where segment i ends
    corners = {}
    for i, distance in desired.items():
        if factor[i] < 1.0 - 1e-9:
            info['clamped'] += 1
        distance *= factor[i]
        if distance <= 1e-9:
            info['tiny'] += 1
            continue
        prev_i = (i - 1) % n
        trim_end[prev_i] = bezier_param_at_length(
            seg_pts[prev_i], seg_len[prev_i] - distance, samples)
        trim_start[i] = bezier_param_at_length(seg_pts[i], distance, samples)
        corners[i] = True

    if not corners:
        return None, info

    sub = {i: bezier_slice(p, trim_start[i], trim_end[i])
           for i, p in seg_pts.items()}

    out = []
    for i, k in enumerate(knots):
        prev_i = (i - 1) % n
        incoming = sub.get(prev_i)
        outgoing = sub.get(i)

        if i not in corners:
            hl = incoming[2] if incoming is not None else k.hl
            hr = outgoing[1] if outgoing is not None else k.hr
            linear_in = incoming is None or is_linear(seg_pts[prev_i])
            linear_out = outgoing is None or is_linear(seg_pts[i])
            out.append(Knot(k.co, hl, hr,
                            'VECTOR' if linear_in and k.hlt == 'VECTOR' else 'FREE',
                            'VECTOR' if linear_out and k.hrt == 'VECTOR' else 'FREE',
                            k.sel))
            continue

        p_a = incoming[3]
        p_b = outgoing[0]
        t_a = bezier_tangent(seg_pts[prev_i], trim_end[prev_i])
        t_b = bezier_tangent(seg_pts[i], trim_start[i])
        t_a.normalize()
        t_b.normalize()

        linear_in = is_linear(seg_pts[prev_i])
        linear_out = is_linear(seg_pts[i])

        if chamfer:
            out.append(Knot(p_a, incoming[2], p_a + (p_b - p_a) / 3.0,
                            'VECTOR' if linear_in else 'FREE', 'VECTOR', True))
            out.append(Knot(p_b, p_b - (p_b - p_a) / 3.0, outgoing[1],
                            'VECTOR', 'VECTOR' if linear_out else 'FREE', True))
            continue

        arc = arc_knots(p_a, t_a, p_b, t_b)
        if arc is None:
            out.append(Knot(k.co, incoming[2], outgoing[1], k.hlt, k.hrt, k.sel))
            continue

        samples, handle, _radius, _alpha = arc
        last = len(samples) - 1
        for j, (pos, tangent) in enumerate(samples):
            hl = incoming[2] if j == 0 else pos - tangent * handle
            hr = outgoing[1] if j == last else pos + tangent * handle
            hlt = ('VECTOR' if (j == 0 and linear_in) else 'ALIGNED')
            hrt = ('VECTOR' if (j == last and linear_out) else 'ALIGNED')
            out.append(Knot(pos, hl, hr, hlt, hrt, True))

    info['filleted'] = len(corners)
    return out, info


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------

def refine_knots(knots, cyclic, seg_index, t):
    """Insert a point at parameter t of a segment without changing the shape."""
    segs = {i: p for i, _j, p in knot_segments(knots, cyclic)}
    if seg_index not in segs:
        return None
    t = max(1e-4, min(1.0 - 1e-4, t))
    left, right = bezier_split(segs[seg_index], t)
    straight = is_linear(segs[seg_index])

    out = []
    n = len(knots)
    j = (seg_index + 1) % n
    for i, k in enumerate(knots):
        new = k.copy()
        if i == seg_index:
            new.hr = left[1]
            if not straight:
                new.hrt = 'FREE'
        if i == j:
            new.hl = right[2]
            if not straight:
                new.hlt = 'FREE'
        out.append(new)

    htype = 'VECTOR' if straight else 'ALIGNED'
    inserted = Knot(left[3], left[2], right[1], htype, htype, True)
    out.insert(seg_index + 1, inserted)
    return out


def nearest_on_knots(knots, cyclic, matrix, region, rv3d, mouse, coarse=24,
                     refine_steps=20):
    """Closest point on the curve to the mouse, in screen space."""
    best = None
    for i, _j, p in knot_segments(knots, cyclic):
        world = [matrix @ v for v in p]

        def screen(t):
            return view3d_utils.location_3d_to_region_2d(
                region, rv3d, bezier_at(world, t))

        lo, hi = 0.0, 1.0
        best_t, best_d = 0.0, None
        for s in range(coarse + 1):
            t = s / coarse
            pos = screen(t)
            if pos is None:
                continue
            d = (pos.x - mouse[0]) ** 2 + (pos.y - mouse[1]) ** 2
            if best_d is None or d < best_d:
                best_d, best_t = d, t
        if best_d is None:
            continue
        # Golden-section style tightening around the best coarse sample.
        span = 1.0 / coarse
        lo, hi = max(0.0, best_t - span), min(1.0, best_t + span)
        for _ in range(refine_steps):
            mid = (lo + hi) * 0.5
            a, b = (lo + mid) * 0.5, (mid + hi) * 0.5
            pa, pb = screen(a), screen(b)
            if pa is None or pb is None:
                break
            da = (pa.x - mouse[0]) ** 2 + (pa.y - mouse[1]) ** 2
            db = (pb.x - mouse[0]) ** 2 + (pb.y - mouse[1]) ** 2
            if da < db:
                hi = mid
                if da < best_d:
                    best_d, best_t = da, a
            else:
                lo = mid
                if db < best_d:
                    best_d, best_t = db, b
        if best is None or best_d < best[0]:
            best = (best_d, i, best_t, matrix @ bezier_at(p, best_t))
    return best


# ---------------------------------------------------------------------------
# Outline / offset
# ---------------------------------------------------------------------------

def perp_xy(v):
    return Vector((v.y, -v.x, 0.0))


def offset_knots(knots, cyclic, distance, miter_limit=8.0):
    """Offset in the object's local XY plane. Exact for straight segments."""
    n = len(knots)
    if n < 2 or abs(distance) < 1e-12:
        return None
    handles = resolve_handles(knots, cyclic)

    tangents = []
    for i, k in enumerate(knots):
        prev = knots[(i - 1) % n] if (cyclic or i > 0) else None
        nxt = knots[(i + 1) % n] if (cyclic or i < n - 1) else None

        t_in = k.co - handles[i][0]
        if t_in.length_squared < 1e-16 and prev is not None:
            t_in = k.co - prev.co
        t_out = handles[i][1] - k.co
        if t_out.length_squared < 1e-16 and nxt is not None:
            t_out = nxt.co - k.co

        t_in = Vector((t_in.x, t_in.y, 0.0))
        t_out = Vector((t_out.x, t_out.y, 0.0))
        if prev is None:
            t_in = t_out.copy()
        if nxt is None:
            t_out = t_in.copy()
        if t_in.length_squared < 1e-16:
            t_in = Vector((1.0, 0.0, 0.0))
        if t_out.length_squared < 1e-16:
            t_out = t_in.copy()
        tangents.append((t_in.normalized(), t_out.normalized()))

    moved = []
    for i, k in enumerate(knots):
        n_in = perp_xy(tangents[i][0])
        n_out = perp_xy(tangents[i][1])
        miter = n_in + n_out
        if miter.length_squared < 1e-12:
            # Doubling back on itself; nothing sensible to miter to.
            miter = n_out.copy()
            scale = distance
        else:
            miter.normalize()
            denom = miter.dot(n_in)
            if abs(denom) < 1.0 / miter_limit:
                denom = math.copysign(1.0 / miter_limit, denom if denom else 1.0)
            scale = distance / denom
        moved.append(k.co + miter * scale)

    out = [Knot(moved[i], moved[i], moved[i], knots[i].hlt, knots[i].hrt, True)
           for i in range(n)]

    for i in range(n if cyclic else n - 1):
        j = (i + 1) % n
        old_chord = (knots[j].co - knots[i].co).length
        new_chord = (moved[j] - moved[i]).length
        s = (new_chord / old_chord) if old_chord > 1e-9 else 1.0
        out[i].hr = moved[i] + (handles[i][1] - knots[i].co) * s
        out[j].hl = moved[j] + (handles[j][0] - knots[j].co) * s

    if not cyclic:
        out[0].hl = moved[0] + (handles[0][0] - knots[0].co)
        out[n - 1].hr = moved[n - 1] + (handles[n - 1][1] - knots[n - 1].co)
    return out


def outline_result(knots, cyclic, distance, center, close_ends):
    """
    Returns (replacement_for_original_or_None, list_of_new_splines).
    Each new spline is a (knots, cyclic) pair.
    """
    if center:
        inner = offset_knots(knots, cyclic, -distance * 0.5)
        outer = offset_knots(knots, cyclic, distance * 0.5)
        if inner is None or outer is None:
            return None, []
        if cyclic or not close_ends:
            return (inner, cyclic), [(outer, cyclic)]
        ring = inner + [k.reversed() for k in reversed(outer)]
        return None, [(ring, True)]

    offset = offset_knots(knots, cyclic, distance)
    if offset is None:
        return None, []
    if cyclic or not close_ends:
        return None, [(offset, cyclic)]
    ring = [k.copy() for k in knots] + [k.reversed() for k in reversed(offset)]
    return None, [(ring, True)]


# ---------------------------------------------------------------------------
# Shared modal dragging
# ---------------------------------------------------------------------------

def world_per_pixel(region, rv3d, point):
    a = view3d_utils.location_3d_to_region_2d(region, rv3d, point)
    right = rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))
    b = view3d_utils.location_3d_to_region_2d(region, rv3d, point + right)
    if a is None or b is None:
        return 0.01
    d = (b - a).length
    return (1.0 / d) if d > 1e-6 else 0.01


def edit_curve_poll(cls, context):
    """Poll shared by the edit tools. Explains itself when it says no."""
    def refuse(message):
        setter = getattr(cls, "poll_message_set", None)
        if setter is not None:
            setter(message)
        return False

    ob = context.object
    if ob is None:
        return refuse("No active object")
    if ob.type != 'CURVE':
        return refuse("Active object is a %s, not a curve" % ob.type.title())
    if ob.mode != 'EDIT':
        return refuse("Only available in Edit Mode (press Tab)")
    if not any(s.type == 'BEZIER' for s in ob.data.splines):
        return refuse("This curve has no Bezier splines")
    return True


class MaxLinePreviewMixin:
    """Draws a GPU preview of the pending result and drags a scalar amount."""

    def _preview_setup(self, context, event):
        self.area, self.region, self.rv3d = find_view3d(context)
        if self.area is None:
            return False
        self.matrix = context.object.matrix_world.copy()
        self.preview = []
        # Left unset until the pointer reaches the viewport, so launching from
        # another editor does not make the value jump on the first move.
        self.start_x = None
        self.base_amount = 0.0
        self.scale = world_per_pixel(self.region, self.rv3d, self.matrix.translation)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_edit_preview, (self, context), 'WINDOW', 'POST_VIEW')
        PREVIEW_STATE['modal'] = True
        context.window.cursor_modal_set('MOVE_X')
        return True

    def _preview_teardown(self, context):
        PREVIEW_STATE['modal'] = False
        if getattr(self, "_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        for action in (lambda: context.window.cursor_modal_restore(),
                       lambda: context.workspace.status_text_set(None),
                       lambda: self.region.tag_redraw()):
            try:
                action()
            except (AttributeError, ReferenceError, TypeError):
                pass

    def _in_region(self, event):
        x = event.mouse_x - self.region.x
        y = event.mouse_y - self.region.y
        return 0 <= x < self.region.width and 0 <= y < self.region.height

    def _drag_amount(self, event):
        x = event.mouse_x - self.region.x
        if self.start_x is None:
            self.start_x = x
        travel = (x - self.start_x) * self.scale
        if event.shift:
            travel *= 0.1
        return self.base_amount + travel

    def modal(self, context, event):
        self.area.tag_redraw()

        if event.type in NAV_EVENTS:
            return {'PASS_THROUGH'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if self.start_x is None and not self._in_region(event):
                # Waiting for the pointer to arrive in the viewport.
                return {'RUNNING_MODAL'}
            self.amount = self._clamp_amount(self._drag_amount(event))
            self._rebuild_preview(context)
            context.workspace.status_text_set(
                "%s: %.4f   |   Shift: fine   |   LMB/Enter: confirm   |   Esc: cancel"
                % (self.bl_label, self.amount))
            return {'RUNNING_MODAL'}

        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER', 'SPACE'} \
                and event.value == 'PRESS':
            self._preview_teardown(context)
            return self.execute(context)

        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._preview_teardown(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self._preview_teardown(context)


def draw_edit_preview(op, context):
    if context.area != op.area:
        return
    if not getattr(op, "preview", None):
        return
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for polyline in op.preview:
        draw_polyline([op.matrix @ v for v in polyline], COL_PREVIEW, 3.0)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')


# ---------------------------------------------------------------------------
# Fillet / Chamfer
# ---------------------------------------------------------------------------

class _MaxLineFilletShared(MaxLinePreviewMixin):
    """Implementation shared by Fillet and Chamfer."""

    bl_options = {'REGISTER', 'UNDO'}

    amount: bpy.props.FloatProperty(
        name="Amount", description="Fillet radius, or trim distance in Distance mode",
        default=0.1, min=0.0, soft_max=10.0, subtype='DISTANCE',
    )
    measure: bpy.props.EnumProperty(
        name="Measure",
        description="How the amount is interpreted",
        items=MEASURE_ITEMS,
        default='RADIUS',
    )
    interactive: bpy.props.BoolProperty(
        name="Interactive",
        description="Drag in the viewport instead of applying the amount at once",
        default=True, options={'SKIP_SAVE', 'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return edit_curve_poll(cls, context)

    def _clamp_amount(self, value):
        return max(0.0, value)

    def _rebuild_preview(self, context):
        self.preview = []
        for knots, cyclic in self.source:
            result, _info = fillet_knots(knots, cyclic, self.amount,
                                         self.chamfer,
                                         self.measure == 'RADIUS')
            if result is not None:
                self.preview.extend(tessellate_knots(result, cyclic, 12))

    def invoke(self, context, event):
        obj = context.object
        splines = selected_bezier_splines(obj)
        if not splines:
            self.report({'WARNING'}, "Select some corner points first")
            return {'CANCELLED'}
        if not self.interactive:
            return self.execute(context)

        self.source = [read_spline(s) for s in splines]
        if not self._preview_setup(context, event):
            self.report({'WARNING'}, "%s needs a 3D viewport open" % self.bl_label)
            return {'CANCELLED'}
        # Start from something visible. Confirming without dragging then gives
        # a sensible result rather than a silent no-op.
        self.base_amount = self._suggest_amount()
        self.amount = self.base_amount
        self._rebuild_preview(context)
        context.workspace.status_text_set(
            "%s: %.4f   |   Drag to adjust   |   Shift: fine   |   "
            "LMB/Enter: confirm   |   Esc: cancel" % (self.bl_label, self.amount))
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _suggest_amount(self):
        """A tenth of the shortest segment next to any selected corner."""
        shortest = None
        for knots, cyclic in self.source:
            n = len(knots)
            lengths = {i: bezier_length(p, 16)
                       for i, _j, p in knot_segments(knots, cyclic)}
            for i, k in enumerate(knots):
                if not k.sel:
                    continue
                if not cyclic and (i == 0 or i == n - 1):
                    continue
                prev_i = (i - 1) % n
                if prev_i not in lengths or i not in lengths:
                    continue
                local = min(lengths[prev_i], lengths[i])
                shortest = local if shortest is None else min(shortest, local)
        return 0.0 if shortest is None else shortest * 0.1

    def execute(self, context):
        obj = context.object
        splines = selected_bezier_splines(obj)
        if not splines:
            self.report({'WARNING'}, "Select some corner points first")
            return {'CANCELLED'}

        totals = {key: 0 for key in FILLET_COUNTERS}
        totals['zero_amount'] = False
        for spline in list(splines):
            knots, cyclic = read_spline(spline)
            result, info = fillet_knots(knots, cyclic, self.amount,
                                        self.chamfer,
                                        self.measure == 'RADIUS')
            for key in FILLET_COUNTERS:
                totals[key] += info.get(key, 0)
            totals['zero_amount'] = totals['zero_amount'] or info['zero_amount']
            if result is None:
                continue
            write_spline(obj, result, cyclic, spline.resolution_u,
                         select=True, replace=spline)

        cfg = settings(context)
        cfg.fillet_measure = self.measure
        cfg.preview_mode = 'CHAMFER' if self.chamfer else 'FILLET'

        if totals['filleted']:
            # Value returns to zero so the next drag starts fresh instead of
            # compounding onto what was just applied.
            cfg.fillet_amount = 0.0
            remember_applied(obj, context)
            message = "%s %d corner%s" % (
                "Chamfered" if self.chamfer else "Filleted",
                totals['filleted'], "" if totals['filleted'] == 1 else "s")
            if totals['clamped']:
                message += " - %d limited by the adjacent segment length" % \
                    totals['clamped']
            self.report({'INFO'}, message)
            return {'FINISHED'}

        if totals['zero_amount']:
            self.report({'WARNING'},
                        "Amount is 0 - set it in the panel first, then click "
                        "again. It resets to 0 after each use")
        elif totals['selected'] == 0:
            self.report({'WARNING'}, "No control points are selected")
        elif totals['endpoints'] and not totals['straight']:
            self.report({'WARNING'},
                        "Only endpoints selected - an open spline's first and "
                        "last points have no corner to round")
        elif totals['straight']:
            self.report({'WARNING'},
                        "Selected points lie on a straight line, so there is "
                        "no corner to round")
        elif totals['tiny']:
            self.report({'WARNING'},
                        "Amount is too small for these segments")
        else:
            self.report({'WARNING'}, "Nothing to fillet")
        return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "amount")
        layout.prop(self, "measure")


class CURVE_OT_maxline_fillet(_MaxLineFilletShared, bpy.types.Operator):
    """Round the selected corner points with an arc"""

    bl_idname = "curve.maxline_fillet"
    bl_label = "Fillet"
    bl_options = {'REGISTER', 'UNDO'}

    chamfer: bpy.props.BoolProperty(
        name="Chamfer", description="Cut the corner straight instead of rounding it",
        default=False,
    )


class CURVE_OT_maxline_chamfer(_MaxLineFilletShared, bpy.types.Operator):
    """Cut the selected corner points straight"""

    bl_idname = "curve.maxline_chamfer"
    bl_label = "Chamfer"
    bl_options = {'REGISTER', 'UNDO'}

    chamfer: bpy.props.BoolProperty(
        name="Chamfer", description="Cut the corner straight instead of rounding it",
        default=True,
    )


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------

class CURVE_OT_maxline_refine(bpy.types.Operator):
    """Click on a segment to insert a point without changing the shape"""

    bl_idname = "curve.maxline_refine"
    bl_label = "Refine"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return edit_curve_poll(cls, context)

    def invoke(self, context, event):
        self.area, self.region, self.rv3d = find_view3d(context)
        if self.area is None:
            self.report({'WARNING'}, "Refine needs a 3D viewport open")
            return {'CANCELLED'}

        self.obj = context.object
        self.matrix = self.obj.matrix_world.copy()
        self.hit = None
        self.inserted = 0
        self._reload()

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_refine_preview, (self, context), 'WINDOW', 'POST_PIXEL')
        context.window.cursor_modal_set('CROSSHAIR')
        context.workspace.status_text_set(
            "Click a segment to insert a point   |   RMB/Enter: finish   |   Esc: cancel")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _reload(self):
        self.splines = [s for s in self.obj.data.splines if s.type == 'BEZIER']
        self.data = [read_spline(s) for s in self.splines]

    def _teardown(self, context):
        if getattr(self, "_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        for action in (lambda: context.window.cursor_modal_restore(),
                       lambda: context.workspace.status_text_set(None),
                       lambda: self.region.tag_redraw()):
            try:
                action()
            except (AttributeError, ReferenceError, TypeError):
                pass

    def _in_region(self, event):
        x = event.mouse_x - self.region.x
        y = event.mouse_y - self.region.y
        return 0 <= x < self.region.width and 0 <= y < self.region.height

    def _update_hit(self, event):
        if not self._in_region(event):
            self.hit = None
            return
        mouse = (event.mouse_x - self.region.x, event.mouse_y - self.region.y)
        best = None
        for index, (knots, cyclic) in enumerate(self.data):
            found = nearest_on_knots(knots, cyclic, self.matrix,
                                     self.region, self.rv3d, mouse)
            if found is None:
                continue
            if best is None or found[0] < best[0][0]:
                best = (found, index)
        threshold = 30.0 ** 2
        self.hit = best if (best is not None and best[0][0] <= threshold) else None

    def modal(self, context, event):
        self.area.tag_redraw()

        if event.type in NAV_EVENTS:
            return {'PASS_THROUGH'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._update_hit(event)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.hit is not None:
                (_dist, seg_index, t, _pos), spline_index = self.hit
                knots, cyclic = self.data[spline_index]
                result = refine_knots(knots, cyclic, seg_index, t)
                if result is not None:
                    old = self.splines[spline_index]
                    write_spline(self.obj, result, cyclic, old.resolution_u,
                                 select=True, replace=old)
                    self.inserted += 1
                    self._reload()
                    self.hit = None
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE', 'SPACE'} \
                and event.value == 'PRESS':
            self._teardown(context)
            return {'FINISHED'} if self.inserted else {'CANCELLED'}

        if event.type == 'ESC' and event.value == 'PRESS':
            self._teardown(context)
            return {'FINISHED'} if self.inserted else {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self._teardown(context)


def draw_refine_preview(op, context):
    if context.area != op.area or op.hit is None:
        return
    (_dist, _seg, _t, pos), _index = op.hit
    screen = view3d_utils.location_3d_to_region_2d(op.region, op.rv3d, pos)
    if screen is None:
        return
    scale = context.preferences.system.ui_scale
    gpu.state.blend_set('ALPHA')
    draw_quad_2d(screen.x, screen.y, 4.5 * scale, COL_KNOT_BEZ)
    draw_frame_2d(screen.x, screen.y, 8.0 * scale, COL_SNAP, 2.0)
    gpu.state.blend_set('NONE')


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------

class CURVE_OT_maxline_outline(MaxLinePreviewMixin, bpy.types.Operator):
    """Create a parallel copy of the selected splines"""

    bl_idname = "curve.maxline_outline"
    bl_label = "Outline"
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(
        name="Distance", description="Offset distance in the object's local XY plane",
        default=0.1, soft_min=-10.0, soft_max=10.0, subtype='DISTANCE',
    )
    center: bpy.props.BoolProperty(
        name="Center",
        description="Offset both ways by half, keeping the original centred",
        default=False,
    )
    close_ends: bpy.props.BoolProperty(
        name="Close Ends",
        description="On open splines, join the two ends into one closed shape",
        default=True,
    )
    interactive: bpy.props.BoolProperty(
        name="Interactive",
        description="Drag in the viewport instead of applying the distance at once",
        default=True, options={'SKIP_SAVE', 'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return edit_curve_poll(cls, context)

    def _clamp_amount(self, value):
        return value

    def _rebuild_preview(self, context):
        self.preview = []
        for knots, cyclic in self.source:
            replacement, extras = outline_result(
                knots, cyclic, self.distance, self.center, self.close_ends)
            if replacement is not None:
                self.preview.extend(tessellate_knots(
                    replacement[0], replacement[1], 12))
            for new_knots, new_cyclic in extras:
                self.preview.extend(tessellate_knots(new_knots, new_cyclic, 12))

    def _drag_amount(self, event):
        return MaxLinePreviewMixin._drag_amount(self, event)

    def invoke(self, context, event):
        obj = context.object
        splines = selected_bezier_splines(obj)
        if not splines:
            self.report({'WARNING'}, "Select a spline first")
            return {'CANCELLED'}
        if not self.interactive:
            return self.execute(context)

        self.source = [read_spline(s) for s in splines]
        if not self._preview_setup(context, event):
            self.report({'WARNING'}, "Outline needs a 3D viewport open")
            return {'CANCELLED'}
        self.base_amount = self.distance
        self.amount = self.distance
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        result = MaxLinePreviewMixin.modal(self, context, event)
        self.distance = self.amount
        return result

    def execute(self, context):
        obj = context.object
        splines = selected_bezier_splines(obj)
        if not splines:
            self.report({'WARNING'}, "Select a spline first")
            return {'CANCELLED'}

        created = 0
        for spline in list(splines):
            knots, cyclic = read_spline(spline)
            resolution = spline.resolution_u
            replacement, extras = outline_result(
                knots, cyclic, self.distance, self.center, self.close_ends)
            if replacement is None and not extras:
                continue
            for new_knots, new_cyclic in extras:
                write_spline(obj, new_knots, new_cyclic, resolution, select=True)
                created += 1
            if replacement is not None:
                write_spline(obj, replacement[0], replacement[1], resolution,
                             select=False, replace=spline)
            elif extras and not cyclic and self.close_ends and not self.center:
                obj.data.splines.remove(spline)

        cfg = settings(context)
        cfg.outline_center = self.center
        cfg.outline_close_ends = self.close_ends
        cfg.preview_mode = 'OUTLINE'

        if not created:
            if abs(self.distance) < 1e-9:
                self.report({'WARNING'},
                            "Distance is 0 - set it in the panel first, then "
                            "click again. It resets to 0 after each use")
            else:
                self.report({'WARNING'}, "Outline produced nothing")
            return {'CANCELLED'}

        cfg.outline_distance = 0.0
        remember_applied(obj, context)
        self.report({'INFO'}, "Outlined %d spline%s"
                    % (created, "" if created == 1 else "s"))
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "distance")
        layout.prop(self, "center")
        layout.prop(self, "close_ends")


# ---------------------------------------------------------------------------
# Shared object creation, used by every shape tool
# ---------------------------------------------------------------------------

def resolve_origin(context, knots, handles, anchor):
    """Where a newly created object's origin should sit."""
    mode = settings(context).origin_mode
    if mode == 'SURFACE' and anchor is not None:
        return anchor.matrix_world.translation.copy()
    if mode == 'CURSOR':
        return context.scene.cursor.location.copy()
    if mode == 'WORLD':
        return Vector((0.0, 0.0, 0.0))
    if mode == 'MEDIAN':
        total = Vector((0.0, 0.0, 0.0))
        for k in knots:
            total = total + k.co
        return total / len(knots)
    if mode == 'BBOX':
        coords = []
        for k, (hl, hr) in zip(knots, handles):
            coords.append(k.co)
            if k.hlt != 'VECTOR':
                coords.append(hl)
            if k.hrt != 'VECTOR':
                coords.append(hr)
        lo = Vector((min(c.x for c in coords), min(c.y for c in coords),
                     min(c.z for c in coords)))
        hi = Vector((max(c.x for c in coords), max(c.y for c in coords),
                     max(c.z for c in coords)))
        return (lo + hi) * 0.5
    return knots[0].co.copy()


def commit_knots(context, knots, cyclic, anchor, name="MaxLine", push_undo=True,
                 origin_override=None, align_normal=None):
    """Create (or extend) a curve object from world-space knots."""
    if len(knots) < 2:
        return None

    cfg = settings(context)
    handles = resolve_handles(knots, cyclic)

    target = None
    if cfg.attach_to_active:
        active = context.view_layer.objects.active
        if active is not None and active.type == 'CURVE':
            target = active

    if target is not None:
        curve = target.data
        matrix_inv = target.matrix_world.inverted()
    else:
        curve = bpy.data.curves.new(name, 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = cfg.curve_resolution
        target = bpy.data.objects.new(name, curve)
        context.collection.objects.link(target)

        origin = (Vector(origin_override) if origin_override is not None
                  else resolve_origin(context, knots, handles, anchor))
        basis = Matrix.Identity(4)
        if cfg.align_mode == 'OBJECT' and anchor is not None:
            basis = anchor.matrix_world.to_3x3().to_4x4()
        elif cfg.align_mode == 'NORMAL' and align_normal is not None:
            basis = basis_from_normal(Vector(align_normal))
        matrix_world = Matrix.Translation(origin) @ basis

        if cfg.parent_to_surface and anchor is not None:
            target.parent = anchor
            target.matrix_parent_inverse = anchor.matrix_world.inverted()
        # Assign matrix_world rather than location: setting location alone
        # leaves matrix_world stale until the depsgraph re-evaluates.
        target.matrix_world = matrix_world
        matrix_inv = matrix_world.inverted()

    local = [Knot(matrix_inv @ k.co, matrix_inv @ hl, matrix_inv @ hr,
                  k.hlt, k.hrt, True)
             for k, (hl, hr) in zip(knots, handles)]
    write_spline(target, local, cyclic, cfg.curve_resolution, select=True)

    for ob in context.selected_objects:
        ob.select_set(False)
    target.select_set(True)
    context.view_layer.objects.active = target

    if push_undo:
        try:
            bpy.ops.ed.undo_push(message=name)
        except RuntimeError:
            pass
    return target


# ---------------------------------------------------------------------------
# Shape geometry
# ---------------------------------------------------------------------------

def plane_basis(normal):
    """Two in-plane axes for a normal, keeping world X/Y when the plane is flat."""
    n = normal.normalized() if normal.length_squared > 1e-12 else Vector((0.0, 0.0, 1.0))
    x = Vector((0.0, 0.0, 1.0)).cross(n)
    if x.length_squared < 1e-12:
        x = Vector((1.0, 0.0, 0.0))
    x.normalize()
    y = n.cross(x).normalized()
    return x, y, n


def basis_from_normal(normal):
    """Orthonormal rotation whose Z is the given normal."""
    ax, ay, az = plane_basis(normal)
    return Matrix((ax, ay, az)).transposed().to_4x4()


def project_to_plane(point, origin, normal):
    return point - normal * (point - origin).dot(normal)


def circle_knots(center, ax, ay, radius, segments=4):
    """Bezier circle. Handle length is the standard arc approximation."""
    if radius <= 1e-9:
        return None
    step = 2.0 * math.pi / segments
    handle = (4.0 / 3.0) * math.tan(step / 4.0) * radius
    out = []
    for i in range(segments):
        angle = i * step
        radial = ax * math.cos(angle) + ay * math.sin(angle)
        tangent = ay * math.cos(angle) - ax * math.sin(angle)
        co = center + radial * radius
        out.append(Knot(co, co - tangent * handle, co + tangent * handle,
                        'ALIGNED', 'ALIGNED', True))
    return out


def round_corners(knots, corner_radius):
    if corner_radius <= 1e-9:
        return knots
    result, _info = fillet_knots(knots, True, corner_radius, False, True)
    return result if result is not None else knots


def rectangle_knots(center, ax, ay, half_w, half_h, corner_radius=0.0):
    if half_w <= 1e-9 or half_h <= 1e-9:
        return None
    corners = [center - ax * half_w - ay * half_h,
               center + ax * half_w - ay * half_h,
               center + ax * half_w + ay * half_h,
               center - ax * half_w + ay * half_h]
    knots = [Knot(c, c, c, 'VECTOR', 'VECTOR', True) for c in corners]
    return round_corners(knots, corner_radius)


def ngon_knots(center, ax, ay, radius, sides, inscribed=True, corner_radius=0.0):
    if radius <= 1e-9 or sides < 3:
        return None
    # Inscribed puts the corners on the radius; circumscribed puts the edges there.
    outer = radius if inscribed else radius / math.cos(math.pi / sides)
    knots = []
    for i in range(sides):
        angle = math.pi * 0.5 + i * 2.0 * math.pi / sides
        co = center + (ax * math.cos(angle) + ay * math.sin(angle)) * outer
        knots.append(Knot(co, co, co, 'VECTOR', 'VECTOR', True))
    return round_corners(knots, corner_radius)


# ---------------------------------------------------------------------------
# Shape tools
# ---------------------------------------------------------------------------

class _MaxLineShapeShared:
    """Drag-out modal shared by Circle, Rectangle and NGon."""

    bl_options = {'REGISTER', 'UNDO'}

    center: bpy.props.FloatVectorProperty(
        name="Center", size=3, subtype='XYZ', options={'HIDDEN', 'SKIP_SAVE'})
    axis_x: bpy.props.FloatVectorProperty(
        name="Axis X", size=3, default=(1.0, 0.0, 0.0),
        options={'HIDDEN', 'SKIP_SAVE'})
    axis_y: bpy.props.FloatVectorProperty(
        name="Axis Y", size=3, default=(0.0, 1.0, 0.0),
        options={'HIDDEN', 'SKIP_SAVE'})
    anchor_name: bpy.props.StringProperty(
        name="Anchor", options={'HIDDEN', 'SKIP_SAVE'})
    corner_radius: bpy.props.FloatProperty(
        name="Corner Radius", description="Round the corners by this radius",
        default=0.0, min=0.0, soft_max=100.0, subtype='DISTANCE')

    @classmethod
    def poll(cls, context):
        if context.mode != 'OBJECT':
            setter = getattr(cls, "poll_message_set", None)
            if setter is not None:
                setter("Only available in Object Mode")
            return False
        return True

    # -- geometry, provided by each shape ---------------------------------

    def build_knots(self):
        raise NotImplementedError

    def set_size(self, delta_x, delta_y, uniform):
        raise NotImplementedError

    def size_label(self):
        return ""

    def has_size(self):
        raise NotImplementedError

    def adjust_count(self, step):
        """Arrow keys during the drag. Only the NGon has anything to change."""
        return False

    def store_defaults(self, cfg):
        pass

    def has_extra_stage(self):
        """True for shapes needing a third click, such as an arc's bulge."""
        return False

    def set_extra(self, delta_x, delta_y):
        pass

    def is_cyclic(self):
        return True

    def commit_origin(self):
        return Vector(self.center)

    # -- lifecycle ---------------------------------------------------------

    def invoke(self, context, event):
        self.area, self.region, self.rv3d = find_view3d(context)
        if self.area is None:
            self.report({'WARNING'}, "%s needs a 3D viewport open" % self.bl_label)
            return {'CANCELLED'}

        self.stage = 'START'
        self.origin_point = Vector()
        self.normal = Vector((0.0, 0.0, 1.0))
        self.anchor = None
        self.cursor_co = Vector()
        self.snap_label = ""
        self.snap_co = None
        self.press_mouse = (0, 0)
        self.preview = []
        self.mouse = self._mouse(event)

        self.cache = shared_snap_cache()
        self._resolve(context, event)

        self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(
            draw_shape_preview_3d, (self,), 'WINDOW', 'POST_VIEW')
        self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(
            draw_shape_preview_2d, (self,), 'WINDOW', 'POST_PIXEL')
        context.window.cursor_modal_set('CROSSHAIR')
        self._status(context)
        context.window_manager.modal_handler_add(self)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _status(self, context):
        context.workspace.status_text_set(
            "Click and drag to place the %s   |   Ctrl: snap   |   "
            "Shift: uniform   |   %s   |   Enter: confirm   |   Esc: cancel"
            % (self.bl_label.lower(), self.size_label()))

    def _teardown(self, context):
        for name in ("_handle_3d", "_handle_2d"):
            handle = getattr(self, name, None)
            if handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                setattr(self, name, None)
        for action in (lambda: context.window.cursor_modal_restore(),
                       lambda: context.workspace.status_text_set(None),
                       lambda: self.area.tag_redraw()):
            try:
                action()
            except (AttributeError, ReferenceError, TypeError):
                pass

    def _mouse(self, event):
        return (event.mouse_x - self.region.x, event.mouse_y - self.region.y)

    def _in_region(self, event):
        x, y = self._mouse(event)
        return 0 <= x < self.region.width and 0 <= y < self.region.height

    # -- point resolution --------------------------------------------------

    def _resolve(self, context, event):
        """Free cursor position before the first click."""
        mouse = self._mouse(event)
        self.mouse = mouse
        self.snap_label = ""
        self.snap_co = None
        if snapping_active(context, event):
            snapped = query_snap(context, self.region, self.rv3d, self.cache, mouse)
            if snapped is not None:
                self.snap_label, self.cursor_co = snapped
                self.snap_co = self.cursor_co.copy()
                return
        self.cursor_co = raw_point(context, self.region, self.rv3d, mouse)

    def _resolve_on_plane(self, context, event):
        """Second point, constrained to the shape's own plane."""
        mouse = self._mouse(event)
        self.mouse = mouse
        self.snap_label = ""
        self.snap_co = None
        if snapping_active(context, event):
            snapped = query_snap(context, self.region, self.rv3d, self.cache, mouse)
            if snapped is not None:
                self.snap_label = snapped[0]
                self.snap_co = snapped[1].copy()
                return project_to_plane(snapped[1], self.origin_point, self.normal)
        hit = ray_to_plane(context, self.region, self.rv3d, mouse,
                           self.origin_point, self.normal)
        return hit if hit is not None else self.origin_point.copy()

    def _begin(self, context, event):
        self.origin_point = self.cursor_co.copy()
        self.anchor = object_under_mouse(context, self.region, self.rv3d,
                                         self._mouse(event))
        normal = surface_normal_under_mouse(context, self.region, self.rv3d,
                                            self._mouse(event))
        if normal is None:
            cfg = settings(context)
            if cfg.fallback_plane == 'VIEW':
                normal = self.rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
            else:
                normal = Vector((0.0, 0.0, 1.0))
        ax, ay, n = plane_basis(normal)
        self.normal = n
        self.axis_x = ax
        self.axis_y = ay
        self.center = self.origin_point
        self.press_mouse = self._mouse(event)
        self.stage = 'SIZE'

    def _update_size(self, context, event):
        point = self._resolve_on_plane(context, event)
        ax = Vector(self.axis_x)
        ay = Vector(self.axis_y)
        delta = point - self.origin_point
        self.set_size(delta.dot(ax), delta.dot(ay), event.shift)
        self._rebuild()

    def _rebuild(self):
        knots = self.build_knots()
        self.preview = (tessellate_knots(knots, self.is_cyclic(), 12)
                        if knots else [])

    def _update_extra(self, context, event):
        point = self._resolve_on_plane(context, event)
        delta = point - self.origin_point
        self.set_extra(delta.dot(Vector(self.axis_x)),
                       delta.dot(Vector(self.axis_y)))
        self._rebuild()

    def _advance(self, context):
        """Move on to the extra stage, or commit when there isn't one."""
        if self.stage == 'SIZE' and self.has_extra_stage() and self.has_size():
            self.stage = 'EXTRA'
            self._last_angle = None
            self.press_mouse = self.mouse
            return {'RUNNING_MODAL'}
        return self._confirm(context)

    # -- modal -------------------------------------------------------------

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self._teardown(context)
            return {'CANCELLED'}

        self.area.tag_redraw()
        if event.type in NAV_EVENTS:
            return {'PASS_THROUGH'}

        inside = self._in_region(event)

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if not inside:
                return {'RUNNING_MODAL'}
            if self.stage == 'START':
                self._resolve(context, event)
            elif self.stage == 'SIZE':
                self._update_size(context, event)
                self._status(context)
            else:
                self._update_extra(context, event)
                self._status(context)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if not inside:
                return {'RUNNING_MODAL'}
            if event.value == 'PRESS':
                if self.stage == 'START':
                    self._begin(context, event)
                elif self.stage == 'SIZE':
                    return self._advance(context)
                else:
                    return self._confirm(context)
            elif event.value == 'RELEASE' and self.stage == 'SIZE':
                travel = math.hypot(self.mouse[0] - self.press_mouse[0],
                                    self.mouse[1] - self.press_mouse[1])
                # A release without travel means they clicked rather than
                # dragged, so stay live and wait for a second click.
                if travel >= 4 and self.has_size():
                    return self._advance(context)
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS' and event.type in {
                'UP_ARROW', 'DOWN_ARROW', 'PAGE_UP', 'PAGE_DOWN'}:
            step = 1 if event.type in {'UP_ARROW', 'PAGE_UP'} else -1
            if self.adjust_count(step):
                self._rebuild()
                self._status(context)
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if self.stage in {'SIZE', 'EXTRA'} and self.has_size():
                return self._confirm(context)
            return {'RUNNING_MODAL'}

        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            self._teardown(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _confirm(self, context):
        if not self.has_size():
            self._teardown(context)
            return {'CANCELLED'}
        self.anchor_name = self.anchor.name if self.anchor is not None else ""
        self._teardown(context)
        return self.execute(context)

    def execute(self, context):
        knots = self.build_knots()
        if not knots:
            self.report({'WARNING'}, "%s has no size" % self.bl_label)
            return {'CANCELLED'}
        anchor = bpy.data.objects.get(self.anchor_name) if self.anchor_name else None
        # Shapes always sit on their own centre, whatever Origin says.
        # The shape's own plane normal, recoverable after a redo.
        normal = Vector(self.axis_x).cross(Vector(self.axis_y))
        obj = commit_knots(context, knots, self.is_cyclic(), anchor,
                           self.bl_label, push_undo=False,
                           origin_override=self.commit_origin(),
                           align_normal=normal)
        if obj is None:
            return {'CANCELLED'}
        self.store_defaults(settings(context))
        return {'FINISHED'}

    def cancel(self, context):
        self._teardown(context)


class CURVE_OT_maxline_circle(_MaxLineShapeShared, bpy.types.Operator):
    """Drag out a circle from its centre"""

    bl_idname = "curve.maxline_circle"
    bl_label = "Circle"
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(
        name="Radius", default=1.0, min=0.0, soft_max=100.0, subtype='DISTANCE')

    def set_size(self, dx, dy, uniform):
        self.radius = math.hypot(dx, dy)

    def has_size(self):
        return self.radius > 1e-9

    def size_label(self):
        return "Radius %.4f" % self.radius

    def build_knots(self):
        return circle_knots(Vector(self.center), Vector(self.axis_x),
                            Vector(self.axis_y), self.radius)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "radius")


class CURVE_OT_maxline_rectangle(_MaxLineShapeShared, bpy.types.Operator):
    """Drag out a rectangle"""

    bl_idname = "curve.maxline_rectangle"
    bl_label = "Rectangle"
    bl_options = {'REGISTER', 'UNDO'}

    width: bpy.props.FloatProperty(
        name="Width", default=2.0, min=0.0, soft_max=100.0, subtype='DISTANCE')
    height: bpy.props.FloatProperty(
        name="Height", default=2.0, min=0.0, soft_max=100.0, subtype='DISTANCE')
    from_center: bpy.props.BoolProperty(
        name="From Center",
        description="Drag out from the middle instead of from a corner",
        default=False)

    def set_size(self, dx, dy, uniform):
        if uniform:
            size = max(abs(dx), abs(dy))
            dx = math.copysign(size, dx or 1.0)
            dy = math.copysign(size, dy or 1.0)
        if self.from_center:
            self.width = abs(dx) * 2.0
            self.height = abs(dy) * 2.0
            self.center = self.origin_point
        else:
            self.width = abs(dx)
            self.height = abs(dy)
            # Store the true centre: a plain attribute would be lost on redo.
            self.center = (self.origin_point
                           + Vector(self.axis_x) * (dx * 0.5)
                           + Vector(self.axis_y) * (dy * 0.5))

    def has_size(self):
        return self.width > 1e-9 and self.height > 1e-9

    def size_label(self):
        return "%.4f x %.4f" % (self.width, self.height)

    def build_knots(self):
        return rectangle_knots(Vector(self.center), Vector(self.axis_x),
                               Vector(self.axis_y), self.width * 0.5,
                               self.height * 0.5, self.corner_radius)

    def store_defaults(self, cfg):
        cfg.shape_corner_radius = self.corner_radius

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "width")
        layout.prop(self, "height")
        layout.prop(self, "corner_radius")


class CURVE_OT_maxline_ngon(_MaxLineShapeShared, bpy.types.Operator):
    """Drag out a regular polygon from its centre"""

    bl_idname = "curve.maxline_ngon"
    bl_label = "NGon"
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(
        name="Radius", default=1.0, min=0.0, soft_max=100.0, subtype='DISTANCE')
    sides: bpy.props.IntProperty(
        name="Sides", default=6, min=3, max=64)
    inscribed: bpy.props.BoolProperty(
        name="Inscribed",
        description="Corners sit on the radius. Off puts the edges there "
                    "instead, as 3ds Max's Circumscribed does",
        default=True)

    def set_size(self, dx, dy, uniform):
        self.radius = math.hypot(dx, dy)

    def has_size(self):
        return self.radius > 1e-9

    def size_label(self):
        return "Radius %.4f, %d sides (Up/Down)" % (self.radius, self.sides)

    def adjust_count(self, step):
        new = max(3, min(64, self.sides + step))
        if new == self.sides:
            return False
        self.sides = new
        return True

    def store_defaults(self, cfg):
        cfg.ngon_sides = self.sides
        cfg.ngon_inscribed = self.inscribed
        cfg.shape_corner_radius = self.corner_radius

    def build_knots(self):
        return ngon_knots(Vector(self.center), Vector(self.axis_x),
                          Vector(self.axis_y), self.radius, self.sides,
                          self.inscribed, self.corner_radius)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "radius")
        layout.prop(self, "sides")
        layout.prop(self, "corner_radius")
        layout.prop(self, "inscribed")


def circle_from_three(p1, p2, p3):
    """Circumcircle of three 2D points, or None when they are collinear."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    sa = ax * ax + ay * ay
    sb = bx * bx + by * by
    sc = cx * cx + cy * cy
    ux = (sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)) / d
    uy = (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def arc_span_knots(centre, ax, ay, radius, start, sweep, pie=False):
    """
    Bezier arc of 'sweep' radians starting at 'start'. Sweep may be negative.
    Split so no piece exceeds 90 degrees, which keeps the approximation tight.
    """
    if radius <= 1e-9 or abs(sweep) < 1e-6:
        return None
    count = max(1, int(math.ceil(abs(sweep) / (math.pi * 0.5) - 1e-9)))
    step = sweep / count
    handle = (4.0 / 3.0) * math.tan(step / 4.0) * radius

    knots = []
    for i in range(count + 1):
        angle = start + step * i
        radial = ax * math.cos(angle) + ay * math.sin(angle)
        tangent = ay * math.cos(angle) - ax * math.sin(angle)
        co = centre + radial * radius
        knots.append(Knot(co, co - tangent * handle, co + tangent * handle,
                          'ALIGNED', 'ALIGNED', True))

    if pie:
        knots[0].hl = knots[0].co.copy()
        knots[0].hlt = 'VECTOR'
        knots[-1].hr = knots[-1].co.copy()
        knots[-1].hrt = 'VECTOR'
        knots.append(Knot(centre, centre, centre, 'VECTOR', 'VECTOR', True))
    else:
        knots[0].hl = knots[0].co * 2.0 - knots[0].hr
        knots[-1].hr = knots[-1].co * 2.0 - knots[-1].hl
    return knots


class CURVE_OT_maxline_arc(_MaxLineShapeShared, bpy.types.Operator):
    """Drag out an arc"""

    bl_idname = "curve.maxline_arc"
    bl_label = "Arc"
    bl_options = {'REGISTER', 'UNDO'}

    method: bpy.props.EnumProperty(
        name="Method",
        description="How the arc is dragged out",
        items=[
            ('ENDS', "End-End-Middle",
             "Drag the two ends, then move to set the bulge"),
            ('CENTER', "Center-End-End",
             "Click the centre, drag the radius, then sweep the angle"),
        ],
        default='ENDS')
    pie_slice: bpy.props.BoolProperty(
        name="Pie Slice",
        description="Close the arc back through its centre",
        default=False)

    # End-End-Middle, in plane coordinates relative to the first click.
    end_x: bpy.props.FloatProperty(options={'HIDDEN'})
    end_y: bpy.props.FloatProperty(options={'HIDDEN'})
    bulge_x: bpy.props.FloatProperty(options={'HIDDEN'})
    bulge_y: bpy.props.FloatProperty(options={'HIDDEN'})
    # Center-End-End.
    radius: bpy.props.FloatProperty(
        name="Radius", default=1.0, min=0.0, soft_max=100.0, subtype='DISTANCE')
    start_angle: bpy.props.FloatProperty(
        name="From", default=0.0, subtype='ANGLE')
    sweep: bpy.props.FloatProperty(
        name="Sweep", default=math.pi, subtype='ANGLE')

    # -- staged drag -------------------------------------------------------

    def has_extra_stage(self):
        return True

    def is_cyclic(self):
        return self.pie_slice

    def set_size(self, dx, dy, uniform):
        if self.method == 'ENDS':
            self.end_x, self.end_y = dx, dy
            # A quarter-chord bulge, so the preview is already an arc and
            # confirming here without sweeping still gives something sensible.
            length = math.hypot(dx, dy)
            self.bulge_x = dx * 0.5 - dy * 0.25
            self.bulge_y = dy * 0.5 + dx * 0.25
            if length < 1e-9:
                self.bulge_x = self.bulge_y = 0.0
        else:
            self.radius = math.hypot(dx, dy)
            self.start_angle = math.atan2(dy, dx) if self.radius > 1e-9 else 0.0
            self.sweep = math.pi * 2.0

    @staticmethod
    def _wrap(angle):
        while angle > math.pi:
            angle -= math.pi * 2.0
        while angle < -math.pi:
            angle += math.pi * 2.0
        return angle

    def set_extra(self, dx, dy):
        if self.method == 'ENDS':
            self.bulge_x, self.bulge_y = dx, dy
            return
        if dx * dx + dy * dy < 1e-12:
            return
        angle = math.atan2(dy, dx)
        last = getattr(self, "_last_angle", None)
        if last is None:
            self.sweep = self._wrap(angle - self.start_angle)
        else:
            # Accumulate so the sweep can pass 180 degrees and keep going.
            self.sweep += self._wrap(angle - last)
        self._last_angle = angle

    def has_size(self):
        if self.method == 'ENDS':
            return math.hypot(self.end_x, self.end_y) > 1e-9
        return self.radius > 1e-9 and abs(self.sweep) > 1e-6

    def size_label(self):
        if self.method == 'ENDS':
            return "Chord %.4f" % math.hypot(self.end_x, self.end_y)
        return "Radius %.4f, sweep %.1f deg" % (self.radius,
                                                math.degrees(self.sweep))

    # -- geometry ----------------------------------------------------------

    def _solve(self):
        """(centre in plane coords, radius, start angle, sweep) or None."""
        if self.method == 'CENTER':
            return (0.0, 0.0), self.radius, self.start_angle, self.sweep
        found = circle_from_three((0.0, 0.0), (self.end_x, self.end_y),
                                  (self.bulge_x, self.bulge_y))
        if found is None:
            return None
        (ux, uy), radius = found
        two_pi = math.pi * 2.0
        a1 = math.atan2(-uy, -ux)
        a2 = math.atan2(self.end_y - uy, self.end_x - ux)
        a3 = math.atan2(self.bulge_y - uy, self.bulge_x - ux)
        ccw = (a2 - a1) % two_pi
        via = (a3 - a1) % two_pi
        sweep = ccw if via <= ccw else ccw - two_pi
        return (ux, uy), radius, a1, sweep

    def commit_origin(self):
        ax, ay = Vector(self.axis_x), Vector(self.axis_y)
        solved = self._solve()
        if solved is None:
            return Vector(self.center)
        (ux, uy), _r, _s, _w = solved
        return Vector(self.center) + ax * ux + ay * uy

    def build_knots(self):
        ax, ay = Vector(self.axis_x), Vector(self.axis_y)
        anchor = Vector(self.center)
        solved = self._solve()
        if solved is None:
            # Collinear: a straight chord is the honest answer.
            end = anchor + ax * self.end_x + ay * self.end_y
            if (end - anchor).length_squared < 1e-18:
                return None
            return [Knot(anchor, anchor, anchor, 'VECTOR', 'VECTOR', True),
                    Knot(end, end, end, 'VECTOR', 'VECTOR', True)]
        (ux, uy), radius, start, sweep = solved
        centre = anchor + ax * ux + ay * uy
        return arc_span_knots(centre, ax, ay, radius, start, sweep,
                              self.pie_slice)

    def store_defaults(self, cfg):
        cfg.arc_method = self.method
        cfg.arc_pie = self.pie_slice

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        if self.method == 'CENTER':
            layout.prop(self, "radius")
            layout.prop(self, "start_angle")
            layout.prop(self, "sweep")
        layout.prop(self, "pie_slice")


def draw_shape_preview_3d(op):
    context = bpy.context
    if context.area != op.area or not op.preview:
        return
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for polyline in op.preview:
        draw_polyline(polyline, COL_CURVE, 3.0)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')


def draw_shape_preview_2d(op):
    context = bpy.context
    if context.area != op.area:
        return
    scale = context.preferences.system.ui_scale
    gpu.state.blend_set('ALPHA')
    if op.stage in {'SIZE', 'EXTRA'}:
        pos = view3d_utils.location_3d_to_region_2d(
            op.region, op.rv3d, op.origin_point)
        if pos is not None:
            draw_quad_2d(pos.x, pos.y, 3.5 * scale, COL_FIRST)
    if op.snap_co is not None:
        pos = view3d_utils.location_3d_to_region_2d(op.region, op.rv3d, op.snap_co)
        if pos is not None:
            draw_frame_2d(pos.x, pos.y, 7.0 * scale, COL_SNAP, 2.0)
            if op.snap_label:
                blf.size(0, int(11 * scale))
                blf.color(0, *COL_SNAP)
                blf.position(0, pos.x + 12 * scale, pos.y + 12 * scale, 0.0)
                blf.draw(0, op.snap_label)
    gpu.state.blend_set('NONE')


# ---------------------------------------------------------------------------
# Freehand stroke fitting
#
# Schneider's algorithm (Graphics Gems, 1990): fit a cubic by least squares,
# refine the parameterisation with Newton-Raphson, and split at the worst
# point when the error is still too large.
# ---------------------------------------------------------------------------

FIT_MAX_DEPTH = 14
FIT_NEWTON_PASSES = 4
LIVE_FIT_MAX_POINTS = 120     # preview fits at most this many, by striding
LIVE_FIT_EVERY = 10           # samples between live refits

FREEHAND_STROKE = {'points': [], 'anchor': "", 'normal': None,
                   'scale': 0.01, 'closed': False}


def dedupe_points(points, eps=1e-9):
    out = [points[0]]
    for p in points[1:]:
        if (p - out[-1]).length_squared > eps:
            out.append(p)
    return out


def smooth_stroke(points, iterations):
    """Light moving average to take the shake out of a hand-drawn stroke."""
    pts = list(points)
    for _ in range(max(0, iterations)):
        if len(pts) < 3:
            break
        smoothed = [pts[0]]
        for i in range(1, len(pts) - 1):
            smoothed.append((pts[i - 1] + pts[i] * 2.0 + pts[i + 1]) * 0.25)
        smoothed.append(pts[-1])
        pts = smoothed
    return pts


def chord_length_parameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[-1] + (points[i] - points[i - 1]).length)
    total = u[-1]
    if total <= 1e-12:
        n = max(1, len(points) - 1)
        return [i / n for i in range(len(points))]
    return [value / total for value in u]


def fit_generate_bezier(points, params, t1, t2):
    """Least-squares cubic through the samples with the given end tangents."""
    p0, p3 = points[0], points[-1]
    c00 = c01 = c11 = x0 = x1 = 0.0
    for point, u in zip(points, params):
        s = 1.0 - u
        a0 = t1 * (3.0 * u * s * s)
        a1 = t2 * (3.0 * u * u * s)
        c00 += a0.dot(a0)
        c01 += a0.dot(a1)
        c11 += a1.dot(a1)
        base = (p0 * (s * s * s) + p0 * (3.0 * u * s * s)
                + p3 * (3.0 * u * u * s) + p3 * (u * u * u))
        tmp = point - base
        x0 += a0.dot(tmp)
        x1 += a1.dot(tmp)

    det = c00 * c11 - c01 * c01
    seg_len = (p3 - p0).length
    epsilon = 1e-6 * seg_len
    if abs(det) < 1e-12:
        alpha_l = alpha_r = seg_len / 3.0
    else:
        alpha_l = (x0 * c11 - c01 * x1) / det
        alpha_r = (c00 * x1 - c01 * x0) / det
        if alpha_l < epsilon or alpha_r < epsilon:
            alpha_l = alpha_r = seg_len / 3.0
    return [p0, p0 + t1 * alpha_l, p3 + t2 * alpha_r, p3]


def fit_max_error(points, bez, params):
    """Largest squared distance from a sample to the curve, and where."""
    worst = 0.0
    split = len(points) // 2
    for i in range(1, len(points) - 1):
        dist = (bezier_at(bez, params[i]) - points[i]).length_squared
        if dist >= worst:
            worst = dist
            split = i
    return worst, split


def fit_newton_raphson(bez, point, u):
    """One Newton step toward the parameter closest to 'point'."""
    s = 1.0 - u
    q1 = [(bez[i + 1] - bez[i]) * 3.0 for i in range(3)]
    q2 = [(q1[i + 1] - q1[i]) * 2.0 for i in range(2)]
    d = bezier_at(bez, u) - point
    du = q1[0] * (s * s) + q1[1] * (2.0 * s * u) + q1[2] * (u * u)
    duu = q2[0] * s + q2[1] * u
    denominator = du.dot(du) + d.dot(duu)
    if abs(denominator) < 1e-12:
        return u
    return u - d.dot(du) / denominator


def fit_cubic(points, t1, t2, tolerance_sq, depth=0):
    """Returns a list of cubic control-point quads covering 'points'."""
    if len(points) < 2:
        return []
    if len(points) == 2:
        dist = (points[1] - points[0]).length / 3.0
        return [[points[0], points[0] + t1 * dist,
                 points[1] + t2 * dist, points[1]]]

    params = chord_length_parameterize(points)
    bez = fit_generate_bezier(points, params, t1, t2)
    worst, split = fit_max_error(points, bez, params)
    if worst < tolerance_sq:
        return [bez]

    # Close enough to be worth improving the parameterisation before splitting.
    if worst < tolerance_sq * 4.0:
        for _ in range(FIT_NEWTON_PASSES):
            params = [fit_newton_raphson(bez, points[i], params[i])
                      for i in range(len(points))]
            bez = fit_generate_bezier(points, params, t1, t2)
            worst, split = fit_max_error(points, bez, params)
            if worst < tolerance_sq:
                return [bez]

    if depth >= FIT_MAX_DEPTH or split <= 0 or split >= len(points) - 1:
        return [bez]

    centre = points[split - 1] - points[split + 1]
    if centre.length_squared < 1e-16:
        return [bez]
    centre.normalize()
    left = fit_cubic(points[:split + 1], t1, centre, tolerance_sq, depth + 1)
    right = fit_cubic(points[split:], -centre, t2, tolerance_sq, depth + 1)
    return left + right


def split_at_corners(points, angle_degrees, span):
    """
    Break the stroke where it genuinely turns, so corners stay corners.

    The turn is measured over a fixed arc-length 'span' rather than a fixed
    number of samples: with dense input, hand shake between neighbouring
    samples swamps the real direction and invents corners that are not there.
    """
    if angle_degrees <= 0.0 or len(points) < 5 or span <= 0.0:
        return [points]

    acc = [0.0]
    for i in range(1, len(points)):
        acc.append(acc[-1] + (points[i] - points[i - 1]).length)
    if acc[-1] <= span * 2.0:
        return [points]

    n = len(points)
    runs = []
    start = 0
    i = 1
    while i < n - 1:
        back = i
        while back > 0 and acc[i] - acc[back] < span:
            back -= 1
        fwd = i
        while fwd < n - 1 and acc[fwd] - acc[i] < span:
            fwd += 1
        # Both arms must cover real distance. Near the ends they cannot, and
        # comparing against one jitter-sized segment invents corners.
        if (back < i and fwd > i
                and acc[i] - acc[back] >= span * 0.5
                and acc[fwd] - acc[i] >= span * 0.5):
            a = points[i] - points[back]
            b = points[fwd] - points[i]
            if a.length_squared > 1e-16 and b.length_squared > 1e-16:
                dot = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
                if math.degrees(math.acos(dot)) >= angle_degrees and i - start >= 2:
                    runs.append(points[start:i + 1])
                    start = i
                    i = fwd            # clear of this corner before looking again
                    continue
        i += 1
    runs.append(points[start:])
    return [r for r in runs if len(r) >= 2]


def segments_to_knots(segments, closed):
    if not segments:
        return None
    knots = [Knot(segments[0][0], segments[0][0], segments[0][1],
                  'FREE', 'FREE', True)]
    for i, seg in enumerate(segments):
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        right = nxt[1] if nxt is not None else seg[3]
        knots.append(Knot(seg[3], seg[2], right, 'FREE', 'FREE', True))

    if closed and len(knots) > 2:
        knots[0].hl = segments[-1][2].copy()
        knots.pop()
    else:
        knots[0].hl = knots[0].co * 2.0 - knots[0].hr
        knots[-1].hr = knots[-1].co * 2.0 - knots[-1].hl
    return knots


def fit_stroke(points, tolerance, smoothing, corner_angle, closed):
    """Turn raw stroke samples into bezier knots."""
    if len(points) < 2:
        return None, False
    pts = dedupe_points(points)
    if len(pts) < 2:
        return None, False
    pts = smooth_stroke(pts, smoothing)
    if closed and (pts[-1] - pts[0]).length_squared > 1e-12:
        pts = pts + [pts[0].copy()]

    tolerance_sq = max(1e-12, tolerance * tolerance)
    total = sum((pts[i] - pts[i - 1]).length for i in range(1, len(pts)))
    span = max(tolerance * 6.0, total * 0.02)
    segments = []
    for run in split_at_corners(pts, corner_angle, span):
        t1 = run[1] - run[0]
        t2 = run[-2] - run[-1]
        if t1.length_squared < 1e-16 or t2.length_squared < 1e-16:
            continue
        t1.normalize()
        t2.normalize()
        segments.extend(fit_cubic(run, t1, t2, tolerance_sq))
    knots = segments_to_knots(segments, closed)
    if knots is None or len(knots) < 2:
        return None, False
    return knots, closed


def raw_point_full(context, region, rv3d, mouse):
    """Point, object and normal under the mouse from a single ray cast."""
    cfg = settings(context)
    hit, loc, normal, _idx, ob, _mat = scene_ray(context, region, rv3d, mouse)
    if hit:
        co = loc.copy()
        if cfg.surface_offset > 0.0 and normal.length_squared > 0.0:
            co = loc + normal.normalized() * cfg.surface_offset
        return (co, ob.original if ob is not None else None,
                normal.normalized() if normal.length_squared > 1e-12 else None)
    return plane_point(context, region, rv3d, mouse, cfg.fallback_plane), None, None


class CURVE_OT_maxline_freehand(bpy.types.Operator):
    """Draw a freehand stroke, fitted to a smooth curve"""

    bl_idname = "curve.maxline_freehand"
    bl_label = "Draw Curve"
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: bpy.props.FloatProperty(
        name="Tolerance",
        description="How far the fitted curve may stray from your stroke. "
                    "Larger values give fewer, smoother control points",
        default=6.0, min=0.5, soft_max=60.0, subtype='PIXEL')
    spacing: bpy.props.IntProperty(
        name="Sample Spacing",
        description="Minimum distance between recorded stroke samples",
        default=3, min=1, max=40, subtype='PIXEL')
    smoothing: bpy.props.IntProperty(
        name="Smoothing",
        description="Moving-average passes over the raw stroke before fitting",
        default=2, min=0, max=10)
    corner_angle: bpy.props.FloatProperty(
        name="Corner Angle",
        description="Turns sharper than this stay as corners instead of being "
                    "smoothed through. Set to 0 to smooth everything",
        default=60.0, min=0.0, max=180.0)
    auto_close: bpy.props.BoolProperty(
        name="Auto Close",
        description="Close the curve when the stroke ends near where it began",
        default=True)

    @classmethod
    def poll(cls, context):
        if context.mode != 'OBJECT':
            setter = getattr(cls, "poll_message_set", None)
            if setter is not None:
                setter("Only available in Object Mode")
            return False
        return True

    # -- lifecycle ---------------------------------------------------------

    def invoke(self, context, event):
        self.area, self.region, self.rv3d = find_view3d(context)
        if self.area is None:
            self.report({'WARNING'}, "Draw Curve needs a 3D viewport open")
            return {'CANCELLED'}

        self._reset_stroke()
        self.drawing = False
        self.made = 0
        self.continuous = settings(context).continuous
        self.mouse = self._mouse(event)

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_freehand_preview, (self,), 'WINDOW', 'POST_VIEW')
        context.window.cursor_modal_set('PAINT_BRUSH')
        self._status(context)
        context.window_manager.modal_handler_add(self)
        self.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _reset_stroke(self):
        self.samples = []
        self.screen = []
        self.objs = []
        self.normals = []
        self.preview = []
        self.last_fit = 0

    def _teardown(self, context):
        if getattr(self, "_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
        for action in (lambda: context.window.cursor_modal_restore(),
                       lambda: context.workspace.status_text_set(None),
                       lambda: self.area.tag_redraw()):
            try:
                action()
            except (AttributeError, ReferenceError, TypeError):
                pass

    def _status(self, context):
        context.workspace.status_text_set(
            "Hold LMB and draw   |   %d samples, %d segments   |   "
            "RMB/Enter: finish   |   Esc: cancel"
            % (len(self.samples), len(self.preview)))

    def _mouse(self, event):
        return (event.mouse_x - self.region.x, event.mouse_y - self.region.y)

    def _in_region(self, event):
        x, y = self._mouse(event)
        return 0 <= x < self.region.width and 0 <= y < self.region.height

    # -- stroke ------------------------------------------------------------

    def _add_sample(self, context, event, force=False):
        mouse = self._mouse(event)
        if not force and self.screen:
            last = self.screen[-1]
            if math.hypot(mouse[0] - last[0], mouse[1] - last[1]) < self.spacing:
                return False
        co, obj, normal = raw_point_full(context, self.region, self.rv3d, mouse)
        self.samples.append(co)
        self.screen.append(mouse)
        self.objs.append(obj)
        self.normals.append(normal)
        return True

    def _tolerance_world(self):
        reference = self.samples[0] if self.samples else Vector()
        return self.tolerance * world_per_pixel(self.region, self.rv3d, reference)

    def _is_closed(self):
        if not self.auto_close or len(self.screen) < 8:
            return False
        first, last = self.screen[0], self.screen[-1]
        limit = 20.0
        return math.hypot(last[0] - first[0], last[1] - first[1]) <= limit

    def _refit(self, live=True):
        """Live previews fit a strided subset so the cost stays flat as the
        stroke grows. The committed curve is fitted from every sample."""
        pts = self.samples
        if live and len(pts) > LIVE_FIT_MAX_POINTS:
            stride = len(pts) // LIVE_FIT_MAX_POINTS + 1
            pts = pts[::stride]
            if pts[-1] is not self.samples[-1]:
                pts = pts + [self.samples[-1]]
        knots, closed = fit_stroke(pts, self._tolerance_world(),
                                   self.smoothing, self.corner_angle,
                                   self._is_closed())
        self.preview = tessellate_knots(knots, closed, 8) if knots else []

    def _anchor(self):
        counts = {}
        order = []
        for obj in self.objs:
            if obj is None:
                continue
            if obj.name not in counts:
                counts[obj.name] = 0
                order.append(obj)
            counts[obj.name] += 1
        if not order:
            return None
        return max(order, key=lambda ob: counts[ob.name])

    def _anchor_normal(self, anchor):
        chosen = [n for n, o in zip(self.normals, self.objs)
                  if n is not None and (anchor is None or o is anchor)]
        if not chosen:
            return None
        total = Vector((0.0, 0.0, 0.0))
        for n in chosen:
            total = total + n
        return total.normalized() if total.length_squared > 1e-12 else chosen[0]

    def _store(self):
        anchor = self._anchor()
        FREEHAND_STROKE['points'] = [p.copy() for p in self.samples]
        FREEHAND_STROKE['anchor'] = anchor.name if anchor is not None else ""
        FREEHAND_STROKE['normal'] = self._anchor_normal(anchor)
        FREEHAND_STROKE['scale'] = world_per_pixel(
            self.region, self.rv3d, self.samples[0])
        FREEHAND_STROKE['closed'] = self._is_closed()

    # -- modal -------------------------------------------------------------

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            self._teardown(context)
            return {'FINISHED'} if self.made else {'CANCELLED'}

        self.area.tag_redraw()
        if event.type in NAV_EVENTS and not self.drawing:
            return {'PASS_THROUGH'}

        inside = self._in_region(event)

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS' and inside:
                self._reset_stroke()
                self.drawing = True
                self._add_sample(context, event, force=True)
            elif event.value == 'RELEASE' and self.drawing:
                self.drawing = False
                self._add_sample(context, event, force=True)
                return self._finish_stroke(context)
            return {'RUNNING_MODAL'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if self.drawing and inside and self._add_sample(context, event):
                count = len(self.samples)
                if count - self.last_fit >= LIVE_FIT_EVERY:
                    self._refit()
                    self.last_fit = count
                    self._status(context)
            return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} \
                and event.value == 'PRESS':
            self._teardown(context)
            return {'FINISHED'} if self.made else {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def _finish_stroke(self, context):
        if len(self.samples) < 3:
            self._reset_stroke()
            return {'RUNNING_MODAL'}
        self._store()
        if self.continuous:
            # Multiple strokes in one activation, so each needs its own undo
            # step and the redo panel cannot re-run just the last of them.
            result = self._apply(context, push_undo=True)
            self._reset_stroke()
            self._status(context)
            return {'RUNNING_MODAL'} if result else {'RUNNING_MODAL'}
        self._teardown(context)
        return self.execute(context)

    def _apply(self, context, push_undo):
        data = FREEHAND_STROKE
        knots, closed = fit_stroke(data['points'],
                                   self.tolerance * data['scale'],
                                   self.smoothing, self.corner_angle,
                                   data['closed'] and self.auto_close)
        if not knots:
            return False
        anchor = (bpy.data.objects.get(data['anchor'])
                  if data['anchor'] else None)
        obj = commit_knots(context, knots, closed, anchor, "MaxCurve",
                           push_undo=push_undo, align_normal=data['normal'])
        if obj is not None:
            self.made += 1
            self.report({'INFO'}, "Fitted %d samples to %d control points"
                        % (len(data['points']), len(knots)))
        return obj is not None

    def execute(self, context):
        if len(FREEHAND_STROKE['points']) < 3:
            self.report({'WARNING'}, "No stroke to fit")
            return {'CANCELLED'}
        if not self._apply(context, push_undo=False):
            self.report({'WARNING'}, "Could not fit that stroke")
            return {'CANCELLED'}
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "tolerance")
        layout.prop(self, "smoothing")
        layout.prop(self, "corner_angle")
        layout.prop(self, "auto_close")

    def cancel(self, context):
        self._teardown(context)


def draw_freehand_preview(op):
    context = bpy.context
    if context.area != op.area:
        return
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    if len(op.samples) > 1:
        draw_polyline(op.samples, COL_HANDLE, 1.5)
    for polyline in op.preview:
        draw_polyline(polyline, COL_CURVE, 3.0)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')


# ---------------------------------------------------------------------------
# Live preview of the panel values
# ---------------------------------------------------------------------------

PREVIEW_STATE = {'modal': False, 'key': None, 'lines': [],
                 'handle': None, 'applied': None}
PREVIEW_POINT_LIMIT = 4000


def curve_fingerprint(obj, splines):
    """Cheap signature that changes whenever the preview would need redoing."""
    parts = [obj.name]
    for spline in splines:
        points = spline.bezier_points
        count = len(points)
        co = np.empty(count * 3, dtype=np.float32)
        points.foreach_get("co", co)
        selected = tuple(i for i, bp in enumerate(points)
                         if bp.select_control_point)
        parts.append((count, spline.use_cyclic_u, co.tobytes(), selected))
    return tuple(parts)


def preview_key(obj, cfg, splines):
    return (curve_fingerprint(obj, splines), cfg.preview_mode,
            round(cfg.fillet_amount, 6), cfg.fillet_measure,
            round(cfg.outline_distance, 6), cfg.outline_center,
            cfg.outline_close_ends)


def remember_applied(obj, context):
    """Silence the preview for the state we just produced, so the result reads
    as committed rather than as another pending proposal."""
    try:
        cfg = settings(context)
        splines = [s for s in obj.data.splines if s.type == 'BEZIER']
        PREVIEW_STATE['applied'] = preview_key(obj, cfg, splines) if splines else None
    except Exception:
        PREVIEW_STATE['applied'] = None
    tag_view3d_redraw(context)


def compute_preview(obj, cfg, splines):
    mode = cfg.preview_mode
    lines = []
    for spline in splines:
        if not any(bp.select_control_point for bp in spline.bezier_points):
            continue
        knots, cyclic = read_spline(spline)
        if mode in {'FILLET', 'CHAMFER'}:
            result, _info = fillet_knots(
                knots, cyclic, cfg.fillet_amount, mode == 'CHAMFER',
                cfg.fillet_measure == 'RADIUS', samples=16)
            if result is not None:
                lines.extend(tessellate_knots(result, cyclic, 12))
        elif mode == 'OUTLINE':
            replacement, extras = outline_result(
                knots, cyclic, cfg.outline_distance, cfg.outline_center,
                cfg.outline_close_ends)
            if replacement is not None:
                lines.extend(tessellate_knots(replacement[0], replacement[1], 12))
            for new_knots, new_cyclic in extras:
                lines.extend(tessellate_knots(new_knots, new_cyclic, 12))
    return lines


def draw_live_preview():
    """Draws what the panel's Amount / Distance would produce, before applying."""
    if PREVIEW_STATE['modal']:
        return          # a modal drag is already drawing its own preview
    context = bpy.context
    obj = getattr(context, "object", None)
    if obj is None or obj.type != 'CURVE' or obj.mode != 'EDIT':
        return
    cfg = getattr(context.scene, "maxline_settings", None)
    if cfg is None or not cfg.show_preview or cfg.preview_mode == 'NONE':
        return

    splines = [s for s in obj.data.splines if s.type == 'BEZIER']
    if not splines:
        return
    if sum(len(s.bezier_points) for s in splines) > PREVIEW_POINT_LIMIT:
        return          # too heavy to recompute every redraw

    key = preview_key(obj, cfg, splines)
    if PREVIEW_STATE['applied'] == key:
        return          # this is the result we just committed, not a proposal
    if PREVIEW_STATE['key'] != key:
        PREVIEW_STATE['lines'] = compute_preview(obj, cfg, splines)
        PREVIEW_STATE['key'] = key
    if not PREVIEW_STATE['lines']:
        return

    matrix = obj.matrix_world
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for polyline in PREVIEW_STATE['lines']:
        draw_polyline([matrix @ v for v in polyline], COL_PREVIEW, 2.5)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('LESS_EQUAL')


def draw_live_preview_safe():
    try:
        draw_live_preview()
    except Exception:
        # Never let a draw handler spam the console every redraw.
        PREVIEW_STATE['key'] = None
        PREVIEW_STATE['lines'] = []


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class VIEW3D_PT_maxline(bpy.types.Panel):
    bl_label = "Max Line"
    bl_idname = "VIEW3D_PT_maxline"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"

    def draw(self, context):
        layout = self.layout
        cfg = settings(context)

        col = layout.column(align=True)
        col.operator("curve.maxline_draw",
                     text="Draw Line", icon='IPO_LINEAR')
        col.operator(CURVE_OT_maxline_freehand.bl_idname,
                     text="Draw Curve", icon='IPO_BEZIER')
        row = col.row(align=True)
        row.operator(CURVE_OT_maxline_circle.bl_idname,
                     text="Circle", icon='MESH_CIRCLE')
        props = row.operator(CURVE_OT_maxline_arc.bl_idname,
                             text="Arc", icon='SPHERECURVE')
        props.method = cfg.arc_method
        props.pie_slice = cfg.arc_pie
        row = col.row(align=True)
        props = row.operator(CURVE_OT_maxline_rectangle.bl_idname,
                             text="Rectangle", icon='MESH_PLANE')
        props.corner_radius = cfg.shape_corner_radius
        props.from_center = cfg.rect_from_center
        props = row.operator(CURVE_OT_maxline_ngon.bl_idname,
                             text="NGon", icon='SEQ_CHROMA_SCOPE')
        props.corner_radius = cfg.shape_corner_radius
        props.sides = cfg.ngon_sides
        props.inscribed = cfg.ngon_inscribed

        draw_maxline_settings(layout, context)


def draw_maxline_settings(layout, context):
    """The settings of the Max Line panel, also shown by the Max Line tool's Options."""
    cfg = settings(context)
    ts = context.scene.tool_settings
    box = layout.box()
    box.label(text="Shape Options")
    box.prop(cfg, "ngon_sides")
    box.prop(cfg, "shape_corner_radius")
    row = box.row(align=True)
    row.prop(cfg, "ngon_inscribed", toggle=True)
    row.prop(cfg, "rect_from_center", toggle=True)
    box.prop(cfg, "arc_method", text="")
    box.prop(cfg, "arc_pie")

    col = layout.column(align=True)
    col.prop(cfg, "fallback_plane")
    col.prop(cfg, "surface_offset")

    col = layout.column(align=True)
    col.prop(cfg, "origin_mode")

    col = layout.column(align=True)
    col.prop(cfg, "align_mode")
    col.prop(cfg, "parent_to_surface")

    col = layout.column(align=True)
    col.prop(cfg, "continuous")
    col.prop(cfg, "attach_to_active")

    box = layout.box()
    box.label(text="Snapping (hold Ctrl)")
    row = box.row(align=True)
    row.prop(ts, "use_snap", text="Snap")
    row.prop(ts, snap_element_prop(ts), text="")
    box.prop(cfg, "grid_increment")
    box.prop(cfg, "snap_radius")
    box.prop(cfg, "use_occlusion")

    col = layout.column(align=True)
    col.prop(cfg, "close_radius")
    col.prop(cfg, "drag_threshold")
    col.prop(cfg, "curve_resolution")


class VIEW3D_PT_maxline_tool_settings(bpy.types.Panel):
    """Options of the Max Line tool, in the tool settings of the header."""

    bl_label = "Max Line Options"
    bl_idname = "VIEW3D_PT_maxline_tool_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_ui_units_x = 14

    def draw(self, context):
        draw_maxline_settings(self.layout, context)


class VIEW3D_PT_maxline_edit(bpy.types.Panel):
    bl_label = "Max Line Edit"
    bl_idname = "VIEW3D_PT_maxline_edit"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == 'CURVE' and ob.mode == 'EDIT'

    def draw(self, context):
        layout = self.layout
        cfg = settings(context)
        drag = cfg.use_drag

        layout.operator(CURVE_OT_maxline_refine.bl_idname,
                        text="Refine", icon='ADD')

        box = layout.box()
        box.label(text="Fillet / Chamfer")
        box.prop(cfg, "fillet_amount")
        box.prop(cfg, "fillet_measure", text="")
        row = box.row(align=True)
        # EXEC_DEFAULT skips invoke(), so no modal can start and one click
        # applies. INVOKE_DEFAULT is only used when the drag is asked for.
        row.operator_context = 'INVOKE_DEFAULT' if drag else 'EXEC_DEFAULT'
        for idname, label, icon in (
                (CURVE_OT_maxline_fillet.bl_idname, "Fillet", 'SPHERECURVE'),
                (CURVE_OT_maxline_chamfer.bl_idname, "Chamfer", 'LINCURVE')):
            props = row.operator(idname, text=label, icon=icon)
            props.amount = cfg.fillet_amount
            props.measure = cfg.fillet_measure

        box = layout.box()
        box.label(text="Outline")
        box.prop(cfg, "outline_distance")
        row = box.row(align=True)
        row.prop(cfg, "outline_center", toggle=True)
        row.prop(cfg, "outline_close_ends", toggle=True)
        sub = box.column()
        sub.operator_context = 'INVOKE_DEFAULT' if drag else 'EXEC_DEFAULT'
        props = sub.operator(CURVE_OT_maxline_outline.bl_idname,
                             text="Outline", icon='MOD_OFFSET')
        props.distance = cfg.outline_distance
        props.center = cfg.outline_center
        props.close_ends = cfg.outline_close_ends

        adjust = layout.row()
        adjust.operator_context = 'INVOKE_DEFAULT'
        adjust.operator("screen.redo_last", text="Adjust Last Operation",
                        icon='RECOVER_LAST')

        row = layout.row(align=True)
        row.prop(cfg, "show_preview", toggle=True,
                 icon='HIDE_OFF' if cfg.show_preview else 'HIDE_ON')
        row.prop(cfg, "preview_mode", text="")
        layout.prop(cfg, "use_drag")


def edit_menu_func(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(CURVE_OT_maxline_refine.bl_idname, text="Refine")
    layout.operator(CURVE_OT_maxline_fillet.bl_idname, text="Fillet")
    layout.operator(CURVE_OT_maxline_chamfer.bl_idname, text="Chamfer")
    layout.operator(CURVE_OT_maxline_outline.bl_idname, text="Outline")


def menu_func(self, context):
    layout = self.layout
    layout.separator()
    layout.operator("curve.maxline_draw",
                    text="Max Line", icon='IPO_LINEAR')
    layout.operator(CURVE_OT_maxline_freehand.bl_idname,
                    text="Max Curve (Freehand)", icon='IPO_BEZIER')
    layout.operator(CURVE_OT_maxline_circle.bl_idname,
                    text="Max Circle", icon='MESH_CIRCLE')
    layout.operator(CURVE_OT_maxline_arc.bl_idname,
                    text="Max Arc", icon='SPHERECURVE')
    layout.operator(CURVE_OT_maxline_rectangle.bl_idname,
                    text="Max Rectangle", icon='MESH_PLANE')
    layout.operator(CURVE_OT_maxline_ngon.bl_idname,
                    text="Max NGon", icon='SEQ_CHROMA_SCOPE')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    MAXLINE_PG_settings,
    MAXLINE_PG_commit_point,
    CURVE_OT_maxline_commit,
    CURVE_OT_maxline_freehand,
    CURVE_OT_maxline_circle,
    CURVE_OT_maxline_arc,
    CURVE_OT_maxline_rectangle,
    CURVE_OT_maxline_ngon,
    CURVE_OT_maxline_fillet,
    CURVE_OT_maxline_chamfer,
    CURVE_OT_maxline_refine,
    CURVE_OT_maxline_outline,
    VIEW3D_PT_maxline,
    VIEW3D_PT_maxline_edit,
    VIEW3D_PT_maxline_tool_settings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.maxline_settings = bpy.props.PointerProperty(
        type=MAXLINE_PG_settings)
    bpy.types.VIEW3D_MT_curve_add.append(menu_func)
    edit_menu = getattr(bpy.types, 'VIEW3D_MT_edit_curve_ctrlpoints', None)
    if edit_menu is not None:
        edit_menu.append(edit_menu_func)

    for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post):
        if _snap_cache_invalidate not in handlers:
            handlers.append(_snap_cache_invalidate)

    if PREVIEW_STATE['handle'] is None:
        PREVIEW_STATE['handle'] = bpy.types.SpaceView3D.draw_handler_add(
            draw_live_preview_safe, (), 'WINDOW', 'POST_VIEW')


def unregister():
    if PREVIEW_STATE['handle'] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(
            PREVIEW_STATE['handle'], 'WINDOW')
        PREVIEW_STATE['handle'] = None
    PREVIEW_STATE['key'] = None
    PREVIEW_STATE['lines'] = []
    for handlers in (bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.load_post):
        if _snap_cache_invalidate in handlers:
            handlers.remove(_snap_cache_invalidate)

    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)
    edit_menu = getattr(bpy.types, 'VIEW3D_MT_edit_curve_ctrlpoints', None)
    if edit_menu is not None:
        edit_menu.remove(edit_menu_func)
    if hasattr(bpy.types.Scene, "maxline_settings"):
        del bpy.types.Scene.maxline_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
