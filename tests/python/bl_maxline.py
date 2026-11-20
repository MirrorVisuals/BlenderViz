# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Max Line, built in (scripts/startup/viz_maxline.py): it is loaded at startup without being
# enabled, its menu entries and key are there, the shapes and the edit tools work, and the old
# "maxline" extension is not loaded on top of it.
# ./blender.bin --background --factory-startup --python tests/python/bl_maxline.py
import os
import sys

import addon_utils
import bpy
from mathutils import Vector

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

print("Loaded at startup")
check(bpy.ops.curve.maxline_draw.get_rna_type().identifier == "CURVE_OT_maxline_draw",
      "curve.maxline_draw (C)")
for idname in ("commit", "freehand", "circle", "arc", "rectangle", "ngon",
               "fillet", "chamfer", "refine", "outline"):
    check(hasattr(bpy.types, "CURVE_OT_maxline_" + idname), "operator curve.maxline_" + idname)
check(hasattr(bpy.types, "VIEW3D_PT_maxline") and hasattr(bpy.types, "VIEW3D_PT_maxline_edit"),
      "sidebar panels")
check(scene.maxline_settings is not None, "scene settings")
check("viz_maxline" not in bpy.context.preferences.addons, "not an add-on to enable")
draw_funcs = getattr(bpy.types.VIEW3D_MT_curve_add.draw, "_draw_funcs", [])
check(any(getattr(f, "__module__", "") == "viz_maxline" for f in draw_funcs), "Add > Curve menu entries")

print("Toolbar tool")
from bl_ui.space_toolsystem_toolbar import VIEW3D_PT_tools_active  # noqa: E402


def tool_ids(items):
    for item in items:
        if hasattr(item, "idname"):
            yield item.idname
        elif isinstance(item, tuple):
            yield from tool_ids(item)


object_tools = set(tool_ids(VIEW3D_PT_tools_active._tools['OBJECT']))
for tool in ("maxline", "maxline_curve", "maxline_circle", "maxline_arc", "maxline_rectangle", "maxline_ngon"):
    check("builtin." + tool in object_tools, "%s tool in the Object Mode toolbar" % tool)
group = next((item for item in VIEW3D_PT_tools_active._tools['OBJECT']
              if isinstance(item, tuple) and not hasattr(item, "idname")
              and any(getattr(t, "idname", "") == "builtin.maxline" for t in item)), None)
check(group is not None and len(group) == 6, "the Max Line tools are one toolbar group (a flyout)")
check(hasattr(bpy.types, "VIEW3D_PT_maxline_tool_settings"), "tool options popover")

print("Key in the default keymap")
path = os.path.join(bpy.utils.system_resource('SCRIPTS'),
                    "presets", "keyconfig", "keymap_data", "blender_default.py")
with open(path, encoding="utf-8") as f:
    data = f.read()
found = False
kc = bpy.context.window_manager.keyconfigs.default
km = kc.keymaps.get("Object Mode") if kc else None
if km is not None and len(km.keymap_items):
    found = any(k.idname == "curve.maxline_draw" and k.type == 'L' and k.shift and k.alt
                for k in km.keymap_items)
else:
    # No key configuration in background mode: read the keymap data itself.
    found = '("curve.maxline_draw", {"type": \'L\', "value": \'PRESS\', "shift": True, "alt": True}' in data
check(found, "Shift+Alt+L draws a line in Object Mode")
tool_km = kc.keymaps.get("3D View Tool: Object, Max Line") if kc else None
check((tool_km is not None and any(k.idname == "curve.maxline_draw" for k in tool_km.keymap_items))
      or '("Line", "curve.maxline_draw")' in data, "the tool's click keymap")
check(all('("' + label + '", "curve.maxline_' + op + '")' in data
          for label, op in (("Curve", "freehand"), ("Circle", "circle"), ("Arc", "arc"),
                            ("Rectangle", "rectangle"), ("NGon", "ngon"))),
      "click keymaps of the shape tools")
check(all("from_tool" in getattr(bpy.ops.curve, "maxline_" + op).get_rna_type().properties
          for op in ("freehand", "circle", "arc", "rectangle", "ngon")), "shape tools start with the click")

print("Line (what the C tool hands over)")


def new_curve(op, **props):
    before = set(bpy.data.objects)
    result = op(**props)
    new = [ob for ob in bpy.data.objects if ob not in before]
    return result, (new[0] if new else None)


bpy.ops.mesh.primitive_plane_add(size=4.0, location=(5.0, 0.0, 1.0))
floor = bpy.context.object
floor.name = "Floor"
square = [(4.0, -1.0, 1.0), (6.0, -1.0, 1.0), (6.0, 1.0, 1.0), (4.0, 1.0, 1.0)]
result, line = new_curve(
    bpy.ops.curve.maxline_commit,
    points=[{"name": "", "co": co, "object": "Floor", "normal": (0.0, 0.0, 1.0), "has_normal": True}
            for co in square],
    cyclic=True)
check(result == {'FINISHED'} and line is not None and line.type == 'CURVE', "line from points")
if line:
    spline = line.data.splines[0]
    check(spline.use_cyclic_u and len(spline.bezier_points) == 4, "closed, 4 points")
    check(all(p.handle_left_type == 'VECTOR' and p.handle_right_type == 'VECTOR'
              for p in spline.bezier_points), "straight segments (vector handles)")
    world = [line.matrix_world @ p.co for p in spline.bezier_points]
    check(all((a - Vector(b)).length < 1e-5 for a, b in zip(world, square)), "points where they were placed")
    check(bpy.context.object == line and line.select_get(), "the new line is active and selected")
result, _none = new_curve(bpy.ops.curve.maxline_commit, points=[{"name": "", "co": (0.0, 0.0, 0.0), "object": "",
                                                     "normal": (0.0, 0.0, 0.0), "has_normal": False}])
check(result == {'CANCELLED'}, "one point makes no line")

print("Shapes")


result, circle = new_curve(bpy.ops.curve.maxline_circle, radius=2.0)
check(result == {'FINISHED'} and circle is not None and circle.type == 'CURVE', "circle")
if circle:
    spline = circle.data.splines[0]
    check(spline.use_cyclic_u, "circle is closed")
    co = [circle.matrix_world @ p.co for p in spline.bezier_points]
    check(all(abs(v.length - 2.0) < 1e-4 for v in co), "circle points on the radius")

result, rect = new_curve(bpy.ops.curve.maxline_rectangle, width=4.0, height=2.0, from_center=True)
check(result == {'FINISHED'} and rect is not None, "rectangle")
if rect:
    points = rect.data.splines[0].bezier_points
    check(len(points) == 4, "rectangle has 4 corners (%d)" % len(points))
    xs = [(rect.matrix_world @ p.co).x for p in points]
    check(abs(max(xs) - min(xs) - 4.0) < 1e-4, "rectangle width 4")

result, ngon = new_curve(bpy.ops.curve.maxline_ngon, radius=1.0, sides=7)
check(result == {'FINISHED'} and ngon is not None and len(ngon.data.splines[0].bezier_points) == 7,
      "ngon with 7 sides")

print("Edit tools")
if rect:
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = rect
    rect.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.select_all(action='SELECT')
    result = bpy.ops.curve.maxline_fillet(amount=0.25, interactive=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    count = len(rect.data.splines[0].bezier_points)
    check(result == {'FINISHED'} and count == 8, "fillet splits the 4 corners in two (%d points)" % count)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.select_all(action='SELECT')
    result = bpy.ops.curve.maxline_outline(distance=0.5, interactive=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    check(result == {'FINISHED'} and len(rect.data.splines) == 2,
          "outline adds a second spline (%d splines)" % len(rect.data.splines))

print("The old extension is not loaded on top of it")
check(addon_utils.enable("bl_ext.user_default.maxline") is None, "enabling the extension is refused")
check(hasattr(bpy.types, "CURVE_OT_maxline_commit"), "built-in tool still there")

if failures:
    print("\n%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("\nRESULT: ALL PASSED")
