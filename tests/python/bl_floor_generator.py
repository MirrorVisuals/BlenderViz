# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Floor Generator modifier and node (BlenderViz): every pattern covers the floor exactly (with no
# grout the tops of the boards add up to the area of the floor), on a rectangle, an L-shaped
# room and a curve outline; grout, height, bevel, variation, missing boards, materials, UVs.
# ./blender.bin --background --factory-startup --python tests/python/bl_floor_generator.py
import math
import sys

import bpy
import bmesh
from mathutils import Vector

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


PATTERNS = ['STANDARD', 'STACKED', 'HERRINGBONE', 'CHEVRON', 'BASKET_WEAVE', 'HEXAGON', 'VERSAILLES']


def new_floor(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def evaluated(ob):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return ob.evaluated_get(depsgraph).to_mesh()


def top_area(mesh, height):
    """Area of the faces on top of the boards (facing up, at the board height)."""
    total = 0.0
    for poly in mesh.polygons:
        if poly.normal.z > 0.999 and abs(poly.center.z - height) < 1e-5:
            total += poly.area
    return total


bpy.ops.wm.read_factory_settings(use_empty=True)

print("Rectangle 4 x 3 m, no grout, no bevel: the boards cover it exactly")
room = new_floor("Room", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], [(0, 1, 2, 3)])
mod = room.modifiers.new("Floor", 'FLOOR_GENERATOR')
check(mod is not None and mod.type == 'FLOOR_GENERATOR', "modifier added")
mod.grout_length = 0.0
mod.grout_width = 0.0
mod.bevel_height = 0.0
mod.height = 0.02
mod.length = 0.9
mod.width = 0.15
mod.hex_radius = 0.2
mod.panel_size = 0.8
counts = {}
for pattern in PATTERNS:
    mod.pattern = pattern
    mesh = evaluated(room)
    area = top_area(mesh, 0.02)
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    inside = min(xs) > -1e-4 and max(xs) < 4 + 1e-4 and min(ys) > -1e-4 and max(ys) < 3 + 1e-4
    ids = mesh.attributes.get("board_id")
    counts[pattern] = (max(v.value for v in ids.data) + 1) if ids else 0
    check(abs(area - 12.0) < 0.01 and inside,
          "%s: top area %.4f of 12, inside the room, %d boards" % (pattern, area, counts[pattern]))
    room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

print("Attributes")
mod.pattern = 'STANDARD'
mesh = evaluated(room)
check("UVMap" in mesh.uv_layers, "UV map")
check("board_id" in mesh.attributes and mesh.attributes["board_id"].domain == 'FACE', "board_id per face")
check("board_random" in mesh.attributes, "board_random per face")
rand = [v.value for v in mesh.attributes["board_random"].data]
check(min(rand) >= 0.0 and max(rand) < 1.0 and len(set(rand)) > 10, "random values per board")
uv = mesh.uv_layers["UVMap"].data
us = [d.uv.x for d in uv]
check(min(us) > -0.01 and max(us) < 1.01, "UVs of a full board in 0..1")
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

print("Herringbone: double herringbone has twice the boards of a pattern twice as wide")
mod.pattern = 'HERRINGBONE'
mod.parallel_boards = 2
mesh = evaluated(room)
area = top_area(mesh, 0.02)
check(abs(area - 12.0) < 0.01, "double herringbone covers the room (%.4f)" % area)
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
mod.parallel_boards = 1

print("Grout, height and bevel")
mod.pattern = 'STANDARD'
mod.grout_length = 0.004
mod.grout_width = 0.004
mesh = evaluated(room)
area = top_area(mesh, 0.02)
check(10.5 < area < 11.9, "grout leaves gaps (%.3f of 12)" % area)
zs = [v.co.z for v in mesh.vertices]
check(abs(min(zs)) < 1e-6 and abs(max(zs) - 0.02) < 1e-6, "boards from 0 to the height")
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
mod.bevel_height = 0.002
mesh = evaluated(room)
area_bevel = top_area(mesh, 0.02)
check(area_bevel < area, "the bevel makes the tops smaller (%.3f < %.3f)" % (area_bevel, area))
sloped = [p for p in mesh.polygons if 0.5 < p.normal.z < 0.9]
check(len(sloped) > 0, "45 degree bevel faces")
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

print("Variation")
mod.rotation_max = math.radians(3)
mod.offset_x_max = 0.01
mod.tilt_max = math.radians(2)
mesh = evaluated(room)
zs = [v.co.z for v in mesh.vertices]
check(max(zs) > 0.0201, "tilt lifts some corners (%.4f)" % max(zs))
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
before = counts['STANDARD']
mod.rotation_max = 0
mod.offset_x_max = 0
mod.tilt_max = 0
mod.missing = 0.3
mesh = evaluated(room)
left = max(v.value for v in mesh.attributes["board_id"].data) + 1
check(0.55 * before < left < 0.85 * before, "30%% missing boards: %d of %d left" % (left, before))
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
mod.missing = 0.0
for i in range(3):
    room.data.materials.append(bpy.data.materials.new("Wood%d" % i))
mod.material_count = 3
mesh = evaluated(room)
indices = set(p.material_index for p in mesh.polygons)
check(indices == {0, 1, 2}, "random materials 0..2 (%s)" % sorted(indices))
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
mod.material_count = 1

print("Same seed, same floor; another seed, another")
mod.use_vary_length = True
mod.length_min = 0.4
a = [tuple(v.co) for v in evaluated(room).vertices]
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
b = [tuple(v.co) for v in evaluated(room).vertices]
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
mod.seed = 7
c = [tuple(v.co) for v in evaluated(room).vertices]
room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
check(a == b and a != c, "deterministic, and the seed changes it")

print("Versailles: grout makes no board disappear (a room of exactly 3 x 2 panels)")
panels = new_floor("Panels", [(0, 0, 0), (2.4, 0, 0), (2.4, 1.6, 0), (0, 1.6, 0)], [(0, 1, 2, 3)])
vmod = panels.modifiers.new("Floor", 'FLOOR_GENERATOR')
vmod.pattern = 'VERSAILLES'
vmod.panel_size = 0.8
vmod.width = 0.08
vmod.bevel_height = 0.0
vmod.grout_length = vmod.grout_width = 0.0
mesh = evaluated(panels)
no_grout = max(v.value for v in mesh.attributes["board_id"].data) + 1
panels.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
vmod.grout_length = vmod.grout_width = 0.003
vmod.bevel_height = 0.0015
mesh = evaluated(panels)
with_grout = max(v.value for v in mesh.attributes["board_id"].data) + 1
panels.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
check(no_grout == with_grout, "%d boards with grout and bevel, %d without" % (with_grout, no_grout))

print("L-shaped room (one concave face), rotated pattern")
l_room = new_floor("LRoom",
                   [(0, 0, 0), (5, 0, 0), (5, 2, 0), (2, 2, 0), (2, 4, 0), (0, 4, 0)],
                   [(0, 1, 2, 3, 4, 5)])
lmod = l_room.modifiers.new("Floor", 'FLOOR_GENERATOR')
lmod.grout_length = lmod.grout_width = 0.0
lmod.bevel_height = 0.0
lmod.direction = math.radians(30)
for pattern in PATTERNS:
    lmod.pattern = pattern
    mesh = evaluated(l_room)
    area = top_area(mesh, lmod.height)
    check(abs(area - 14.0) < 0.02, "%s on the L room: %.4f of 14" % (pattern, area))
    l_room.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

print("Wall (a vertical face) and a curve outline")
wall = new_floor("Wall", [(0, 5, 0), (3, 5, 0), (3, 5, 2), (0, 5, 2)], [(0, 1, 2, 3)])
wmod = wall.modifiers.new("Floor", 'FLOOR_GENERATOR')
wmod.grout_length = wmod.grout_width = 0.0
wmod.bevel_height = 0.0
wmod.height = 0.0
mesh = evaluated(wall)
area = sum(p.area for p in mesh.polygons)
check(abs(area - 6.0) < 0.01, "boards on a wall: %.4f of 6" % area)
wall.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

curve = bpy.data.curves.new("Outline", 'CURVE')
curve.dimensions = '3D'
spline = curve.splines.new('POLY')
spline.points.add(3)
for point, co in zip(spline.points, [(0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)]):
    point.co = (co[0], co[1], co[2], 1.0)
spline.use_cyclic_u = True
outline = bpy.data.objects.new("Outline", curve)
bpy.context.scene.collection.objects.link(outline)
outline.location = (10, 0, 0)
cmod = outline.modifiers.new("Floor", 'FLOOR_GENERATOR')
check(cmod is not None, "modifier on a curve object")
cmod.grout_length = cmod.grout_width = 0.0
cmod.bevel_height = 0.0
mesh = evaluated(outline)
area = top_area(mesh, cmod.height)
check(abs(area - 6.0) < 0.01, "floor in a curve outline: %.4f of 6" % area)
outline.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

print("Floor Generator node")
node_ob = new_floor("NodeRoom", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], [(0, 1, 2, 3)])
tree = bpy.data.node_groups.new("Floor", 'GeometryNodeTree')
tree.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
tree.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
group_in = tree.nodes.new("NodeGroupInput")
group_out = tree.nodes.new("NodeGroupOutput")
floor_node = tree.nodes.new("GeometryNodeFloorGenerator")
tree.links.new(group_in.outputs[0], floor_node.inputs["Geometry"])
tree.links.new(floor_node.outputs["Floor"], group_out.inputs[0])
floor_node.inputs["Grout Length"].default_value = 0.0
floor_node.inputs["Grout Width"].default_value = 0.0
floor_node.inputs["Bevel Height"].default_value = 0.0
nmod = node_ob.modifiers.new("GN", 'NODES')
nmod.node_group = tree
for pattern in ('HERRINGBONE', 'HEXAGON'):
    try:
        floor_node.inputs["Pattern"].default_value = pattern
    except TypeError:
        floor_node.inputs["Pattern"].default_value = pattern.title()
    mesh = evaluated(node_ob)
    area = top_area(mesh, 0.01)
    check(abs(area - 12.0) < 0.01, "node, %s: %.4f of 12" % (pattern, area))
    node_ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()

if failures:
    print("\n%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("\nRESULT: ALL PASSED")
