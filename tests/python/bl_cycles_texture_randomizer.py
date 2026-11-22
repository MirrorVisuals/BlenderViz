# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Texture Randomizer (BlenderViz), as the V-Ray UVW Randomizer: random offset, rotation, scale and
# flip of the texture coordinates per object, object index, element, face, particle or UV tile.
# An emissive plane shows a texture of its own UVs (red = U, green = V), seen from above, so each
# pixel shows the coordinates the texture was looked up at.
# ./blender.bin --background --factory-startup --python tests/python/bl_cycles_texture_randomizer.py
import math
import os
import sys
import bpy

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


RES = 128
TEX = 64

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.pixel_filter_type = 'BOX'
scene.cycles.filter_width = 0.01
scene.cycles.seed = 1
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.use_compositing = False
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new("Black")
scene.world = world
if world.node_tree:
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = 2.0
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
cam.location = (0, 0, 5)
scene.camera = cam

# The texture: red = U and green = V of the texel center.
image = bpy.data.images.new("UVs", TEX, TEX, float_buffer=True)
image.colorspace_settings.name = 'Non-Color'
pixels = []
for y in range(TEX):
    for x in range(TEX):
        pixels += [(x + 0.5) / TEX, (y + 0.5) / TEX, 0.0, 1.0]
image.pixels = pixels
image.pack()

material = bpy.data.materials.new("Randomized")
tree = material.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
output = tree.nodes.new("ShaderNodeOutputMaterial")
emission = tree.nodes.new("ShaderNodeEmission")
tex = tree.nodes.new("ShaderNodeTexImage")
tex.image = image
tex.interpolation = 'Closest'
mapping = tree.nodes.new("ShaderNodeMapping")
uv = tree.nodes.new("ShaderNodeTexCoord")
randomizer = tree.nodes.new("ShaderNodeTexRandomizer")
tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
tree.links.new(uv.outputs["UV"], mapping.inputs["Vector"])


def use_randomizer(on):
    for link in list(tree.links):
        if link.to_socket in (tex.inputs["Vector"], randomizer.inputs["Vector"]):
            tree.links.remove(link)
    if on:
        tree.links.new(mapping.outputs["Vector"], randomizer.inputs["Vector"])
        tree.links.new(randomizer.outputs["Vector"], tex.inputs["Vector"])
    else:
        tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])


def set_ranges(u_offset=(0, 0), v_offset=(0, 0), rotation=(0, 0), u_scale=(100, 100), steps=0):
    inputs = randomizer.inputs
    inputs["U Offset From"].default_value, inputs["U Offset To"].default_value = u_offset
    inputs["V Offset From"].default_value, inputs["V Offset To"].default_value = v_offset
    inputs["Rotation From"].default_value, inputs["Rotation To"].default_value = rotation
    inputs["U Scale From"].default_value, inputs["U Scale To"].default_value = u_scale
    inputs["Rotation Steps"].default_value = steps


def set_modes(**flags):
    for name in ("object", "object_index", "element", "face", "particle", "stochastic"):
        setattr(randomizer, "use_" + name, flags.get(name, False))


def add_plane(name, location, size=2.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    ob = bpy.context.object
    ob.name = name
    ob.data.materials.append(material)
    return ob


def render(name):
    path = os.path.join(bpy.app.tempdir, "texture_randomizer_" + name + ".exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = list(img.pixels)
    bpy.data.images.remove(img)
    return px


def pixel(px, x, y):
    i = (y * RES + x) * 4
    return px[i], px[i + 1]


def difference(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def socket(name):
    """Also hidden sockets, which a lookup by name skips."""
    return next(s for s in randomizer.inputs if s.identifier == name)


print("Node")
check(randomizer.use_element and not randomizer.use_stochastic, "By Element by default")
check(not socket("Tile Blend").enabled, "Tile Blend only with Stochastic Tiling")
randomizer.use_stochastic = True
check(socket("Tile Blend").enabled, "Stochastic Tiling shows Tile Blend")
randomizer.use_stochastic = False
check(not socket("V Scale From").enabled, "V Scale only when separate")
randomizer.use_v_scale = True
check(socket("V Scale From").enabled, "Separate V Scale shows it")
randomizer.use_v_scale = False

plane = add_plane("Plane", (0, 0, 0))
use_randomizer(False)
reference = render("reference")
check(abs(pixel(reference, 32, 32)[0] - 0.25) < 0.02 and abs(pixel(reference, 96, 32)[0] - 0.75) < 0.02,
      "the texture shows the UVs")

print("No change")
use_randomizer(True)
set_modes(element=True, stochastic=True)
set_ranges()
check(difference(render("identity"), reference) < 1e-6, "no variance: the same image")
set_modes()
set_ranges(u_offset=(0.0, 1.0))
check(difference(render("no_mode"), reference) < 1e-6, "no mode checked: no change, as in V-Ray")

print("Stochastic tiling")
mapping.inputs["Scale"].default_value = (4.0, 4.0, 1.0)
use_randomizer(False)
tiled = render("tiled")
use_randomizer(True)
set_modes(stochastic=True)
set_ranges(rotation=(0.0, 2.0 * math.pi), steps=4)
randomizer.inputs["Tile Blend"].default_value = 0.0
rotated = render("rotated")
# A point a quarter into each tile (32 pixels per tile) turns into one of the four quarter points.
quarters = []
for ty in range(4):
    for tx in range(4):
        quarters.append(pixel(rotated, tx * 32 + 8, ty * 32 + 8))
on_quarter = all(min(abs(c - 0.27), abs(c - 0.73)) < 0.04 for q in quarters for c in q)
check(on_quarter, "Rotation 0-360 in 4 steps: quarter turns only")
check(len({(round(q[0], 1), round(q[1], 1)) for q in quarters}) >= 3, "the tiles turn differently")
check(len({(round(pixel(tiled, tx * 32 + 8, 8)[0], 2)) for tx in range(4)}) == 1,
      "(without it every tile is the same)")

set_ranges(u_offset=(0.0, 1.0), v_offset=(0.0, 1.0))
offset_a = render("offset_a")
check(difference(render("offset_b"), offset_a) < 1e-6, "the same seed gives the same image")
randomizer.inputs["Seed"].default_value = 7.0
check(difference(render("offset_seed"), offset_a) > 1e-3, "another seed gives another image")
randomizer.inputs["Seed"].default_value = 0.0


def seam_jump(px):
    """Sum of the color jumps across the vertical tile edges."""
    edge = 0.0
    for y in range(4, RES, 8):
        for x in range(32, RES, 32):
            edge += abs(pixel(px, x, y)[0] - pixel(px, x - 1, y)[0]) + \
                abs(pixel(px, x, y)[1] - pixel(px, x - 1, y)[1])
    return edge


scene.cycles.samples = 4
tex.interpolation = 'Linear'
hard = seam_jump(render("blend_0"))
randomizer.inputs["Tile Blend"].default_value = 0.5
soft = seam_jump(render("blend_half"))
check(soft < 0.5 * hard, "Tile Blend hides the tile edges ({:.2f} -> {:.2f})".format(hard, soft))
randomizer.inputs["Tile Blend"].default_value = 0.0
scene.cycles.samples = 1
tex.interpolation = 'Closest'
mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)

print("Per element, face, object")
set_ranges(u_offset=(0.0, 1.0), v_offset=(0.0, 1.0))
bpy.data.objects.remove(plane)
left = add_plane("Left", (-0.5, 0, 0), size=1.0)
right = add_plane("Right", (0.5, 0, 0), size=1.0)


def halves_differ(px):
    return difference([c for y in range(RES) for x in range(8, 56) for c in pixel(px, x, y)],
                      [c for y in range(RES) for x in range(72, 120) for c in pixel(px, x, y)]) > 1e-3


set_modes(object=True)
check(halves_differ(render("by_object")), "By Object: two objects differ")
set_modes(object_index=True)
check(not halves_differ(render("by_index_same")), "By Object Index: the same index, the same change")
right.pass_index = 3
check(halves_differ(render("by_index")), "a different index, a different change")

# One object with two separate parts (elements), and two material slots (face IDs).
bpy.ops.object.select_all(action='DESELECT')
left.select_set(True)
right.select_set(True)
bpy.context.view_layer.objects.active = left
bpy.ops.object.join()
joined = bpy.context.object
set_modes(element=True)
check(halves_differ(render("by_element")), "By Element: the two parts of one mesh differ")
set_modes(face=True)
check(not halves_differ(render("by_face_same")), "By Face ID: one material index, one change")
joined.data.materials.append(material)
for poly in joined.data.polygons:
    poly.material_index = 1 if poly.center.x > 0 else 0
check(halves_differ(render("by_face")), "faces with another material index differ")

print("Random output")
set_modes(element=True)
tree.links.new(randomizer.outputs["Random"], emission.inputs["Color"])
random_px = render("random")
a = pixel(random_px, 32, 64)[0]
b = pixel(random_px, 96, 64)[0]
check(0.0 <= a < 1.0 and 0.0 <= b < 1.0 and abs(a - b) > 1e-4, "Random: a value per element")
tree.links.new(tex.outputs["Color"], emission.inputs["Color"])

print("OSL")
scene.cycles.shading_system = True
set_modes(element=True)
osl_px = render("osl")
check(halves_differ(osl_px), "with Open Shading Language too")
scene.cycles.shading_system = False

print()
if failures:
    print("{:d} FAILED:".format(len(failures)))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("RESULT: ALL PASSED")
