# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Mesh Light and Cylinder light (BlenderViz, as in V-Ray). A white floor seen from above:
# - a square Mesh Light lights it as an Area light of the same size and power,
# - double sided, half of it goes up: half the light,
# - in Strength units, as an Emission material of that strength, and the materials are kept,
# - a Cylinder light lights it as a Mesh Light on a tube of the same size and power.
# ./blender.bin --background --factory-startup --python tests/python/bl_cycles_mesh_light.py
import math
import os
import sys
import tempfile

import bpy

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


RES = 24
tmpdir = tempfile.mkdtemp()

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 256
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.seed = 3
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
world.color = (0, 0, 0)

# White floor, camera looking down at it from inside the light-free zone.
bpy.ops.mesh.primitive_plane_add(size=4)
floor = bpy.context.object
white = bpy.data.materials.new("White")
white.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
white.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
floor.data.materials.append(white)
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = 1.0
cam = bpy.data.objects.new("Cam", cam_data)
cam.location = (1.2, 0.0, 0.5)
scene.collection.objects.link(cam)
scene.camera = cam


def render(name):
    scene.render.filepath = os.path.join(tmpdir, name + ".exr")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath)
    pixels = image.pixels[:]
    # Mean of R, G and B.
    value = sum(pixels[i] + pixels[i + 1] + pixels[i + 2] for i in range(0, len(pixels), 4))
    value /= 3 * (len(pixels) // 4)
    bpy.data.images.remove(image)
    return value


def hide(ob, hidden=True):
    ob.hide_render = hidden


H = 1.0
SIZE = 0.6
POWER = 60.0

print("Area light (reference)")
bpy.ops.object.light_add(type='AREA', location=(0, 0, H))
area = bpy.context.object
area.data.size = SIZE
area.data.energy = POWER
ref = render("area")
check(ref > 1e-3, "the area light lights the floor (%.4f)" % ref)
hide(area)

print("Mesh Light: a square of the same size and power, facing down")
bpy.ops.mesh.primitive_plane_add(size=SIZE, location=(0, 0, H), rotation=(math.pi, 0, 0))
panel = bpy.context.object
glass = bpy.data.materials.new("Kept")
panel.data.materials.append(glass)
panel.cycles.is_mesh_light = True
panel.cycles.mesh_light_units = 'POWER'
panel.cycles.mesh_light_power = POWER
mesh_light = render("mesh_light")
check(abs(mesh_light / ref - 1.0) < 0.06, "as the area light: %.4f / %.4f" % (mesh_light, ref))
check(panel.material_slots[0].material == glass, "its material is kept")

panel.cycles.mesh_light_double_sided = True
double = render("double")
check(abs(double / mesh_light - 0.5) < 0.05, "double sided: half goes up (%.3f)" % (double / mesh_light))
panel.cycles.mesh_light_double_sided = False

panel.cycles.mesh_light_exposure = 1.0
exposed = render("exposure")
check(abs(exposed / mesh_light - 2.0) < 0.1, "exposure +1: twice the light (%.3f)" % (exposed / mesh_light))
panel.cycles.mesh_light_exposure = 0.0

panel.cycles.mesh_light_color = (1.0, 0.0, 0.0)
red_scene = render("red")
check(red_scene < 0.5 * mesh_light, "a red light gives less white light (%.4f)" % red_scene)
panel.cycles.mesh_light_color = (1.0, 1.0, 1.0)

print("Strength units: as an Emission material of that strength")
panel.cycles.mesh_light_units = 'STRENGTH'
panel.cycles.mesh_light_strength = 20.0
strength = render("strength")
panel.cycles.is_mesh_light = False
emit = bpy.data.materials.new("Emit")
nt = emit.node_tree
nt.nodes.clear()
node = nt.nodes.new("ShaderNodeEmission")
node.inputs["Strength"].default_value = 20.0
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(node.outputs[0], out.inputs["Surface"])
panel.material_slots[0].material = emit
emission = render("emission")
check(abs(strength / emission - 1.0) < 0.06, "as the Emission material: %.4f / %.4f" % (strength, emission))
hide(panel)

print("Cylinder light: as a Mesh Light on a tube of the same size and power")
LENGTH = 1.0
RADIUS = 0.05
bpy.ops.cycles.cylinder_light_add(length=LENGTH, radius=RADIUS, power=POWER)
tube = bpy.context.object
check(tube.type == 'LIGHT' and tube.data.shape == 'CYLINDER', "Add > Light > Cylinder")
tube.location = (0, 0, H)
tube.rotation_euler = (0, math.pi / 2, 0)
cylinder = render("cylinder")
check(cylinder > 1e-3, "the cylinder light lights the floor (%.4f)" % cylinder)
hide(tube)
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=RADIUS, depth=LENGTH, location=(0, 0, H),
                                    rotation=(0, math.pi / 2, 0))
twin = bpy.context.object
twin.cycles.is_mesh_light = True
twin.cycles.mesh_light_power = POWER
mesh_tube = render("mesh_tube")
check(abs(cylinder / mesh_tube - 1.0) < 0.06, "as the mesh tube: %.4f / %.4f" % (cylinder, mesh_tube))
hide(twin)

hide(tube, False)
tube.data.energy = 2 * POWER
doubled = render("cylinder_double")
check(abs(doubled / cylinder - 2.0) < 0.12, "twice the power, twice the light (%.3f)" % (doubled / cylinder))

print("Light Mix sees Mesh Lights as lights")
from cycles import light_mix  # noqa: E402 (the add-on is loaded with Blender)
twin.hide_render = False
check(light_mix.is_mesh_light(twin) and not light_mix.is_mesh_light(floor), "is_mesh_light")

if failures:
    print("\n%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("\nRESULT: ALL PASSED")
