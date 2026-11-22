# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Sky Texture clouds (BlenderViz): cirrus as a layer of the physical sky, lit by its sun.
# ./blender.bin --background --factory-startup --python tests/python/bl_cycles_sky_clouds.py
import math
import os
import sys
import bpy

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 4
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 160
scene.render.resolution_y = 120
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.use_compositing = False

cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.lens = 24


def look(pitch, yaw=0.0):
    cam.location = (0, 0, 1.7)
    cam.rotation_euler = (math.radians(90 + pitch), 0, math.radians(yaw))


print("Operator and panel")
check(bpy.ops.cycles.world_sky_clouds_add.poll(), "Add Sky with Clouds is available")
bpy.ops.cycles.world_sky_clouds_add()
world = scene.world
sky = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")
check(sky.use_clouds and sky.sky_type == 'MULTIPLE_SCATTERING', "a sky with clouds, in the world")
check(abs(sky.cloud_altitude - 9000.0) < 1e-3 and abs(sky.cloud_amount - 0.45) < 1e-6, "cirrus defaults")
sky.sun_elevation = math.radians(40.0)
sky.sun_rotation = math.radians(180.0)


def render(name):
    path = os.path.join(bpy.app.tempdir, "sky_clouds_" + name + ".exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = list(img.pixels)
    bpy.data.images.remove(img)
    return px


def difference(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def mean(px, channel):
    return sum(px[channel::4]) / (len(px) // 4)


def rows(px, y0, y1, width=160):
    return [c for y in range(y0, y1) for c in px[y * width * 4:(y + 1) * width * 4]]


look(55)
sky.use_clouds = False
clear = render("clear")
sky.use_clouds = True
cloudy = render("cloudy")
check(difference(cloudy, clear) > 0.01, "clouds change the sky ({:.3f})".format(difference(cloudy, clear)))
check(mean(cloudy, 0) / mean(cloudy, 2) > mean(clear, 0) / mean(clear, 2),
      "and make it whiter (less blue)")

sky.cloud_amount = 0.0
check(difference(render("none"), clear) < 0.2 * difference(cloudy, clear), "Amount 0: (almost) clear sky")
sky.cloud_amount = 0.45

sky.cloud_seed = 5
check(difference(render("seed"), cloudy) > 0.005, "another seed, other clouds")
sky.cloud_seed = 0
sky.cloud_dapple = 1.0
check(difference(render("dapple"), cloudy) > 0.005, "Dapple changes the clouds")
sky.cloud_dapple = 0.0
sky.cloud_density = 0.0
check(difference(render("no_density"), clear) < 1e-6, "Density 0: exactly the clear sky")
sky.cloud_density = 1.0

print("Horizon")
look(-5)
sky.use_clouds = False
low_clear = render("low_clear")
sky.use_clouds = True
low_cloudy = render("low_cloudy")
# Rows 0..39 are below the horizon (looking 5 degrees down with a 24 mm lens).
check(difference(rows(low_cloudy, 0, 40), rows(low_clear, 0, 40)) < 1e-6, "no clouds below the horizon")

print("Sunset")
look(20)
sky.sun_elevation = math.radians(40.0)
day = render("day")
sky.sun_elevation = math.radians(1.0)
dusk = render("dusk")
check(mean(dusk, 0) / max(mean(dusk, 2), 1e-6) > mean(day, 0) / max(mean(day, 2), 1e-6),
      "at sunset the clouds turn warm")
sky.sun_elevation = math.radians(-1.5)
after = render("after_sunset")
sky.use_clouds = False
after_clear = render("after_sunset_clear")
check(mean(after, 0) > mean(after_clear, 0), "just after sunset they are still lit from above")

print()
if failures:
    print("{:d} FAILED:".format(len(failures)))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("RESULT: ALL PASSED")
