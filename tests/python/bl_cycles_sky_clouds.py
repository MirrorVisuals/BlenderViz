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
check(sky.cloud_map is not None, "with clouds from a photo, that come with BlenderViz")
check(sky.cloud_map is not None and sky.cloud_map.packed_file is not None,
      "the cloud map is packed into the file")
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


print("Photo clouds")
look(55)
sky.use_clouds = False
clear = render("clear")
sky.use_clouds = True
photo = render("photo")
check(difference(photo, clear) > 0.01, "photo clouds change the sky ({:.3f})".format(difference(photo, clear)))
check(mean(photo, 0) / mean(photo, 2) > mean(clear, 0) / mean(clear, 2), "and make it whiter")
sky.cloud_direction += math.radians(90.0)
check(difference(render("photo_turned"), photo) > 0.005, "Rotation turns them")
sky.cloud_direction -= math.radians(90.0)
sky.cloud_map_shading = 0.0
check(difference(render("photo_flat"), photo) > 1e-4, "Photo Shading changes them")
sky.cloud_map_shading = 0.5
check(abs(sky.cloud_detail - 0.3) < 1e-6, "Detail on by default")
# Detail is finer than a pixel of this small wide view: look closer.
cam.data.lens = 200
detailed = render("photo_detailed")
sky.cloud_detail = 0.0
smooth = render("photo_smooth")
check(difference(smooth, detailed) > 1e-4,
      "Detail changes them, seen close ({:.5f})".format(difference(smooth, detailed)))
sky.cloud_detail = 0.3
check(abs(sky.cloud_edge_softness - 0.5) < 1e-6 and abs(sky.cloud_edge_detail - 0.5) < 1e-6,
      "Edge Softness and Edge Detail on by default")
sky.cloud_edge_softness = 0.0
sky.cloud_edge_detail = 0.0
check(difference(render("photo_hard_edges"), detailed) > 1e-4, "the edge settings change them")
sky.cloud_edge_softness = 0.5
sky.cloud_edge_detail = 0.5
cam.data.lens = 24
sky.cloud_density = 0.0
check(difference(render("photo_none"), clear) < 1e-6, "Density 0: exactly the clear sky")
sky.cloud_density = 1.0

print("Cloud map from HDRI")
# An equirectangular HDRI: blue sky, a white cloud 60 degrees up toward +X, a sun toward -Y.
w, h = 256, 128
pixels = []
for y in range(h):
    elevation = (y + 0.5) / h * math.pi - math.pi / 2
    for x in range(w):
        azimuth = (0.5 - (x + 0.5) / w) * 2 * math.pi
        dx = math.cos(elevation) * math.cos(azimuth)
        dy = math.cos(elevation) * math.sin(azimuth)
        dz = math.sin(elevation)
        if dz < 0:
            pixels += (0.1, 0.08, 0.05, 1.0)
            continue
        r, g, b = 0.15, 0.3, 0.8
        cloud = math.exp(-((dx - 0.5) ** 2 + dy ** 2 + (dz - 0.866) ** 2) / 0.02)
        r, g, b = (r * (1 - cloud) + 1.2 * cloud, g * (1 - cloud) + 1.2 * cloud,
                   b * (1 - cloud) + 1.2 * cloud)
        sun = 1000.0 if dy < -0.97 and abs(dz - 0.2) < 0.03 else 0.0
        pixels += (r + sun, g + sun, b + sun, 1.0)
hdri = bpy.data.images.new("test_hdri", w, h, float_buffer=True)
hdri.pixels.foreach_set(pixels)
hdri_path = os.path.join(bpy.app.tempdir, "test_hdri.exr")
hdri.filepath_raw = hdri_path
hdri.file_format = 'OPEN_EXR'
hdri.save()
bpy.data.images.remove(hdri)
old_map = sky.cloud_map
check(bpy.ops.cycles.world_cloud_map_from_hdri.poll(), "Cloud Map from HDRI is available")
bpy.ops.cycles.world_cloud_map_from_hdri('EXEC_DEFAULT', filepath=hdri_path, size='2048')
made = sky.cloud_map
check(made is not None and made is not old_map and made.packed_file is not None,
      "it makes a cloud map, packed")
if made is not None and made is not old_map:
    size = made.size[0]
    px = made.pixels[:]

    def opacity_at(dx, dy, dz):
        r = math.acos(dz) / (math.pi / 2)
        rho = math.hypot(dx, dy)
        u = 0.5 + 0.5 * r * dx / rho
        v = 0.5 + 0.5 * r * dy / rho
        i = (int(v * size) * size + int(u * size)) * 4
        return px[i]
    at_cloud = opacity_at(0.5, 0.0, 0.866)
    at_clear = opacity_at(-0.5, 0.3, 0.81)
    check(at_cloud > 0.5 and at_clear < 0.1,
          "the cloud is opaque in it, the clear sky not ({:.2f}, {:.2f})".format(at_cloud, at_clear))
    check(not any(img.filepath == hdri_path for img in bpy.data.images), "the HDRI isn't kept")
sky.cloud_map = old_map

print("Procedural cirrus")
sky.cloud_map = None
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
