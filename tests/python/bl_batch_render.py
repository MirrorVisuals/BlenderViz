# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Batch Camera Renderer (built in): render setups with their own collections and Light Mix.
# ./blender.bin --background --factory-startup --python tests/python/bl_batch_render.py
import glob
import json
import os
import shutil
import sys
import tempfile
import addon_utils
import bpy
from cycles import light_mix

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 8
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.cycles.seed = 1
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.use_compositing = False
world = bpy.data.worlds.new("Black")
scene.world = world
if world.node_tree:
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)

bpy.ops.mesh.primitive_plane_add(size=10.0)
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = 10.0
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
cam.location = (0, 0, 10)
scene.camera = cam

view_layer = scene.view_layers[0]
for name, x in (("A", -2.0), ("B", 2.0)):
    data = bpy.data.lights.new(name, 'POINT')
    data.energy = 300.0
    ob = bpy.data.objects.new(name, data)
    scene.collection.objects.link(ob)
    ob.location = (x, 0, 3.0)
    view_layer.lightgroups.add(name=name)
    ob.lightgroup = name
view_layer.use_light_mix = True

# Collections: Furniture (a box in the middle), Plants > Trees (a box on the side).
furniture = bpy.data.collections.new("Furniture")
plants = bpy.data.collections.new("Plants")
trees = bpy.data.collections.new("Trees")
scene.collection.children.link(furniture)
scene.collection.children.link(plants)
plants.children.link(trees)
for collection, location in ((furniture, (0, 0, 0.5)), (trees, (3, 3, 0.5))):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    box = bpy.context.object
    for user in list(box.users_collection):
        user.objects.unlink(box)
    collection.objects.link(box)

out_dir = tempfile.mkdtemp(prefix="bcr_")
props = scene.bcr

print("Built in")
check(hasattr(bpy.types.Scene, "bcr") and hasattr(bpy.ops.bcr, "batch_render"), "Scene.bcr and bcr operators")
check(addon_utils.enable("batch_camera_renderer") is None, "the old add-on is not loaded as well")
check(any(getattr(f, "__name__", "") == "_draw_render_menu"
          for f in getattr(bpy.types.TOPBAR_MT_render.draw, "_draw_funcs", ())), "Render menu entry")

# Light Mix: a saved setup with A off, and a JSON file with A off.
group_a = view_layer.lightgroups["A"]
group_a.use_mix = False
saved = scene.cycles_light_mix.setups.add()
saved.name = "A off"
saved.data = json.dumps(light_mix.setup_from_view_layer(view_layer))
json_path = os.path.join(out_dir, "a_off.json")
with open(json_path, "w") as fh:
    fh.write(saved.data)
group_a.use_mix = True


def add_setup(name):
    bpy.ops.bcr.add_setup()
    setup = props.setups[-1]
    setup.name = name
    setup.camera = cam
    setup.view_layer_name = view_layer.name
    setup.aspect_ratio = 'CUSTOM'
    setup.resolution_preset = 'CUSTOM'
    setup.res_x = setup.res_y = 32
    return setup


reference = add_setup("Reference")
no_furniture = add_setup("NoFurniture")
no_furniture.hidden_collections.add().collection = furniture
no_plants = add_setup("NoPlants")
no_plants.hidden_collections.add().collection = plants
mix_saved = add_setup("MixSaved")
mix_saved.light_mix_source = 'SETUP'
mix_saved.light_mix_setup = "A off"
mix_file = add_setup("MixFile")
mix_file.light_mix_source = 'FILE'
mix_file.light_mix_file = json_path
check(not group_a.use_mix, "the Light Mix of the active setup shows right away")
group_a.use_mix = True
after_mix = add_setup("AfterMix")
missing_file = add_setup("MissingFile")
missing_file.light_mix_source = 'FILE'
missing_file.light_mix_file = os.path.join(out_dir, "nope.json")
hidden = add_setup("Hidden")
hidden.is_hidden = True

# Items of a collection property move when more are added: look them up again by name.


def setup_index(name):
    return props.setups.find(name)


props.output_path = out_dir + os.sep
props.use_date_folder = False
props.include_setup_name = True

print("Duplicate")
props.active_index = setup_index("MixSaved")
bpy.ops.bcr.duplicate_setup()
dup = props.setups[-1]
check(dup.light_mix_source == 'SETUP' and dup.light_mix_setup == "A off", "keeps the Light Mix")
props.active_index = setup_index("NoFurniture")
bpy.ops.bcr.duplicate_setup()
dup2 = props.setups[-1]
check([i.collection for i in dup2.hidden_collections] == [furniture], "keeps the hidden collections")
props.setups.remove(len(props.setups) - 1)
props.setups.remove(len(props.setups) - 1)

print("Collections checklist")
props.active_index = setup_index("Reference")
reference = props.setups["Reference"]
bpy.ops.bcr.toggle_collection(collection_name="Plants")
check([i.collection for i in reference.hidden_collections] == [plants], "unchecking hides it")
bpy.ops.bcr.toggle_collection(collection_name="Plants")
check(len(reference.hidden_collections) == 0, "checking shows it again")
bpy.ops.bcr.toggle_collection(collection_name="Trees", only=True)
check({i.collection for i in reference.hidden_collections} == {furniture},
      "only this one: the others are hidden, its parents stay")
bpy.ops.bcr.show_all_collections()
check(len(reference.hidden_collections) == 0, "Show All")

print("Batch render")
# Selecting a setup shows its Light Mix: put the view layer's own mix back.
props.active_index = setup_index("MixSaved")
props.active_index = setup_index("Reference")
group_a.use_mix = True
resolution = (scene.render.resolution_x, scene.render.resolution_y)
result = bpy.ops.bcr.batch_render()
check(result == {'FINISHED'}, "batch render ran")


def image(setup_name):
    files = glob.glob(os.path.join(out_dir, setup_name + "_Cam_*.exr"))
    if not files:
        return None
    img = bpy.data.images.load(files[0])
    px = list(img.pixels)
    bpy.data.images.remove(img)
    return px


def mean(px):
    return sum(px[i] + px[i + 1] + px[i + 2] for i in range(0, len(px), 4)) / (3 * len(px) // 4)


def difference(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


ref = image("Reference")
check(ref is not None and mean(ref) > 0.01, "the reference is rendered")
statuses = {s.name: s.status for s in props.setups}
check(all(statuses[n] == "Done" for n in ("Reference", "NoFurniture", "NoPlants", "MixSaved", "MixFile",
                                          "AfterMix")),
      "statuses Done ({:s})".format(str(statuses)))
check(statuses["MissingFile"].startswith("Error"), "a missing Light Mix file is an error for that setup only")
check(image("MissingFile") is None and image("Hidden") is None, "no file for the error and the hidden setup")

if ref:
    a = image("NoFurniture")
    check(a is not None and difference(a, ref) > 1e-3, "an unchecked collection is not rendered")
    b = image("NoPlants")
    check(b is not None and difference(b, ref) > 1e-3, "its child collections neither")
    c = image("MixSaved")
    check(c is not None and abs(mean(c) / mean(ref) - 0.5) < 0.1,
          "a saved Light Mix setup is applied ({:.3f} of the light)".format(mean(c) / mean(ref) if c else 0))
    d = image("MixFile")
    check(d is not None and c is not None and difference(c, d) < 1e-6, "a JSON file gives the same image")
    f = image("AfterMix")
    check(f is not None and difference(f, ref) < 1e-6, "a setup without Light Mix after one with it: the own mix")

print("Restored afterwards")
check(not furniture.hide_render and not plants.hide_render, "collections render again")
check(group_a.use_mix and view_layer.lightgroups["B"].use_mix, "the Light Mix of the view layer is back")
check((scene.render.resolution_x, scene.render.resolution_y) == resolution, "the scene resolution is back")

shutil.rmtree(out_dir, ignore_errors=True)
print()
if failures:
    print("{:d} FAILED:".format(len(failures)))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("RESULT: ALL PASSED")
