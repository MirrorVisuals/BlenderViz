# SPDX-FileCopyrightText: 2026 Mirror Visuals
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""BlenderViz: Batch Camera Renderer (was the "Batch Camera Renderer" add-on by Mirror Visuals).

Render setups: a camera with its own view layer, world, resolution, frames, hidden objects, visible
collections and Light Mix, rendered one after the other. In the 3D Viewport sidebar (Batch Render tab),
Render > Batch Camera Renderer, and Render Properties. The data stays in ``Scene.bcr``, so files made with
the add-on keep their setups.
"""

import bpy
import os
import sys
import datetime
from pathlib import Path
from bpy.props import (
    StringProperty,
    BoolProperty,
    EnumProperty,
    IntProperty,
    CollectionProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, Operator, Panel, UIList
from bpy.app.translations import pgettext_rpt as rpt_


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

ASPECT_PRESETS = [
    ('CUSTOM', "Custom", "Use custom resolution"),
    ('1:1', "1 : 1", "Square"),
    ('9:16', "9 : 16", "Vertical"),
    ('16:9', "16 : 9", "Widescreen"),
    ('3:4', "3 : 4", "Portrait"),
    ('4:3', "4 : 3", "Standard"),
    ('3:2', "3 : 2", "Classic"),
    ('2:3', "2 : 3", "Portrait Classic"),
    ('5:4', "5 : 4", "Photo"),
    ('4:5', "4 : 5", "Portrait Photo"),
    ('21:9', "21 : 9", "Ultra-wide"),
    ('9:21', "9 : 21", "Ultra-tall"),
    ('2:1', "2 : 1", "Panoramic"),
    ('1:2', "1 : 2", "Tall Panoramic"),
]

# Width:Height ratios for computation
ASPECT_RATIOS = {
    '1:1': (1, 1),
    '9:16': (9, 16),
    '16:9': (16, 9),
    '3:4': (3, 4),
    '4:3': (4, 3),
    '3:2': (3, 2),
    '2:3': (2, 3),
    '5:4': (5, 4),
    '4:5': (4, 5),
    '21:9': (21, 9),
    '9:21': (9, 21),
    '2:1': (2, 1),
    '1:2': (1, 2),
}

RESOLUTION_PRESETS = [
    ('CUSTOM', "Custom", "Use custom resolution"),
    ('SD', "SD (480p)", "Standard Definition"),
    ('HD', "HD (720p)", "High Definition"),
    ('FHD', "Full HD (1080p)", "Full High Definition"),
    ('2K', "2K QHD (1440p)", "Quad High Definition"),
    ('4K', "4K UHD (2160p)", "Ultra High Definition"),
    ('8K', "8K (4320p)", "8K Ultra HD"),
]

# Short edge (height for landscape, width for portrait)
RESOLUTION_SHORT_EDGES = {
    'SD': 480,
    'HD': 720,
    'FHD': 1080,
    '2K': 1440,
    '4K': 2160,
    '8K': 4320,
}


def _compute_resolution(aspect_key, preset_key):
    """Compute width x height from aspect ratio and resolution preset."""
    if aspect_key == 'CUSTOM' or preset_key == 'CUSTOM':
        return None
    if aspect_key not in ASPECT_RATIOS or preset_key not in RESOLUTION_SHORT_EDGES:
        return None
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_key]
    short_edge = RESOLUTION_SHORT_EDGES[preset_key]
    if w_ratio >= h_ratio:  # landscape or square
        height = short_edge
        width = int(height * w_ratio / h_ratio)
    else:  # portrait
        width = short_edge
        height = int(width * h_ratio / w_ratio)
    return (width, height)


def _get_default_output_path():
    """Return a safe absolute default output path."""
    if bpy.data.filepath:
        blend_dir = os.path.dirname(bpy.path.abspath(bpy.data.filepath))
        return os.path.join(blend_dir, "render") + os.sep
    if sys.platform in ('win32', 'darwin'):
        return str(Path.home() / "Desktop" / "BlenderRender") + os.sep
    else:
        return str(Path.home() / "BlenderRender") + os.sep


def _resolve_output_path(raw_path):
    """Resolve a path, handling empty and // notation safely."""
    if not raw_path or raw_path.isspace():
        return _get_default_output_path()
    if raw_path.startswith("//"):
        if bpy.data.filepath:
            return bpy.path.abspath(raw_path)
        else:
            # Blend not saved — fall back to Desktop
            return _get_default_output_path()
    return os.path.abspath(bpy.path.abspath(raw_path))


# ─────────────────────────────────────────────────────────────────────────────
#  Render Pass helpers — auto-create/remove File Output nodes
#  Handles both Blender 3.x/4.x (scene.node_tree) and 5.x+ (scene.compositing_node_group)
# ─────────────────────────────────────────────────────────────────────────────

# Mapping: view_layer pass property → Render Layers node output socket name → display name
PASS_MAP = [
    ('use_pass_combined', 'Image', 'Combined'),
    ('use_pass_z', 'Depth', 'Depth'),
    ('use_pass_mist', 'Mist', 'Mist'),
    ('use_pass_normal', 'Normal', 'Normal'),
    ('use_pass_position', 'Position', 'Position'),
    ('use_pass_diffuse_color', 'DiffCol', 'DiffuseColor'),
    ('use_pass_diffuse_direct', 'DiffDir', 'DiffuseDirect'),
    ('use_pass_diffuse_indirect', 'DiffInd', 'DiffuseIndirect'),
    ('use_pass_glossy_color', 'GlossCol', 'GlossyColor'),
    ('use_pass_glossy_direct', 'GlossDir', 'GlossyDirect'),
    ('use_pass_glossy_indirect', 'GlossInd', 'GlossyIndirect'),
    ('use_pass_transmission_color', 'TransCol', 'TransmissionColor'),
    ('use_pass_transmission_direct', 'TransDir', 'TransmissionDirect'),
    ('use_pass_transmission_indirect', 'TransInd', 'TransmissionIndirect'),
    ('use_pass_emit', 'Emit', 'Emit'),
    ('use_pass_environment', 'Env', 'Environment'),
    ('use_pass_shadow', 'Shadow', 'Shadow'),
    ('use_pass_ambient_occlusion', 'AO', 'AO'),
]

_BCR_FILE_OUTPUT_NODE_NAME = "BCR_FileOutput_AutoGenerated"
_BCR_RENDER_LAYERS_NODE_NAME = "BCR_RenderLayers_AutoGenerated"
_IS_BLENDER_5 = bpy.app.version[0] >= 5


def _get_compositor_tree(scene):
    """Get the compositor node tree, handling Blender 5.x API change."""
    if _IS_BLENDER_5:
        return getattr(scene, 'compositing_node_group', None)
    else:
        return getattr(scene, 'node_tree', None)


def _ensure_compositor(scene):
    """Enable compositor nodes and return the node tree."""
    scene.use_nodes = True
    tree = _get_compositor_tree(scene)
    return tree


def _create_pass_file_output(scene, base_path, auto_name_prefix):
    """Create pass nodes for batch render."""
    try:
        props = scene.bcr

        vl = None
        for v in scene.view_layers:
            if v.use:
                vl = v
                break
        if not vl:
            return False

        scene.render.use_compositing = True
        scene.use_nodes = True

        if _IS_BLENDER_5:
            tree = getattr(scene, 'compositing_node_group', None)
        else:
            tree = getattr(scene, 'node_tree', None)
        if not tree:
            return False

        # Remove any existing BCR nodes
        for node in list(tree.nodes):
            if node.name in (_BCR_FILE_OUTPUT_NODE_NAME, _BCR_RENDER_LAYERS_NODE_NAME):
                tree.nodes.remove(node)

        # Create RL node
        rl_node = tree.nodes.new('CompositorNodeRLayers')
        rl_node.name = _BCR_RENDER_LAYERS_NODE_NAME
        rl_node.label = "BCR Render Layers"
        rl_node.layer = vl.name
        rl_node.location = (0, -400)

        # Create FO node
        fo_node = tree.nodes.new('CompositorNodeOutputFile')
        fo_node.name = _BCR_FILE_OUTPUT_NODE_NAME
        fo_node.label = "BCR Auto Passes"
        fo_node.location = (400, -350)
        fo_node.width = 300

        if not base_path.endswith(('/', '\\')):
            base_path += os.sep
        if _IS_BLENDER_5:
            fo_node.directory = base_path
            fo_node.file_name = ""
        else:
            fo_node.base_path = base_path

        # Set format — must switch media_type first for non-EXR formats
        chosen_format = props.pass_format
        try:
            if chosen_format == 'OPEN_EXR_MULTILAYER':
                fo_node.format.media_type = 'MULTI_LAYER_IMAGE'
                fo_node.format.file_format = 'OPEN_EXR_MULTILAYER'
            else:
                fo_node.format.media_type = 'IMAGE'
                fo_node.format.file_format = chosen_format
        except Exception as e:
            print(f"[BCR] Could not set pass format: {e}")

        # Link with naming convention
        for out in rl_node.outputs:
            if not out.enabled:
                continue

            pass_name = out.name.replace(' ', '')
            slot_name = f"{auto_name_prefix}{pass_name}_"

            slot = 0
            found = False
            for src in fo_node.inputs:
                if src.identifier == out.identifier:
                    found = True
                    if _IS_BLENDER_5:
                        if slot < len(fo_node.file_output_items):
                            fo_node.file_output_items[slot].name = slot_name
                    else:
                        if slot < len(fo_node.file_slots):
                            fo_node.file_slots[slot].path = slot_name
                    tree.links.new(out, fo_node.inputs[slot])
                    break
                slot = slot + 1
            if not found:
                if _IS_BLENDER_5:
                    socketType = out.type
                    if socketType == 'VALUE':
                        socketType = 'FLOAT'
                    fo_node.file_output_items.new(socketType, slot_name)
                    tree.links.new(out, fo_node.inputs[-2])
                else:
                    fo_node.file_slots.new(slot_name)
                    tree.links.new(out, fo_node.inputs[-1])

        return True

    except Exception as e:
        print(f"[BCR] Error creating pass nodes: {e}")
        import traceback
        traceback.print_exc()
        return False
        return False


def _remove_pass_file_output(scene):
    """Remove BOTH the auto-generated Render Layers and File Output nodes."""
    try:
        tree = _get_compositor_tree(scene)
        if not tree:
            return
        for node in list(tree.nodes):
            if node.name in (_BCR_FILE_OUTPUT_NODE_NAME, _BCR_RENDER_LAYERS_NODE_NAME):
                tree.nodes.remove(node)
    except Exception:
        pass


def _unhide_all_excluded(props):
    """Restore viewport visibility for ALL setups' excluded objects."""
    for setup in props.setups:
        for exc_item in setup.exclude_objects:
            if exc_item.obj:
                exc_item.obj.hide_viewport = False


def _hide_excluded_for_setup(setup):
    """Hide excluded objects in the viewport for this setup."""
    for exc_item in setup.exclude_objects:
        if exc_item.obj:
            exc_item.obj.hide_viewport = True


class BCR_OT_RefreshPassNodes(Operator):
    bl_idname = "bcr.refresh_pass_nodes"
    bl_label = "Refresh Pass Nodes"
    bl_description = "Create/update File Output node in the compositor with all enabled render passes"
    bl_options = {'REGISTER', 'UNDO'}

    def GetNodeTree(self):
        """Get compositor node tree — same method as reference addon."""
        if bpy.app.version[0] >= 5:
            return bpy.context.scene.compositing_node_group
        else:
            return bpy.context.scene.node_tree

    def CreateRenderLayers(self, position):
        """Create a new Render Layers node — same as reference addon."""
        nodeTree = self.GetNodeTree()
        node = nodeTree.nodes.new('CompositorNodeRLayers')
        node.name = _BCR_RENDER_LAYERS_NODE_NAME
        node.label = "BCR Render Layers"
        node.location = position
        return node

    def CreateFileOutput(self, position, base_path):
        """Create a new File Output node."""
        nodeTree = self.GetNodeTree()
        node = nodeTree.nodes.new("CompositorNodeOutputFile")
        node.name = _BCR_FILE_OUTPUT_NODE_NAME
        node.label = "BCR Auto Passes"
        node.location = position
        node.width = 300

        # Set output path
        if not base_path.endswith(('/', '\\')):
            base_path += os.sep
        if bpy.app.version[0] >= 5:
            node.directory = base_path
            node.file_name = ""
        else:
            node.base_path = base_path

        return node

    def LinkRenderLayers(self, sourceNode, targetNode, name_prefix=""):
        """Link all outputs and name slots with our naming convention."""
        nodeTree = self.GetNodeTree()
        for out in sourceNode.outputs:
            if not out.enabled:
                continue

            # Build slot name: prefix + pass name (spaces removed)
            pass_name = out.name.replace(' ', '')
            slot_name = f"{name_prefix}{pass_name}_" if name_prefix else f"{out.identifier}"

            slot = 0
            found = False
            for src in targetNode.inputs:
                if (src.identifier == out.identifier):
                    found = True
                    # Rename existing slot
                    if bpy.app.version[0] >= 5:
                        if slot < len(targetNode.file_output_items):
                            targetNode.file_output_items[slot].name = slot_name
                    else:
                        if slot < len(targetNode.file_slots):
                            targetNode.file_slots[slot].path = slot_name
                    nodeTree.links.new(out, targetNode.inputs[slot])
                    break
                slot = slot + 1
            if not found:
                if bpy.app.version[0] >= 5:
                    socketType = out.type
                    if socketType == 'VALUE':
                        socketType = 'FLOAT'
                    targetNode.file_output_items.new(socketType, slot_name)
                    nodeTree.links.new(out, targetNode.inputs[-2])
                else:
                    targetNode.file_slots.new(slot_name)
                    nodeTree.links.new(out, targetNode.inputs[-1])

    def execute(self, context):
        scene = context.scene
        props = scene.bcr

        # Always remove old BCR nodes first
        _remove_pass_file_output(scene)

        if not props.save_passes:
            self.report({'INFO'}, "Save Passes is off — nodes removed")
            return {'FINISHED'}

        # Check compositor tree exists
        nodeTree = self.GetNodeTree()
        if nodeTree is None:
            self.report({'ERROR'}, "Compositor not initialized. Open the Compositor tab first.")
            return {'CANCELLED'}

        # Find the active/enabled view layer
        vl = None
        for v in scene.view_layers:
            if v.use:
                vl = v
                break
        if not vl:
            self.report({'WARNING'}, "No enabled view layer")
            return {'CANCELLED'}

        # Build output path and name prefix
        if not props.setups or props.active_index < 0 or props.active_index >= len(props.setups):
            self.report({'WARNING'}, "No active setup")
            return {'CANCELLED'}

        setup = props.setups[props.active_index]
        if not setup.camera:
            self.report({'WARNING'}, "No camera assigned to active setup")
            return {'CANCELLED'}

        base_path = _resolve_output_path(props.output_path)
        if props.use_setup_folder:
            base_path = os.path.join(base_path, setup.name)
        if props.use_date_folder:
            date_str = datetime.date.today().strftime("%Y-%m-%d")
            base_path = os.path.join(base_path, date_str)
        if props.use_viewlayer_folder:
            base_path = os.path.join(base_path, setup.view_layer_name)

        # Build name prefix (same as file naming)
        parts = []
        if props.include_setup_name:
            parts.append(setup.name)
        parts.append(setup.camera.name)
        if props.include_view_layer:
            parts.append(setup.view_layer_name)
        name_prefix = "_".join(parts) + "_"

        # Enable compositing
        scene.render.use_compositing = True
        scene.use_nodes = True

        # Find position below existing nodes
        min_y = 0
        for node in nodeTree.nodes:
            if node.location.y < min_y:
                min_y = node.location.y

        # Create nodes
        layersNode = self.CreateRenderLayers((0, min_y - 400))
        layersNode.layer = vl.name

        outputNode = self.CreateFileOutput((400, min_y - 350), base_path)

        # Set format on the File Output node
        chosen_format = props.pass_format
        try:
            if chosen_format == 'OPEN_EXR_MULTILAYER':
                # Multi-Layer EXR mode — all passes in one file
                outputNode.format.media_type = 'MULTI_LAYER_IMAGE'
                outputNode.format.file_format = 'OPEN_EXR_MULTILAYER'
            else:
                # Individual file mode — switch to Image first, then set format
                outputNode.format.media_type = 'IMAGE'
                outputNode.format.file_format = chosen_format
        except Exception as e:
            print(f"[BCR] Could not set pass format: {e}")

        # Link with our naming convention
        self.LinkRenderLayers(layersNode, outputNode, name_prefix)

        self.report({'INFO'}, f"Pass nodes created ({chosen_format}) — check the Compositor")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
#  Collections: a checklist per setup (only the hidden ones are stored)
# ─────────────────────────────────────────────────────────────────────────────

_BCR_HIDDEN_TAG = "_bcr_hidden"


def _scene_collections(scene):
    """``(collection, depth)`` for every collection of the scene, parents before children."""
    result = []

    def walk(parent, depth):
        for child in parent.children:
            result.append((child, depth))
            walk(child, depth + 1)

    walk(scene.collection, 0)
    return result


def _hidden_collections(setup):
    return {item.collection for item in setup.hidden_collections if item.collection}


def _find_layer_collection(layer_collection, collection):
    if layer_collection.collection == collection:
        return layer_collection
    for child in layer_collection.children:
        found = _find_layer_collection(child, collection)
        if found:
            return found
    return None


def _apply_collections_to_viewport(scene, view_layer, setup):
    """Hide the setup's hidden collections in the viewport (the eye in the Outliner).

    Collections hidden for another setup are shown again first; they are tagged so the ones hidden by hand
    are left alone.
    """
    hidden = _hidden_collections(setup) if setup else set()
    for collection, _depth in _scene_collections(scene):
        layer_collection = _find_layer_collection(view_layer.layer_collection, collection)
        if layer_collection is None:
            continue
        if collection in hidden:
            if not layer_collection.hide_viewport:
                layer_collection.hide_viewport = True
                collection[_BCR_HIDDEN_TAG] = True
        elif collection.get(_BCR_HIDDEN_TAG):
            layer_collection.hide_viewport = False
            del collection[_BCR_HIDDEN_TAG]


def _hide_collections_for_render(setup):
    """Disable the setup's hidden collections in renders. Returns what to restore."""
    backup = {}
    for collection in _hidden_collections(setup):
        backup[collection.name] = collection.hide_render
        collection.hide_render = True
    return backup


def _restore_collections(backup):
    for name, hide_render in backup.items():
        collection = bpy.data.collections.get(name)
        if collection:
            collection.hide_render = hide_render


# ─────────────────────────────────────────────────────────────────────────────
#  Light Mix per setup: a saved setup of the scene, or a JSON file
# ─────────────────────────────────────────────────────────────────────────────

def _light_mix_module():
    try:
        from cycles import light_mix
    except ImportError:
        return None
    return light_mix


def _light_mix_data(scene, setup):
    """The Light Mix setup dictionary for a render setup, ``None`` when it has none. Raises on errors."""
    if setup.light_mix_source == 'NONE':
        return None
    light_mix = _light_mix_module()
    if light_mix is None:
        raise RuntimeError("Light Mix is not available")
    if setup.light_mix_source == 'SETUP':
        saved = scene.cycles_light_mix.setups.get(setup.light_mix_setup)
        if saved is None:
            raise RuntimeError("no Light Mix setup \"{:s}\"".format(setup.light_mix_setup))
        return light_mix.setup_parse(saved.data)
    path = bpy.path.abspath(setup.light_mix_file)
    if not setup.light_mix_file or not os.path.isfile(path):
        raise RuntimeError("Light Mix file not found: {:s}".format(setup.light_mix_file or "(none)"))
    with open(path, encoding="utf-8") as fh:
        return light_mix.setup_parse(fh.read())


def _apply_light_mix(scene, view_layer, setup, backup=None):
    """Apply the setup's Light Mix to the view layer. With ``backup``, keeps the mix it had before."""
    data = _light_mix_data(scene, setup)
    if data is None:
        return False
    light_mix = _light_mix_module()
    if backup is not None and view_layer.name not in backup:
        backup[view_layer.name] = (view_layer.use_light_mix, light_mix.setup_from_view_layer(view_layer))
    light_mix.setup_apply(view_layer, data)
    return True


def _restore_light_mix(scene, backup):
    light_mix = _light_mix_module()
    if light_mix is None:
        return
    for name, (use_light_mix, data) in backup.items():
        view_layer = scene.view_layers.get(name)
        if view_layer:
            light_mix.setup_apply(view_layer, data)
            view_layer.use_light_mix = use_light_mix
    backup.clear()


def _apply_setup_to_viewport(setup):
    """Push a setup's camera, view layer, and resolution to the viewport."""
    ctx = bpy.context
    scene = ctx.scene
    if ctx.window is None or ctx.screen is None:
        return

    if setup.camera:
        scene.camera = setup.camera
        for area in ctx.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                break

    vl_name = setup.view_layer_name
    if vl_name and vl_name in scene.view_layers:
        ctx.window.view_layer = scene.view_layers[vl_name]

    # Apply world if set
    if setup.world:
        scene.world = setup.world

    scene.render.resolution_x = setup.res_x
    scene.render.resolution_y = setup.res_y
    scene.render.resolution_percentage = 100

    # Show all excluded objects first, then hide this setup's excluded ones
    _unhide_all_excluded(scene.bcr)
    _hide_excluded_for_setup(setup)

    view_layer = ctx.window.view_layer
    _apply_collections_to_viewport(scene, view_layer, setup)
    try:
        _apply_light_mix(scene, view_layer, setup)
    except (RuntimeError, ValueError, OSError) as ex:
        print("[BCR] Light Mix of \"{:s}\": {:s}".format(setup.name, str(ex)))

    for area in ctx.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _get_view_layer_items(self, context):
    items = []
    if context and context.scene:
        for vl in context.scene.view_layers:
            items.append((vl.name, vl.name, f"View Layer: {vl.name}", 'RENDERLAYERS', len(items)))
    return items if items else [('NONE', "No View Layers", "", 'ERROR', 0)]


# ─────────────────────────────────────────────────────────────────────────────
#  Property update callbacks – real-time viewport updates
# ─────────────────────────────────────────────────────────────────────────────

def _on_aspect_update(self, context):
    """When aspect ratio changes, recompute resolution from preset."""
    result = _compute_resolution(self.aspect_ratio, self.resolution_preset)
    if result:
        self.res_x, self.res_y = result
    _push_resolution_to_scene(self)


def _on_resolution_preset_update(self, context):
    """When resolution preset changes, recompute resolution from aspect."""
    result = _compute_resolution(self.aspect_ratio, self.resolution_preset)
    if result:
        self.res_x, self.res_y = result
    _push_resolution_to_scene(self)


def _on_resolution_update(self, context):
    """When res_x/res_y manually changes, push to scene."""
    _push_resolution_to_scene(self)


def _push_resolution_to_scene(setup):
    """Push resolution to the scene render settings."""
    try:
        scene = bpy.context.scene
        scene.render.resolution_x = setup.res_x
        scene.render.resolution_y = setup.res_y
        scene.render.resolution_percentage = 100
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    except Exception:
        pass


def _on_view_layer_update(self, context):
    try:
        vl_name = self.view_layer_name
        scene = bpy.context.scene
        if vl_name and vl_name in scene.view_layers:
            bpy.context.window.view_layer = scene.view_layers[vl_name]
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _on_camera_update(self, context):
    try:
        if self.camera:
            scene = bpy.context.scene
            scene.camera = self.camera
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
                    area.tag_redraw()
                    break
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Data: one "Render Setup" item
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Data: excluded object reference
# ─────────────────────────────────────────────────────────────────────────────

class BCR_ExcludeObject(PropertyGroup):
    """An object to exclude from a render setup (ray visibility turned off)."""
    obj: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Object to exclude from this render setup",
    )


class BCR_CollectionRef(PropertyGroup):
    """A collection hidden in a render setup."""
    collection: PointerProperty(name="Collection", type=bpy.types.Collection)


def _on_light_mix_update(self, context):
    """Show the chosen Light Mix right away when it belongs to the active setup."""
    try:
        props = context.scene.bcr
        if props.setups and props.setups[props.active_index] == self:
            _apply_light_mix(context.scene, context.view_layer, self)
    except (RuntimeError, ValueError, OSError, IndexError, AttributeError):
        pass


def _on_world_update(self, context):
    """When world changes on the active setup, apply it to the scene."""
    try:
        if self.world:
            bpy.context.scene.world = self.world
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Data: one "Render Setup" item
# ─────────────────────────────────────────────────────────────────────────────

class BCR_RenderSetup(PropertyGroup):

    name: StringProperty(name="Name", default="Setup")

    camera: PointerProperty(
        name="Camera", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'CAMERA',
        update=_on_camera_update,
    )

    view_layer_name: EnumProperty(
        name="View Layer", items=_get_view_layer_items,
        update=_on_view_layer_update,
    )

    world: PointerProperty(
        name="World",
        type=bpy.types.World,
        description="World / environment to use for this setup (leave empty to use scene default)",
        update=_on_world_update,
    )

    render_mode: EnumProperty(
        name="Mode",
        items=[
            ('FRAME', "Single Frame", "Render a specific frame"),
            ('ANIMATION', "Animation", "Render a frame range"),
        ],
        default='FRAME',
    )

    frame_single: IntProperty(
        name="Frame", default=0, min=0,
        description="Frame number to render",
    )

    frame_start: IntProperty(
        name="Start", default=0, min=0,
        description="First frame of the animation range",
    )

    frame_end: IntProperty(
        name="End", default=100, min=0,
        description="Last frame of the animation range",
    )

    is_hidden: BoolProperty(name="Hidden", default=False)

    aspect_ratio: EnumProperty(
        name="Aspect Ratio", items=ASPECT_PRESETS,
        default='16:9', update=_on_aspect_update,
    )

    res_x: IntProperty(
        name="X", default=1920, min=1, max=65536,
        update=_on_resolution_update,
    )

    res_y: IntProperty(
        name="Y", default=1080, min=1, max=65536,
        update=_on_resolution_update,
    )

    resolution_preset: EnumProperty(
        name="Resolution",
        items=RESOLUTION_PRESETS,
        default='FHD',
        update=_on_resolution_preset_update,
    )

    status: StringProperty(name="Status", default="")

    # ── Excluded objects ──
    exclude_objects: CollectionProperty(type=BCR_ExcludeObject)
    exclude_active_index: IntProperty(default=0)

    # ── Collections: unchecked ones are hidden in this setup ──
    hidden_collections: CollectionProperty(type=BCR_CollectionRef)

    # ── Light Mix ──
    light_mix_source: EnumProperty(
        name="Light Mix",
        items=[
            ('NONE', "None", "Keep the Light Mix of the view layer as it is"),
            ('SETUP', "Saved Setup", "A Light Mix setup saved in this file"),
            ('FILE', "JSON File", "A Light Mix setup exported to a .json file"),
        ],
        default='NONE',
        description="Light Mix to apply to the view layer for this setup",
        update=_on_light_mix_update,
    )
    light_mix_setup: StringProperty(
        name="Light Mix Setup",
        description="Light Mix setup of this file (View Layer Properties > Light Mix > Setups)",
        update=_on_light_mix_update,
    )
    light_mix_file: StringProperty(
        name="Light Mix File",
        description="Light Mix setup exported as .json (Light Mix > Setups > Export)",
        subtype='FILE_PATH',
        update=_on_light_mix_update,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Scene-level properties
# ─────────────────────────────────────────────────────────────────────────────

def _on_pass_format_update(self, context):
    """When pass format changes, also update the scene's Output Properties format."""
    try:
        fmt = self.pass_format
        img = context.scene.render.image_settings
        if fmt == 'OPEN_EXR_MULTILAYER':
            img.file_format = 'OPEN_EXR_MULTILAYER'
        else:
            img.file_format = fmt
    except Exception:
        pass


def _on_active_index_update(self, context):
    if self.setups and 0 <= self.active_index < len(self.setups):
        _apply_setup_to_viewport(self.setups[self.active_index])


class BCR_SceneProperties(PropertyGroup):
    setups: CollectionProperty(type=BCR_RenderSetup)
    active_index: IntProperty(default=0, update=_on_active_index_update)

    output_path: StringProperty(
        name="Output Path",
        default="",
        subtype='DIR_PATH',
    )

    use_date_folder: BoolProperty(
        name="Date Folder", default=True,
    )

    use_setup_folder: BoolProperty(
        name="Setup Folder",
        default=False,
        description="Create a sub-folder with the setup name (before date folder)",
    )

    use_viewlayer_folder: BoolProperty(
        name="View Layer Folder",
        default=False,
        description="Create a sub-folder with the view layer name (after date folder)",
    )

    # ── File naming ──
    include_setup_name: BoolProperty(
        name="Include Setup Name",
        default=False,
        description="Include the setup name in the file name",
    )

    include_view_layer: BoolProperty(
        name="Include View Layer",
        default=False,
        description="Include the view layer name in the file name",
    )

    save_passes: BoolProperty(
        name="Save Passes",
        default=False,
        description="Save each enabled render pass as a separate file using File Output nodes",
    )

    pass_format: EnumProperty(
        name="Pass Format",
        items=[
            ('OPEN_EXR_MULTILAYER', "MultiLayer EXR", "All passes in one EXR file with layers"),
            ('OPEN_EXR', "OpenEXR", "Individual EXR file per pass"),
            ('PNG', "PNG", "Individual PNG file per pass"),
            ('JPEG', "JPEG", "Individual JPEG file per pass"),
            ('TIFF', "TIFF", "Individual TIFF file per pass"),
            ('BMP', "BMP", "Individual BMP file per pass"),
        ],
        default='OPEN_EXR_MULTILAYER',
        description="File format for render pass output and main render output",
        update=lambda self, ctx: _on_pass_format_update(self, ctx),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  UIList
# ─────────────────────────────────────────────────────────────────────────────

class BCR_UL_SetupList(UIList):
    bl_idname = "BCR_UL_SetupList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        setup = item
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)

            eye_icon = 'HIDE_OFF' if not setup.is_hidden else 'HIDE_ON'
            row.prop(setup, "is_hidden", text="", icon=eye_icon, emboss=False, invert_checkbox=True)
            row.prop(setup, "name", text="", emboss=False)

            if setup.camera:
                row.label(text=setup.camera.name, icon='CAMERA_DATA')
            else:
                row.label(text="No Camera", icon='ERROR')

            row.label(text=setup.view_layer_name, icon='RENDERLAYERS')
            row.label(text=setup.world.name if setup.world else "—", icon='WORLD')
            aspect_label = setup.aspect_ratio if setup.aspect_ratio != 'CUSTOM' else ""
            row.label(text=f"{setup.res_x}x{setup.res_y}  {aspect_label}")

            if setup.light_mix_source != 'NONE':
                row.label(text="", icon='OUTLINER_OB_LIGHT')
            if setup.hidden_collections:
                row.label(text="", icon='OUTLINER_COLLECTION')

            mode_text = "Frame" if setup.render_mode == 'FRAME' else "Anim"
            mode_icon = 'IMAGE_DATA' if setup.render_mode == 'FRAME' else 'RENDER_ANIMATION'
            row.label(text=mode_text, icon=mode_icon)

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=setup.name, icon='CAMERA_DATA')


# ─────────────────────────────────────────────────────────────────────────────
#  Operators
# ─────────────────────────────────────────────────────────────────────────────

class BCR_OT_AddSetup(Operator):
    bl_idname = "bcr.add_setup"
    bl_label = "Add Render Setup"

    def execute(self, context):
        props = context.scene.bcr
        setup = props.setups.add()
        setup.name = f"Setup {len(props.setups)}"
        if context.scene.camera:
            setup.camera = context.scene.camera
        if context.scene.world:
            setup.world = context.scene.world
        setup.res_x = context.scene.render.resolution_x
        setup.res_y = context.scene.render.resolution_y
        props.active_index = len(props.setups) - 1
        return {'FINISHED'}


class BCR_OT_RemoveSetup(Operator):
    bl_idname = "bcr.remove_setup"
    bl_label = "Remove Render Setup"

    def execute(self, context):
        props = context.scene.bcr
        if props.setups and 0 <= props.active_index < len(props.setups):
            props.setups.remove(props.active_index)
            props.active_index = max(0, props.active_index - 1)
        return {'FINISHED'}


class BCR_OT_DuplicateSetup(Operator):
    bl_idname = "bcr.duplicate_setup"
    bl_label = "Duplicate Setup"

    def execute(self, context):
        props = context.scene.bcr
        if not props.setups or props.active_index < 0:
            return {'CANCELLED'}
        index = props.active_index
        dup = props.setups.add()
        # Adding can move the items in memory: get the source only now.
        src = props.setups[index]
        for attr in ('name', 'camera', 'view_layer_name', 'world', 'render_mode',
                     'frame_single', 'frame_start', 'frame_end',
                     'is_hidden', 'aspect_ratio', 'res_x', 'res_y',
                     'resolution_preset', 'light_mix_source', 'light_mix_setup', 'light_mix_file'):
            setattr(dup, attr, getattr(src, attr))
        dup.name = src.name + " (copy)"
        dup.status = ""
        # Copy exclude list
        for src_item in src.exclude_objects:
            dup_item = dup.exclude_objects.add()
            dup_item.obj = src_item.obj
        for src_item in src.hidden_collections:
            dup.hidden_collections.add().collection = src_item.collection
        props.active_index = len(props.setups) - 1
        return {'FINISHED'}


class BCR_OT_MoveSetup(Operator):
    bl_idname = "bcr.move_setup"
    bl_label = "Move Setup"
    direction: EnumProperty(items=[('UP', "Up", ""), ('DOWN', "Down", "")])

    def execute(self, context):
        props = context.scene.bcr
        idx = props.active_index
        if self.direction == 'UP' and idx > 0:
            props.setups.move(idx, idx - 1)
            props.active_index -= 1
        elif self.direction == 'DOWN' and idx < len(props.setups) - 1:
            props.setups.move(idx, idx + 1)
            props.active_index += 1
        return {'FINISHED'}


class BCR_OT_NavigateToCamera(Operator):
    bl_idname = "bcr.navigate_to_camera"
    bl_label = "Look Through Camera"

    def execute(self, context):
        props = context.scene.bcr
        if props.setups and 0 <= props.active_index < len(props.setups):
            _apply_setup_to_viewport(props.setups[props.active_index])
        return {'FINISHED'}


class BCR_OT_AddExcludeObject(Operator):
    bl_idname = "bcr.add_exclude_object"
    bl_label = "Exclude Selected Objects"
    bl_description = "Add all selected objects to this setup's exclude list"

    def execute(self, context):
        props = context.scene.bcr
        if not props.setups or props.active_index < 0:
            return {'CANCELLED'}

        setup = props.setups[props.active_index]
        selected = context.selected_objects

        if not selected:
            self.report({'WARNING'}, "No objects selected")
            return {'CANCELLED'}

        # Get existing excluded object references
        existing = {item.obj for item in setup.exclude_objects if item.obj}

        added = 0
        for obj in selected:
            # Don't exclude cameras
            if obj.type == 'CAMERA':
                continue
            if obj not in existing:
                item = setup.exclude_objects.add()
                item.obj = obj
                added += 1

        if added:
            setup.exclude_active_index = len(setup.exclude_objects) - 1
            _hide_excluded_for_setup(setup)
            self.report({'INFO'}, f"Excluded {added} object(s)")
        else:
            self.report({'INFO'}, "Selected objects already excluded")

        return {'FINISHED'}


class BCR_OT_RemoveExcludeObject(Operator):
    bl_idname = "bcr.remove_exclude_object"
    bl_label = "Remove from Exclude List"
    bl_description = "Remove the selected entry from the exclude list"

    def execute(self, context):
        props = context.scene.bcr
        if not props.setups or props.active_index < 0:
            return {'CANCELLED'}

        setup = props.setups[props.active_index]
        if setup.exclude_objects and 0 <= setup.exclude_active_index < len(setup.exclude_objects):
            # Unhide the object being removed
            removed_obj = setup.exclude_objects[setup.exclude_active_index].obj
            if removed_obj:
                removed_obj.hide_viewport = False
            setup.exclude_objects.remove(setup.exclude_active_index)
            setup.exclude_active_index = max(0, setup.exclude_active_index - 1)

        return {'FINISHED'}


class BCR_OT_ClearExcludeObjects(Operator):
    bl_idname = "bcr.clear_exclude_objects"
    bl_label = "Clear All Excluded Objects"
    bl_description = "Remove all objects from this setup's exclude list"

    def execute(self, context):
        props = context.scene.bcr
        if not props.setups or props.active_index < 0:
            return {'CANCELLED'}

        setup = props.setups[props.active_index]
        # Unhide all excluded objects before clearing
        for exc_item in setup.exclude_objects:
            if exc_item.obj:
                exc_item.obj.hide_viewport = False
        setup.exclude_objects.clear()
        setup.exclude_active_index = 0
        return {'FINISHED'}


def _active_setup(context):
    props = context.scene.bcr
    if props.setups and 0 <= props.active_index < len(props.setups):
        return props.setups[props.active_index]
    return None


class BCR_OT_ToggleCollection(Operator):
    bl_idname = "bcr.toggle_collection"
    bl_label = "Toggle Collection"
    bl_description = "Show or hide this collection in the active setup (Shift: only this one)"
    bl_options = {'REGISTER', 'UNDO'}

    collection_name: StringProperty()
    only: BoolProperty(options={'SKIP_SAVE'})

    def invoke(self, context, event):
        self.only = event.shift
        return self.execute(context)

    def execute(self, context):
        setup = _active_setup(context)
        collection = bpy.data.collections.get(self.collection_name)
        if setup is None or collection is None:
            return {'CANCELLED'}
        if self.only:
            setup.hidden_collections.clear()
            for other, _depth in _scene_collections(context.scene):
                if other != collection and other not in _collection_parents(context.scene, collection):
                    setup.hidden_collections.add().collection = other
        else:
            for i, item in enumerate(setup.hidden_collections):
                if item.collection == collection:
                    setup.hidden_collections.remove(i)
                    break
            else:
                setup.hidden_collections.add().collection = collection
        _apply_collections_to_viewport(context.scene, context.view_layer, setup)
        return {'FINISHED'}


def _collection_parents(scene, collection):
    parents = set()

    def walk(parent, chain):
        for child in parent.children:
            if child == collection:
                parents.update(chain)
            walk(child, chain + [child])

    walk(scene.collection, [])
    return parents


class BCR_OT_ShowAllCollections(Operator):
    bl_idname = "bcr.show_all_collections"
    bl_label = "Show All Collections"
    bl_description = "Check every collection in the active setup"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        setup = _active_setup(context)
        if setup is None:
            return {'CANCELLED'}
        setup.hidden_collections.clear()
        _apply_collections_to_viewport(context.scene, context.view_layer, setup)
        return {'FINISHED'}


class BCR_UL_ExcludeList(UIList):
    bl_idname = "BCR_UL_ExcludeList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if item.obj:
            obj_icon = 'OUTLINER_OB_' + item.obj.type if item.obj.type in {
                'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'ARMATURE',
                'LATTICE', 'EMPTY', 'LIGHT', 'LIGHT_PROBE', 'SPEAKER',
                'VOLUME', 'GREASEPENCIL'
            } else 'OBJECT_DATA'
            layout.label(text=item.obj.name, icon=obj_icon)
        else:
            layout.label(text="(missing)", icon='ERROR')


# ── Batch Render (modal with render window) ──────────────────────────────────

# Shared state for the modal batch render
_bcr_batch = {
    'render_done': False,
    'render_cancelled': False,
}


def _on_render_complete(scene):
    _bcr_batch['render_done'] = True


def _on_render_cancel(scene):
    _bcr_batch['render_done'] = True
    _bcr_batch['render_cancelled'] = True


class BCR_OT_BatchRender(Operator):
    bl_idname = "bcr.batch_render"
    bl_label = "Batch Render All Visible"
    bl_description = "Render every unhidden setup one by one"

    _timer = None
    _queue = []
    _current = -1
    _phase = 'IDLE'      # IDLE → SETUP → WAIT → NEXT → DONE
    _originals = {}
    _ray_vis_backup = {}
    _errors = []
    _done_count = 0
    _orig_shading = {}
    _collection_backup = {}
    _light_mix_backup = {}

    def _save_originals(self, context):
        scene = context.scene
        self._originals = {
            'camera': scene.camera,
            'world': scene.world,
            'filepath': scene.render.filepath,
            'use_file_ext': scene.render.use_file_extension,
            'res_x': scene.render.resolution_x,
            'res_y': scene.render.resolution_y,
            'res_pct': scene.render.resolution_percentage,
            'frame_current': scene.frame_current,
            'frame_start': scene.frame_start,
            'frame_end': scene.frame_end,
            'vl_states': {vl.name: vl.use for vl in scene.view_layers},
            'use_nodes': getattr(scene, 'use_nodes', False),
            'use_compositing': scene.render.use_compositing,
        }

    def _restore_originals(self, context):
        scene = context.scene
        o = self._originals
        scene.camera = o['camera']
        scene.world = o['world']
        scene.render.filepath = o['filepath']
        scene.render.use_file_extension = o['use_file_ext']
        scene.render.resolution_x = o['res_x']
        scene.render.resolution_y = o['res_y']
        scene.render.resolution_percentage = o['res_pct']
        scene.frame_current = o['frame_current']
        scene.frame_start = o['frame_start']
        scene.frame_end = o['frame_end']
        for vl in scene.view_layers:
            if vl.name in o['vl_states']:
                vl.use = o['vl_states'][vl.name]
        _remove_pass_file_output(scene)
        _restore_light_mix(scene, self._light_mix_backup)
        try:
            scene.use_nodes = o['use_nodes']
            scene.render.use_compositing = o['use_compositing']
        except Exception:
            pass

    def _switch_viewport_shading(self, context, mode):
        """Switch all 3D viewports to a shading mode and save originals."""
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if mode == 'SAVE':
                            self._orig_shading[id(space)] = space.shading.type
                        elif mode == 'SOLID':
                            space.shading.type = 'SOLID'
                        elif mode == 'RESTORE':
                            orig = self._orig_shading.get(id(space))
                            if orig:
                                space.shading.type = orig
                area.tag_redraw()

    def _restore_excluded(self, setup):
        """Restore ray visibility for excluded objects and collections after a render."""
        _restore_collections(self._collection_backup)
        self._collection_backup = {}
        for exc_item in setup.exclude_objects:
            obj = exc_item.obj
            if not obj or obj.name not in self._ray_vis_backup:
                continue
            backup = self._ray_vis_backup[obj.name]
            obj.hide_viewport = backup['hide_viewport']
            obj.hide_render = backup['hide_render']
            if hasattr(obj, 'cycles_visibility') and 'camera' in backup:
                cv = obj.cycles_visibility
                cv.camera = backup['camera']
                cv.diffuse = backup['diffuse']
                cv.glossy = backup['glossy']
                cv.transmission = backup['transmission']
                cv.scatter = backup['scatter']
                cv.shadow = backup['shadow']
            elif 'visible_camera' in backup:
                obj.visible_camera = backup['visible_camera']
                obj.visible_diffuse = backup['visible_diffuse']
                obj.visible_glossy = backup['visible_glossy']
                obj.visible_transmission = backup['visible_transmission']
                obj.visible_volume_scatter = backup['visible_volume_scatter']
                obj.visible_shadow = backup['visible_shadow']

    def _configure_setup(self, context, setup):
        """Apply one setup's settings to the scene. Called BETWEEN renders only."""
        scene = context.scene
        props = scene.bcr

        scene.camera = setup.camera

        # Apply world if set
        if setup.world:
            scene.world = setup.world

        for vl in scene.view_layers:
            vl.use = (vl.name == setup.view_layer_name)

        scene.render.resolution_x = setup.res_x
        scene.render.resolution_y = setup.res_y
        scene.render.resolution_percentage = 100

        # Disable visibility for excluded objects
        self._ray_vis_backup = {}
        for exc_item in setup.exclude_objects:
            obj = exc_item.obj
            if obj and hasattr(obj, 'cycles_visibility'):
                cv = obj.cycles_visibility
                self._ray_vis_backup[obj.name] = {
                    'hide_viewport': obj.hide_viewport,
                    'hide_render': obj.hide_render,
                    'camera': cv.camera, 'diffuse': cv.diffuse,
                    'glossy': cv.glossy, 'transmission': cv.transmission,
                    'scatter': cv.scatter, 'shadow': cv.shadow,
                }
                obj.hide_viewport = True
                obj.hide_render = True
                cv.camera = cv.diffuse = cv.glossy = False
                cv.transmission = cv.scatter = cv.shadow = False
            elif obj:
                self._ray_vis_backup[obj.name] = {
                    'hide_viewport': obj.hide_viewport,
                    'hide_render': obj.hide_render,
                    'visible_camera': obj.visible_camera,
                    'visible_diffuse': obj.visible_diffuse,
                    'visible_glossy': obj.visible_glossy,
                    'visible_transmission': obj.visible_transmission,
                    'visible_volume_scatter': obj.visible_volume_scatter,
                    'visible_shadow': obj.visible_shadow,
                }
                obj.hide_viewport = True
                obj.hide_render = True
                obj.visible_camera = obj.visible_diffuse = obj.visible_glossy = False
                obj.visible_transmission = obj.visible_volume_scatter = obj.visible_shadow = False

        # Collections unchecked in this setup, and its Light Mix
        self._collection_backup = _hide_collections_for_render(setup)
        view_layer = scene.view_layers[setup.view_layer_name]
        if not _apply_light_mix(scene, view_layer, setup, self._light_mix_backup):
            # No Light Mix of its own: the view layer's mix, not the one of the setup before.
            _restore_light_mix(scene, self._light_mix_backup)

        # Build output path: base / [setup] / [date] / [view_layer] /
        base_path = _resolve_output_path(props.output_path)
        if props.use_setup_folder:
            base_path = os.path.join(base_path, setup.name)
        if props.use_date_folder:
            date_str = datetime.date.today().strftime("%Y-%m-%d")
            base_path = os.path.join(base_path, date_str)
        if props.use_viewlayer_folder:
            base_path = os.path.join(base_path, setup.view_layer_name)

        os.makedirs(base_path, exist_ok=True)

        parts = []
        if props.include_setup_name:
            parts.append(setup.name)
        parts.append(setup.camera.name)
        if props.include_view_layer:
            parts.append(setup.view_layer_name)
        auto_name = "_".join(parts) + "_"
        scene.render.filepath = os.path.join(base_path, auto_name)
        scene.render.use_file_extension = True

        # ── Create File Output node for passes if enabled ──
        _remove_pass_file_output(scene)  # Clean any leftover
        if props.save_passes:
            _create_pass_file_output(scene, base_path, auto_name)

        if setup.render_mode == 'FRAME':
            scene.frame_set(setup.frame_single)
        else:
            scene.frame_start = setup.frame_start
            scene.frame_end = setup.frame_end

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        props = context.scene.bcr

        # ── SETUP: configure the next setup (no render is running) ──
        if self._phase == 'SETUP':
            self._current += 1

            # Find next valid setup
            while self._current < len(self._queue):
                setup_idx = self._queue[self._current]
                setup = props.setups[setup_idx]

                if not setup.camera:
                    setup.status = "Skip: no camera"
                    self._errors.append(f"{setup.name}: no camera")
                    self._current += 1
                    continue
                if setup.view_layer_name not in context.scene.view_layers:
                    setup.status = "Skip: bad layer"
                    self._errors.append(f"{setup.name}: layer missing")
                    self._current += 1
                    continue
                break
            else:
                self._phase = 'DONE'
                return {'PASS_THROUGH'}

            setup = props.setups[self._queue[self._current]]

            try:
                self._configure_setup(context, setup)
            except Exception as e:
                setup.status = f"Error: {str(e)[:25]}"
                self._errors.append(f"{setup.name}: {e}")
                self._restore_excluded(setup)
                self._phase = 'SETUP'
                return {'PASS_THROUGH'}

            total = len(self._queue)
            num = self._current + 1
            self.report({'INFO'}, f"Rendering {num}/{total}: {setup.name}")
            print(f"[BCR] Rendering {num}/{total}: {setup.name} | "
                  f"Camera={setup.camera.name} | Layer={setup.view_layer_name}")

            # Reset flags and launch render with visible window
            _bcr_batch['render_done'] = False
            _bcr_batch['render_cancelled'] = False

            try:
                if setup.render_mode == 'ANIMATION':
                    bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True)
                else:
                    bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)
            except Exception as e:
                setup.status = "Error"
                self._errors.append(f"{setup.name}: {e}")
                self._restore_excluded(setup)
                self._phase = 'SETUP'
                return {'PASS_THROUGH'}

            self._phase = 'WAIT'
            return {'PASS_THROUGH'}

        # ── WAIT: do nothing until the render handler signals completion ──
        elif self._phase == 'WAIT':
            if not _bcr_batch['render_done']:
                return {'PASS_THROUGH'}

            # Render finished — restore excluded objects, remove pass node, mark status
            setup = props.setups[self._queue[self._current]]
            self._restore_excluded(setup)
            _remove_pass_file_output(context.scene)

            if _bcr_batch['render_cancelled']:
                setup.status = "Cancelled"
                self._errors.append(f"{setup.name}: cancelled by user")
            else:
                setup.status = "Done"
                self._done_count += 1

            self._phase = 'SETUP'
            return {'PASS_THROUGH'}

        # ── DONE: clean up ──
        elif self._phase == 'DONE':
            # Remove handlers
            if _on_render_complete in bpy.app.handlers.render_complete:
                bpy.app.handlers.render_complete.remove(_on_render_complete)
            if _on_render_cancel in bpy.app.handlers.render_cancel:
                bpy.app.handlers.render_cancel.remove(_on_render_cancel)

            context.window_manager.event_timer_remove(self._timer)

            # Restore all original state
            self._restore_originals(context)

            # Restore viewport shading
            self._switch_viewport_shading(context, 'RESTORE')

            # Force depsgraph update
            try:
                depsgraph = context.evaluated_depsgraph_get()
                depsgraph.update()
            except Exception:
                pass

            # Reapply viewport exclusions for active setup
            if props.setups and 0 <= props.active_index < len(props.setups):
                _unhide_all_excluded(props)
                _hide_excluded_for_setup(props.setups[props.active_index])

            if self._errors:
                self.report({'WARNING'}, f"Batch done: {self._done_count} rendered, {len(self._errors)} error(s).")
            else:
                self.report({'INFO'}, f"Batch render complete — {self._done_count} setup(s) rendered.")

            for area in context.screen.areas:
                area.tag_redraw()

            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _valid(self, context, setup):
        if not setup.camera:
            setup.status = "Skip: no camera"
            self._errors.append(f"{setup.name}: no camera")
            return False
        if setup.view_layer_name not in context.scene.view_layers:
            setup.status = "Skip: bad layer"
            self._errors.append(f"{setup.name}: layer missing")
            return False
        return True

    def execute(self, context):
        """Render every visible setup, waiting for each render (used without a window, e.g. from scripts)."""
        props = context.scene.bcr
        if not props.output_path or (props.output_path.startswith("//") and not bpy.data.filepath):
            props.output_path = _get_default_output_path()
        queue = [setup for setup in props.setups if not setup.is_hidden]
        if not queue:
            self.report({'WARNING'}, "No visible setups to render")
            return {'CANCELLED'}
        self._save_originals(context)
        self._errors = []
        self._done_count = 0
        self._collection_backup = {}
        self._light_mix_backup = {}
        try:
            for setup in queue:
                setup.status = ""
                if not self._valid(context, setup):
                    continue
                try:
                    self._configure_setup(context, setup)
                    print(f"[BCR] Rendering {setup.name} | Camera={setup.camera.name} | "
                          f"Layer={setup.view_layer_name}")
                    bpy.ops.render.render(animation=(setup.render_mode == 'ANIMATION'), write_still=True)
                    setup.status = "Done"
                    self._done_count += 1
                except Exception as ex:
                    setup.status = f"Error: {str(ex)[:25]}"
                    self._errors.append(f"{setup.name}: {ex}")
                finally:
                    self._restore_excluded(setup)
                    _remove_pass_file_output(context.scene)
        finally:
            self._restore_originals(context)
        if self._errors:
            self.report({'WARNING'}, "Batch done: {:d} rendered, {:d} error(s): {:s}".format(
                self._done_count, len(self._errors), "; ".join(self._errors)))
        else:
            self.report({'INFO'}, f"Batch render complete — {self._done_count} setup(s) rendered.")
        return {'FINISHED'}

    def invoke(self, context, event):
        props = context.scene.bcr
        scene = context.scene

        # Auto-fix empty output path
        if not props.output_path or (props.output_path.startswith("//") and not bpy.data.filepath):
            props.output_path = _get_default_output_path()

        # Build queue
        self._queue = [i for i, s in enumerate(props.setups) if not s.is_hidden]
        if not self._queue:
            self.report({'WARNING'}, "No visible setups to render")
            return {'CANCELLED'}

        # Validate output path
        try:
            test_path = _resolve_output_path(props.output_path)
            os.makedirs(test_path, exist_ok=True)
        except PermissionError:
            self.report({'ERROR'},
                        f"Cannot write to '{test_path}'. "
                        "Save your .blend file or pick a writable folder.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Output path error: {e}")
            return {'CANCELLED'}

        # Clear statuses
        for i in self._queue:
            props.setups[i].status = ""

        # Save state
        self._save_originals(context)
        self._current = -1
        self._errors = []
        self._done_count = 0
        self._phase = 'SETUP'
        self._orig_shading = {}
        self._collection_backup = {}
        self._light_mix_backup = {}

        # Switch viewport to SOLID to save resources
        self._switch_viewport_shading(context, 'SAVE')
        self._switch_viewport_shading(context, 'SOLID')

        # Register render handlers
        if _on_render_complete not in bpy.app.handlers.render_complete:
            bpy.app.handlers.render_complete.append(_on_render_complete)
        if _on_render_cancel not in bpy.app.handlers.render_cancel:
            bpy.app.handlers.render_cancel.append(_on_render_cancel)

        # Start timer (checks every 0.5s — only reads a flag, never modifies scene while rendering)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)

        self.report({'INFO'}, f"Starting batch render: {len(self._queue)} setup(s)…")
        return {'RUNNING_MODAL'}


class BCR_OT_ClearStatuses(Operator):
    bl_idname = "bcr.clear_statuses"
    bl_label = "Clear Statuses"

    def execute(self, context):
        for setup in context.scene.bcr.setups:
            setup.status = ""
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
#  Popup Panel
# ─────────────────────────────────────────────────────────────────────────────

class BCR_PT_SidebarPanel(Panel):
    bl_label = "Batch Camera Renderer"
    bl_idname = "BCR_PT_SidebarPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Batch Render"

    def draw(self, context):
        layout = self.layout
        props = context.scene.bcr

        # ── List ──
        # Column headers
        header = layout.row(align=True)
        header.scale_y = 0.6
        header.label(text="")  # eye icon spacer
        header.label(text="Setup")
        header.label(text="Camera")
        header.label(text="View Layer")
        header.label(text="World")
        header.label(text="Resolution")
        header.label(text="Mode")

        row = layout.row()
        row.template_list(
            "BCR_UL_SetupList", "",
            props, "setups",
            props, "active_index",
            rows=5,
        )

        col = row.column(align=True)
        col.operator("bcr.add_setup", icon='ADD', text="")
        col.operator("bcr.remove_setup", icon='REMOVE', text="")
        col.separator()
        col.operator("bcr.duplicate_setup", icon='DUPLICATE', text="")
        col.separator()
        op = col.operator("bcr.move_setup", icon='TRIA_UP', text="")
        op.direction = 'UP'
        op = col.operator("bcr.move_setup", icon='TRIA_DOWN', text="")
        op.direction = 'DOWN'

        # ── Active setup details ──
        if props.setups and 0 <= props.active_index < len(props.setups):
            setup = props.setups[props.active_index]

            box = layout.box()

            header = box.row(align=True)
            header.label(text="", icon='PREFERENCES')
            header.prop(setup, "name", text="")
            header.operator("bcr.navigate_to_camera", text="", icon='VIEWZOOM')

            box.prop(setup, "camera", icon='CAMERA_DATA')
            box.prop(setup, "view_layer_name", text="View Layer", icon='RENDERLAYERS')
            box.prop(setup, "world", text="World", icon='WORLD')
            # Render mode + frame controls inline
            if setup.render_mode == 'FRAME':
                mode_row = box.row(align=True)
                mode_row.prop(setup, "render_mode", expand=True)
                mode_row.prop(setup, "frame_single", text="")
            else:
                mode_row = box.row(align=True)
                mode_row.prop(setup, "render_mode", expand=True)
                mode_row.prop(setup, "frame_start", text="")
                mode_row.prop(setup, "frame_end", text="")

            box.separator()

            res_box = box.box()
            res_box.label(text="Resolution", icon='IMAGE_DATA')
            res_box.prop(setup, "aspect_ratio", text="Aspect Ratio")

            row_res = res_box.row(align=True)
            row_res.prop(setup, "res_x", text="X")
            row_res.prop(setup, "res_y", text="Y")
            if setup.aspect_ratio != 'CUSTOM' and setup.resolution_preset != 'CUSTOM':
                row_res.enabled = False

            res_box.prop(setup, "resolution_preset", text="Quality")

            # Show final resolution
            final_row = res_box.row(align=True)
            final_row.enabled = False
            final_row.label(text=f"Output: {setup.res_x} × {setup.res_y} px", icon='RENDER_RESULT')

            box.separator()

            # ── Exclude Objects ──
            exc_box = box.box()
            exc_header = exc_box.row(align=True)
            exc_header.label(text="Exclude Objects", icon='GHOST_DISABLED')
            exc_header.operator("bcr.clear_exclude_objects", text="", icon='TRASH')

            exc_row = exc_box.row()
            exc_row.template_list(
                "BCR_UL_ExcludeList", "",
                setup, "exclude_objects",
                setup, "exclude_active_index",
                rows=3,
            )

            exc_col = exc_row.column(align=True)
            exc_col.operator("bcr.add_exclude_object", icon='ADD', text="")
            exc_col.operator("bcr.remove_exclude_object", icon='REMOVE', text="")

            if not setup.exclude_objects:
                hint = exc_box.row()
                hint.enabled = False
                hint.label(text="Select object(s), then +", icon='INFO')

            # ── Collections ──
            hidden = _hidden_collections(setup)
            header, body = box.panel("bcr_collections", default_closed=True)
            header.label(text="Collections", icon='OUTLINER_COLLECTION')
            if hidden:
                header.label(text=rpt_("{:d} hidden").format(len(hidden)))
            if body:
                collections = _scene_collections(context.scene)
                if not collections:
                    body.label(text="The scene has no collections", icon='INFO')
                col = body.column(align=True)
                parents_hidden = set()
                for collection, depth in collections:
                    row = col.row(align=True)
                    for _ in range(depth):
                        row.separator(factor=2.0)
                    visible = collection not in hidden
                    under_hidden = any(parent in hidden for parent in _collection_parents(
                        context.scene, collection)) if depth else False
                    row.active = not under_hidden
                    op = row.operator("bcr.toggle_collection", text=collection.name, emboss=False,
                                      icon='CHECKBOX_HLT' if visible else 'CHECKBOX_DEHLT')
                    op.collection_name = collection.name
                row = body.row()
                row.operator("bcr.show_all_collections", icon='HIDE_OFF')
                hint = body.row()
                hint.enabled = False
                hint.label(text="Unchecked collections are not rendered. Shift: only this one", icon='INFO')

            # ── Light Mix ──
            header, body = box.panel("bcr_light_mix", default_closed=True)
            header.label(text="Light Mix", icon='OUTLINER_OB_LIGHT')
            if setup.light_mix_source != 'NONE':
                header.label(text=setup.light_mix_setup if setup.light_mix_source == 'SETUP'
                             else os.path.basename(setup.light_mix_file))
            if body:
                body.prop(setup, "light_mix_source", expand=True)
                if setup.light_mix_source == 'SETUP':
                    settings = getattr(context.scene, "cycles_light_mix", None)
                    if settings is not None:
                        body.prop_search(setup, "light_mix_setup", settings, "setups", text="Setup")
                    else:
                        body.label(text="Light Mix is not available", icon='ERROR')
                elif setup.light_mix_source == 'FILE':
                    body.prop(setup, "light_mix_file", text="File")
                if setup.light_mix_source != 'NONE':
                    try:
                        _light_mix_data(context.scene, setup)
                    except (RuntimeError, ValueError, OSError) as ex:
                        body.label(text=str(ex), icon='ERROR')
                    hint = body.row()
                    hint.enabled = False
                    hint.label(text="Applied to the view layer \"{:s}\" for this setup".format(
                        setup.view_layer_name), icon='INFO')

        layout.separator()

        # ── Global output ──
        out_box = layout.box()
        out_box.label(text="Output", icon='FILE_FOLDER')
        out_box.prop(props, "output_path", text="Path")

        folder_row = out_box.row(align=True)
        folder_row.prop(props, "use_setup_folder", icon='PREFERENCES')
        folder_row.prop(props, "use_date_folder", icon='TIME')
        folder_row.prop(props, "use_viewlayer_folder", icon='RENDERLAYERS')

        # Preview resolved path
        resolved = _resolve_output_path(props.output_path)
        if props.setups and 0 <= props.active_index < len(props.setups):
            setup = props.setups[props.active_index]
            if props.use_setup_folder:
                resolved = os.path.join(resolved, setup.name)
            if props.use_date_folder:
                resolved = os.path.join(resolved, datetime.date.today().strftime("%Y-%m-%d"))
            if props.use_viewlayer_folder:
                resolved = os.path.join(resolved, setup.view_layer_name)
        else:
            if props.use_setup_folder:
                resolved = os.path.join(resolved, "<setup>")
            if props.use_date_folder:
                resolved = os.path.join(resolved, datetime.date.today().strftime("%Y-%m-%d"))
            if props.use_viewlayer_folder:
                resolved = os.path.join(resolved, "<viewlayer>")
        hint_row = out_box.row()
        hint_row.enabled = False
        hint_row.label(text=f"→ {resolved}", icon='DISCLOSURE_TRI_RIGHT')

        # ── File naming ──
        name_box = layout.box()
        name_box.label(text="File Naming", icon='SORTALPHA')
        name_row = name_box.row(align=True)
        name_row.prop(props, "include_setup_name", icon='PREFERENCES')
        name_row.prop(props, "include_view_layer", icon='RENDERLAYERS')

        # Preview the file name
        if props.setups and 0 <= props.active_index < len(props.setups):
            setup = props.setups[props.active_index]
            cam_name = setup.camera.name if setup.camera else "Camera"
            parts = []
            if props.include_setup_name:
                parts.append(setup.name)
            parts.append(cam_name)
            if props.include_view_layer:
                parts.append(setup.view_layer_name)
            preview = "_".join(parts) + "_####"
            preview_row = name_box.row()
            preview_row.enabled = False
            preview_row.label(text=f"→ {preview}", icon='FILE_IMAGE')

            # Save Passes
            name_box.separator()
            pass_row = name_box.row(align=True)
            pass_row.prop(props, "save_passes", icon='RENDERLAYERS')
            pass_row.operator("bcr.refresh_pass_nodes", text="", icon='FILE_REFRESH')
            if props.save_passes:
                name_box.prop(props, "pass_format", text="Pass Format")
                prefix = "_".join(parts) + "_"
                if props.pass_format == 'OPEN_EXR_MULTILAYER':
                    pp_row = name_box.row()
                    pp_row.enabled = False
                    pp_row.label(text=f"→ {prefix}#### (all passes in one EXR)", icon='NODE')
                else:
                    pp_row = name_box.row()
                    pp_row.enabled = False
                    pp_row.label(text=f"→ {prefix}Combined_####", icon='NODE')
                    pp_row2 = name_box.row()
                    pp_row2.enabled = False
                    pp_row2.label(text=f"→ {prefix}DiffuseColor_####  ...", icon='NODE')

        layout.separator()

        # ── Batch render ──
        visible_count = sum(1 for s in props.setups if not s.is_hidden)
        total_count = len(props.setups)

        row_bottom = layout.row(align=True)
        row_bottom.scale_y = 1.5
        row_bottom.operator(
            "bcr.batch_render",
            text=f"Batch Render ({visible_count}/{total_count} visible)",
            icon='RENDER_STILL',
        )
        row_bottom.operator("bcr.clear_statuses", text="", icon='X')


class BCR_OT_ToggleSidebar(Operator):
    bl_idname = "bcr.toggle_sidebar"
    bl_label = "Batch Camera Renderer"
    bl_description = "Open the Batch Camera Renderer N-panel"

    def execute(self, context):
        # Auto-fix default path if empty or uses // without saved blend
        props = context.scene.bcr
        if not props.output_path or (props.output_path.startswith("//") and not bpy.data.filepath):
            props.output_path = _get_default_output_path()

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.show_region_ui = True
                area.tag_redraw()
                break
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
#  Properties Panel shortcut
# ─────────────────────────────────────────────────────────────────────────────


class BCR_PT_RenderPanel(Panel):
    bl_label = "Batch Camera Renderer"
    bl_idname = "BCR_PT_RenderPanel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.operator("bcr.toggle_sidebar", text="Open Batch Camera Renderer", icon='CAMERA_DATA')


# ─────────────────────────────────────────────────────────────────────────────
#  Registration
# ─────────────────────────────────────────────────────────────────────────────

def _draw_render_menu(self, _context):
    self.layout.separator()
    self.layout.operator("bcr.toggle_sidebar", text="Batch Camera Renderer", icon='CAMERA_DATA')


classes = (
    BCR_ExcludeObject,
    BCR_CollectionRef,
    BCR_RenderSetup,
    BCR_SceneProperties,
    BCR_UL_SetupList,
    BCR_UL_ExcludeList,
    BCR_OT_AddSetup,
    BCR_OT_RemoveSetup,
    BCR_OT_DuplicateSetup,
    BCR_OT_MoveSetup,
    BCR_OT_NavigateToCamera,
    BCR_OT_AddExcludeObject,
    BCR_OT_RemoveExcludeObject,
    BCR_OT_ClearExcludeObjects,
    BCR_OT_ToggleCollection,
    BCR_OT_ShowAllCollections,
    BCR_OT_RefreshPassNodes,
    BCR_OT_BatchRender,
    BCR_OT_ClearStatuses,
    BCR_OT_ToggleSidebar,
    BCR_PT_SidebarPanel,
    BCR_PT_RenderPanel,
)


@bpy.app.handlers.persistent
def _bcr_load_handler(dummy):
    """Auto-fix output path after file load."""
    for scene in bpy.data.scenes:
        if hasattr(scene, 'bcr'):
            props = scene.bcr
            if not props.output_path or (props.output_path.startswith("//") and not bpy.data.filepath):
                props.output_path = _get_default_output_path()


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bcr = PointerProperty(type=BCR_SceneProperties)
    bpy.app.handlers.load_post.append(_bcr_load_handler)
    bpy.types.TOPBAR_MT_render.append(_draw_render_menu)


def unregister():
    bpy.types.TOPBAR_MT_render.remove(_draw_render_menu)
    if _bcr_load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_bcr_load_handler)
    del bpy.types.Scene.bcr
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
