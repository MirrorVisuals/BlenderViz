# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""
BlenderViz lights, as in V-Ray: the Mesh Light (any mesh made a light, its materials kept) and the
Cylinder Light (an area light of shape Cylinder: a tube that emits all around). Cycles renders both
as emissive meshes with the power spread over their surface, as it does for area lights. See
intern/cycles/blender/shader.cpp (sync_mesh_lights) and light.cpp (sync_light_cylinder_mesh).
"""

from __future__ import annotations

import bpy
from bpy.props import FloatProperty
from bpy.types import Operator, Panel

from .ui import CyclesButtonsPanel

MESH_LIGHT_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}


class CYCLES_OBJECT_PT_mesh_light(CyclesButtonsPanel, Panel):
    bl_label = "Mesh Light"
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return CyclesButtonsPanel.poll(context) and ob is not None and ob.type in MESH_LIGHT_TYPES

    def draw_header(self, context):
        self.layout.prop(context.object.cycles, "is_mesh_light", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        ob = context.object
        cob = ob.cycles
        layout.active = cob.is_mesh_light

        col = layout.column()
        row = col.row(heading="Temperature")
        row.prop(cob, "mesh_light_use_temperature", text="")
        sub = row.row()
        sub.active = cob.mesh_light_use_temperature
        sub.prop(cob, "mesh_light_temperature", text="")
        col.prop(cob, "mesh_light_color", text="Tint" if cob.mesh_light_use_temperature else "Color")

        col = layout.column()
        col.prop(cob, "mesh_light_units")
        if cob.mesh_light_units == 'POWER':
            col.prop(cob, "mesh_light_power")
        else:
            col.prop(cob, "mesh_light_strength")
        col.prop(cob, "mesh_light_exposure")

        col = layout.column(heading="Options")
        col.prop(cob, "mesh_light_double_sided")
        col.prop(ob, "visible_camera", text="Visible to Camera")

        view_layer = context.view_layer
        if view_layer.lightgroups:
            layout.prop_search(ob, "lightgroup", view_layer, "lightgroups", text="Light Group")
        else:
            layout.prop(ob, "lightgroup", text="Light Group")


class CYCLES_OT_cylinder_light_add(Operator):
    """Add a Cylinder light: a tube that emits all around, like a fluorescent tube or a neon"""
    bl_idname = "cycles.cylinder_light_add"
    bl_label = "Cylinder Light"
    bl_options = {'REGISTER', 'UNDO'}

    length: FloatProperty(name="Length", default=1.2, min=0.0, subtype='DISTANCE')
    radius: FloatProperty(name="Radius", default=0.013, min=0.0, subtype='DISTANCE')
    power: FloatProperty(name="Power", default=40.0, min=0.0, subtype='POWER', unit='POWER')

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        bpy.ops.object.light_add(type='AREA', location=context.scene.cursor.location)
        ob = context.object
        ob.name = "Cylinder Light"
        light = ob.data
        light.name = "Cylinder Light"
        light.shape = 'CYLINDER'
        light.size = self.length
        light.shadow_soft_size = self.radius
        light.energy = self.power
        return {'FINISHED'}


def light_add_menu(self, _context):
    self.layout.operator(CYCLES_OT_cylinder_light_add.bl_idname, text="Cylinder", icon='MESH_CYLINDER')


classes = (
    CYCLES_OBJECT_PT_mesh_light,
    CYCLES_OT_cylinder_light_add,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_light_add.append(light_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_light_add.remove(light_add_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
