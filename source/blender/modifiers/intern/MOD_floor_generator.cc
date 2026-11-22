/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup modifiers
 *
 * BlenderViz: Floor Generator modifier, as FloorGenerator for 3ds Max. Fills the flat faces of
 * the mesh (or the closed outlines of a curve or of edges) with boards or tiles in a pattern. The
 * work is done by #geometry::floor_generate, shared with the Floor Generator node.
 */

#include "BLI_math_vector_types.hh"

#include "BLT_translation.hh"

#include "DNA_mesh_types.h"
#include "DNA_screen_types.h"

#include "BKE_modifier.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "RNA_access.hh"
#include "RNA_prototypes.hh"

#include "GEO_floor_generator.hh"

#include "MOD_modifiertypes.hh"
#include "MOD_ui_common.hh"

namespace blender {

static void init_data(ModifierData *md)
{
  FloorGeneratorModifierData *fmd = reinterpret_cast<FloorGeneratorModifierData *>(md);
  INIT_DEFAULT_STRUCT_AFTER(fmd, modifier);
}

static geometry::FloorGeneratorParams params_from_modifier(const FloorGeneratorModifierData &fmd)
{
  using namespace geometry;
  FloorGeneratorParams p;
  p.pattern = FloorPattern(fmd.pattern);
  const bool vary_length = fmd.flag & MOD_FLOORGEN_VARY_LENGTH;
  const bool vary_width = fmd.flag & MOD_FLOORGEN_VARY_WIDTH;
  p.length = {vary_length ? std::min(fmd.length_min, fmd.length) : fmd.length,
              fmd.length,
              fmd.length_spread};
  p.width = {
      vary_width ? std::min(fmd.width_min, fmd.width) : fmd.width, fmd.width, fmd.width_spread};
  p.grout_length = fmd.grout_length;
  p.grout_width = fmd.grout_width;
  p.row_offset = fmd.row_offset;
  p.height = fmd.height;
  p.bevel_height = fmd.bevel_height;
  p.bevel_width = (fmd.flag & MOD_FLOORGEN_BEVEL_45) ? fmd.bevel_height : fmd.bevel_width;
  p.parallel_boards = fmd.parallel_boards;
  p.chevron_angle = fmd.chevron_angle;
  p.sub_boards = fmd.sub_boards;
  p.hex_radius = fmd.hex_radius;
  p.panel_size = fmd.panel_size;
  p.rotation = {fmd.rotation_min, fmd.rotation_max, fmd.rotation_spread};
  p.offset_x = {fmd.offset_x_min, fmd.offset_x_max, fmd.offset_x_spread};
  p.offset_y = (fmd.flag & MOD_FLOORGEN_OFFSET_Y_LOCKED) ?
                   p.offset_x :
                   FloorRandomRange{fmd.offset_y_min, fmd.offset_y_max, fmd.offset_y_spread};
  p.tilt = {fmd.tilt_min, fmd.tilt_max, fmd.tilt_spread};
  p.overlap = fmd.overlap;
  p.missing = fmd.missing;
  p.scale = fmd.pattern_scale;
  p.direction = fmd.direction;
  p.offset = float2(fmd.pattern_offset[0], fmd.pattern_offset[1]);
  p.seed = fmd.seed;
  p.weld = fmd.weld;
  p.random_flip_u = fmd.flag & MOD_FLOORGEN_RANDOM_FLIP_U;
  p.random_flip_v = fmd.flag & MOD_FLOORGEN_RANDOM_FLIP_V;
  p.random_offset_u = fmd.flag & MOD_FLOORGEN_RANDOM_OFFSET_U;
  p.random_offset_v = fmd.flag & MOD_FLOORGEN_RANDOM_OFFSET_V;
  p.uv_rotation = fmd.uv_rotation;
  p.random_rotate_uv = fmd.flag & MOD_FLOORGEN_RANDOM_ROTATE_UV;
  p.material_count = fmd.material_count;
  p.uv_map_name = fmd.uvmap_name;
  return p;
}

static Mesh *modify_mesh(ModifierData *md, const ModifierEvalContext *ctx, Mesh *mesh)
{
  const FloorGeneratorModifierData *fmd = reinterpret_cast<const FloorGeneratorModifierData *>(md);
  std::string warning;
  Mesh *result = geometry::floor_generate(*mesh, params_from_modifier(*fmd), &warning);
  if (!warning.empty()) {
    BKE_modifier_set_warning(ctx->object, md, "%s", warning.c_str());
  }
  if (result == nullptr) {
    return mesh;
  }
  return result;
}

static void panel_draw(const bContext * /*C*/, Panel *panel)
{
  ui::Layout &layout = *panel->layout;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, nullptr);
  const int pattern = RNA_enum_get(ptr, "pattern");
  layout.use_property_split_set(true);

  layout.prop(ptr, "pattern", UI_ITEM_NONE, std::nullopt, ICON_NONE);

  ui::Layout *col = &layout.column(true);
  switch (pattern) {
    case MOD_FLOORGEN_STANDARD:
    case MOD_FLOORGEN_STACKED: {
      col->prop(ptr, "length", UI_ITEM_NONE, IFACE_("Length"), ICON_NONE);
      col->prop(ptr, "use_vary_length", UI_ITEM_NONE, IFACE_("Vary"), ICON_NONE);
      if (RNA_boolean_get(ptr, "use_vary_length")) {
        col->prop(ptr, "length_min", UI_ITEM_NONE, IFACE_("Min"), ICON_NONE);
        col->prop(ptr, "length_spread", UI_ITEM_NONE, IFACE_("Spread"), ICON_NONE);
      }
      col = &layout.column(true);
      col->prop(ptr, "width", UI_ITEM_NONE, IFACE_("Width"), ICON_NONE);
      col->prop(ptr, "use_vary_width", UI_ITEM_NONE, IFACE_("Vary"), ICON_NONE);
      if (RNA_boolean_get(ptr, "use_vary_width")) {
        col->prop(ptr, "width_min", UI_ITEM_NONE, IFACE_("Min"), ICON_NONE);
        col->prop(ptr, "width_spread", UI_ITEM_NONE, IFACE_("Spread"), ICON_NONE);
      }
      if (pattern == MOD_FLOORGEN_STANDARD) {
        layout.prop(ptr, "row_offset", UI_ITEM_NONE, std::nullopt, ICON_NONE);
      }
      break;
    }
    case MOD_FLOORGEN_HERRINGBONE:
    case MOD_FLOORGEN_CHEVRON:
      col->prop(ptr, "length", UI_ITEM_NONE, IFACE_("Length"), ICON_NONE);
      col->prop(ptr, "width", UI_ITEM_NONE, IFACE_("Width"), ICON_NONE);
      if (pattern == MOD_FLOORGEN_HERRINGBONE) {
        layout.prop(ptr, "parallel_boards", UI_ITEM_NONE, std::nullopt, ICON_NONE);
      }
      else {
        layout.prop(ptr, "chevron_angle", UI_ITEM_NONE, IFACE_("Angle"), ICON_NONE);
      }
      break;
    case MOD_FLOORGEN_BASKET_WEAVE:
      col->prop(ptr, "length", UI_ITEM_NONE, IFACE_("Square Size"), ICON_NONE);
      layout.prop(ptr, "sub_boards", UI_ITEM_NONE, std::nullopt, ICON_NONE);
      break;
    case MOD_FLOORGEN_HEXAGON:
      col->prop(ptr, "hex_radius", UI_ITEM_NONE, IFACE_("Radius"), ICON_NONE);
      break;
    case MOD_FLOORGEN_VERSAILLES:
      col->prop(ptr, "panel_size", UI_ITEM_NONE, std::nullopt, ICON_NONE);
      col->prop(ptr, "width", UI_ITEM_NONE, IFACE_("Frame Width"), ICON_NONE);
      layout.prop(ptr, "sub_boards", UI_ITEM_NONE, std::nullopt, ICON_NONE);
      break;
  }

  col = &layout.column(true);
  col->prop(ptr, "grout_length", UI_ITEM_NONE, IFACE_("Grout Length"), ICON_NONE);
  col->prop(ptr, "grout_width", UI_ITEM_NONE, IFACE_("Width"), ICON_NONE);

  modifier_error_message_draw(layout, ptr);
}

static void height_panel_draw(const bContext * /*C*/, Panel *panel)
{
  ui::Layout &layout = *panel->layout;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, nullptr);
  layout.use_property_split_set(true);
  layout.prop(ptr, "height", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  ui::Layout &col = layout.column(true);
  col.prop(ptr, "bevel_height", UI_ITEM_NONE, IFACE_("Bevel Height"), ICON_NONE);
  col.prop(ptr, "use_bevel_45", UI_ITEM_NONE, IFACE_("45 Degrees"), ICON_NONE);
  if (!RNA_boolean_get(ptr, "use_bevel_45")) {
    col.prop(ptr, "bevel_width", UI_ITEM_NONE, IFACE_("Width"), ICON_NONE);
  }
}

static void range_draw(ui::Layout &layout, PointerRNA *ptr, const char *name, const char *label)
{
  ui::Layout &col = layout.column(true);
  col.prop(ptr, (std::string(name) + "_max").c_str(), UI_ITEM_NONE, label, ICON_NONE);
  col.prop(ptr, (std::string(name) + "_min").c_str(), UI_ITEM_NONE, IFACE_("Min"), ICON_NONE);
  col.prop(
      ptr, (std::string(name) + "_spread").c_str(), UI_ITEM_NONE, IFACE_("Spread"), ICON_NONE);
}

static void variation_panel_draw(const bContext * /*C*/, Panel *panel)
{
  ui::Layout &layout = *panel->layout;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, nullptr);
  layout.use_property_split_set(true);
  range_draw(layout, ptr, "rotation", IFACE_("Rotation"));
  range_draw(layout, ptr, "offset_x", IFACE_("Offset X"));
  layout.prop(ptr, "use_offset_y_locked", UI_ITEM_NONE, IFACE_("Offset Y as X"), ICON_NONE);
  if (!RNA_boolean_get(ptr, "use_offset_y_locked")) {
    range_draw(layout, ptr, "offset_y", IFACE_("Offset Y"));
  }
  range_draw(layout, ptr, "tilt", IFACE_("Tilt"));
  layout.prop(ptr, "overlap", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "missing", UI_ITEM_NONE, IFACE_("Missing Boards"), ICON_NONE);
}

static void placement_panel_draw(const bContext * /*C*/, Panel *panel)
{
  ui::Layout &layout = *panel->layout;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, nullptr);
  layout.use_property_split_set(true);
  layout.prop(ptr, "pattern_scale", UI_ITEM_NONE, IFACE_("Scale"), ICON_NONE);
  layout.prop(ptr, "direction", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "pattern_offset", UI_ITEM_NONE, IFACE_("Offset"), ICON_NONE);
  layout.prop(ptr, "seed", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "weld", UI_ITEM_NONE, IFACE_("Weld Threshold"), ICON_NONE);
}

static void uv_panel_draw(const bContext * /*C*/, Panel *panel)
{
  ui::Layout &layout = *panel->layout;
  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, nullptr);
  layout.use_property_split_set(true);
  layout.prop(ptr, "uv_layer", UI_ITEM_NONE, IFACE_("UV Map"), ICON_NONE);
  layout.prop(ptr, "uv_rotation", UI_ITEM_NONE, IFACE_("UV Rotation"), ICON_NONE);
  layout.prop(ptr, "use_random_rotate_uv", UI_ITEM_NONE, IFACE_("Random 90°"), ICON_NONE);
  ui::Layout &col = layout.column(true, IFACE_("Random Flip"));
  col.prop(ptr, "use_random_flip_u", UI_ITEM_NONE, IFACE_("U"), ICON_NONE);
  col.prop(ptr, "use_random_flip_v", UI_ITEM_NONE, IFACE_("V"), ICON_NONE);
  ui::Layout &col2 = layout.column(true, IFACE_("Random Offset"));
  col2.prop(ptr, "use_random_offset_u", UI_ITEM_NONE, IFACE_("U"), ICON_NONE);
  col2.prop(ptr, "use_random_offset_v", UI_ITEM_NONE, IFACE_("V"), ICON_NONE);
  layout.prop(ptr, "material_count", UI_ITEM_NONE, IFACE_("Random Materials"), ICON_NONE);
}

static void panel_register(ARegionType *region_type)
{
  PanelType *panel_type = modifier_panel_register(
      region_type, eModifierType_FloorGenerator, panel_draw);
  modifier_subpanel_register(
      region_type, "height", "Height and Bevel", nullptr, height_panel_draw, panel_type);
  modifier_subpanel_register(
      region_type, "variation", "Variation", nullptr, variation_panel_draw, panel_type);
  modifier_subpanel_register(
      region_type, "placement", "Placement", nullptr, placement_panel_draw, panel_type);
  modifier_subpanel_register(
      region_type, "uv", "UVs and Materials", nullptr, uv_panel_draw, panel_type);
}

ModifierTypeInfo modifierType_FloorGenerator = {
    /*idname*/ "FloorGenerator",
    /*name*/ N_("FloorGenerator"),
    /*struct_name*/ "FloorGeneratorModifierData",
    /*struct_size*/ sizeof(FloorGeneratorModifierData),
    /*srna*/ &RNA_FloorGeneratorModifier,
    /*type*/ ModifierTypeType::Constructive,
    /*flags*/
    (eModifierTypeFlag_AcceptsMesh | eModifierTypeFlag_AcceptsCVs |
     eModifierTypeFlag_SupportsEditmode | eModifierTypeFlag_EnableInEditmode),
    /*icon*/ ICON_MESH_GRID,

    /*copy_data*/ BKE_modifier_copydata_generic,

    /*deform_verts*/ nullptr,
    /*deform_matrices*/ nullptr,
    /*deform_verts_EM*/ nullptr,
    /*deform_matrices_EM*/ nullptr,
    /*modify_mesh*/ modify_mesh,
    /*modify_geometry_set*/ nullptr,

    /*init_data*/ init_data,
    /*required_data_mask*/ nullptr,
    /*free_data*/ nullptr,
    /*is_disabled*/ nullptr,
    /*update_depsgraph*/ nullptr,
    /*depends_on_time*/ nullptr,
    /*depends_on_normals*/ nullptr,
    /*foreach_ID_link*/ nullptr,
    /*foreach_tex_link*/ nullptr,
    /*free_runtime_data*/ nullptr,
    /*panel_register*/ panel_register,
    /*blend_write*/ nullptr,
    /*blend_read*/ nullptr,
    /*foreach_cache*/ nullptr,
    /*foreach_working_space_color*/ nullptr,
};

}  // namespace blender
