/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup shdnodes
 *
 * BlenderViz: Texture Randomizer, as the V-Ray UVW Randomizer. Placed before a texture, it gives
 * the texture coordinates a random offset, rotation, scale and flip per object, object index, mesh
 * element, face (material index), particle or UV tile (stochastic tiling). Before an Image
 * Texture, Tile Blend hides the edges between the tiles. Cycles only; EEVEE passes the coordinates
 * through.
 */

#include "node_shader_util.hh"
#include "node_util.hh"

#include "BLI_math_constants.h"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

namespace blender {

namespace nodes::node_shader_tex_randomizer_cc {

/* The same bits as #NodeTexRandomizerMode and #NodeTexRandomizerOption in Cycles. */
enum {
  TEX_RANDOMIZER_OBJECT = (1 << 0),
  TEX_RANDOMIZER_OBJECT_INDEX = (1 << 1),
  TEX_RANDOMIZER_ELEMENT = (1 << 2),
  TEX_RANDOMIZER_FACE = (1 << 3),
  TEX_RANDOMIZER_PARTICLE = (1 << 4),
  TEX_RANDOMIZER_STOCHASTIC = (1 << 5),
};
enum {
  TEX_RANDOMIZER_FLIP_U = (1 << 0),
  TEX_RANDOMIZER_FLIP_V = (1 << 1),
  TEX_RANDOMIZER_V_SCALE = (1 << 2),
};

static void add_range(PanelDeclarationBuilder &panel,
                      const char *name,
                      const float from,
                      const float to,
                      const PropertySubType subtype,
                      const char *description,
                      const bool available = true)
{
  const std::string prefix = name;
  panel.add_input<decl::Float>(UString(prefix + " From"))
      .default_value(from)
      .subtype(subtype)
      .available(available)
      .description(std::string("Smallest ") + description);
  panel.add_input<decl::Float>(UString(prefix + " To"))
      .default_value(to)
      .subtype(subtype)
      .available(available)
      .description(std::string("Largest ") + description);
  panel.add_input<decl::Float>(UString(prefix + " Steps"))
      .default_value(0.0f)
      .min(0.0f)
      .max(1000.0f)
      .available(available)
      .description(
          "Number of values from From to To (a full turn of rotation leaves out To). 0: any "
          "value in between");
}

static void sh_node_tex_randomizer_declare(NodeDeclarationBuilder &b)
{
  const bNode *node = b.node_or_null();
  const int modes = node ? node->custom1 : TEX_RANDOMIZER_ELEMENT;
  const int options = node ? node->custom2 : 0;

  b.add_input<decl::Vector>("Vector"_ustr)
      .default_input_type(NODE_DEFAULT_INPUT_POSITION_FIELD)
      .description("Texture coordinates to randomize, the active UV map when not connected");
  b.add_input<decl::Float>("Seed"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .max(100000.0f)
      .description("Changes all the random values");
  b.add_input<decl::Float>("Tile Blend"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .available(modes & TEX_RANDOMIZER_STOCHASTIC)
      .description(
          "Blend the edges between UV tiles, in an Image Texture connected to the Vector output");

  PanelDeclarationBuilder &offset = b.add_panel("Offset"_ustr).default_closed(false);
  add_range(offset, "U Offset", 0.0f, 1.0f, PROP_NONE, "move of the texture along U");
  add_range(offset, "V Offset", 0.0f, 1.0f, PROP_NONE, "move of the texture along V");

  PanelDeclarationBuilder &rotation = b.add_panel("Rotation"_ustr).default_closed(false);
  add_range(rotation, "Rotation", 0.0f, float(M_PI * 2.0), PROP_ANGLE, "rotation of the texture");

  PanelDeclarationBuilder &scale = b.add_panel("Scale"_ustr).default_closed(false);
  add_range(scale, "U Scale", 100.0f, 100.0f, PROP_PERCENTAGE, "size of the texture (along U)");
  add_range(scale,
            "V Scale",
            100.0f,
            100.0f,
            PROP_PERCENTAGE,
            "size of the texture along V",
            options & TEX_RANDOMIZER_V_SCALE);

  b.add_output<decl::Vector>("Vector"_ustr).description("Randomized texture coordinates");
  b.add_output<decl::Float>("Random"_ustr)
      .description("Random value from 0 to 1, the same where the coordinates get the same change");
}

static void node_shader_init_tex_randomizer(bNodeTree * /*ntree*/, bNode *node)
{
  node->custom1 = TEX_RANDOMIZER_ELEMENT;
  node->custom2 = 0;
}

static void node_shader_buts_tex_randomizer(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.label(IFACE_("Randomize by:"), ICON_NONE);
  ui::Layout &grid = layout.grid_flow(true, 2, true, false, false);
  grid.prop(ptr, "use_object", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  grid.prop(ptr, "use_element", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  grid.prop(ptr, "use_object_index", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  grid.prop(ptr, "use_face", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  grid.prop(ptr, "use_particle", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  grid.prop(ptr, "use_stochastic", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  ui::Layout &row = layout.row(true);
  row.prop(ptr, "use_flip_u", ui::ITEM_R_TOGGLE, std::nullopt, ICON_NONE);
  row.prop(ptr, "use_flip_v", ui::ITEM_R_TOGGLE, std::nullopt, ICON_NONE);
  layout.prop(ptr, "use_v_scale", UI_ITEM_NONE, std::nullopt, ICON_NONE);
}

/* EEVEE: the coordinates as they are. */
static int node_shader_gpu_tex_randomizer(GPUMaterial *mat,
                                          bNode * /*node*/,
                                          bNodeExecData * /*execdata*/,
                                          GPUNodeStack *in,
                                          GPUNodeStack *out)
{
  if (in[0].link) {
    GPU_link(mat, "set_rgb", in[0].link, &out[0].link);
  }
  else {
    GPU_link(mat, "node_uvmap", GPU_attribute(mat, CD_AUTO_FROM_NAME, ""), &out[0].link);
  }
  static float zero = 0.0f;
  GPU_link(mat, "set_value", GPU_constant(&zero), &out[1].link);
  return true;
}

}  // namespace nodes::node_shader_tex_randomizer_cc

void register_node_type_sh_tex_randomizer()
{
  namespace file_ns = nodes::node_shader_tex_randomizer_cc;

  static bke::bNodeType ntype;

  sh_node_type_base(&ntype, "ShaderNodeTexRandomizer"_ustr);
  ntype.ui_name = "Texture Randomizer";
  ntype.ui_description =
      "Random offset, rotation, scale and flip of the texture coordinates per object, element, "
      "face, particle or UV tile, placed before a texture (Cycles)";
  ntype.enum_name_legacy = "TEX_RANDOMIZER";
  ntype.nclass = NODE_CLASS_OP_VECTOR;
  ntype.declare = file_ns::sh_node_tex_randomizer_declare;
  ntype.draw_buttons = file_ns::node_shader_buts_tex_randomizer;
  ntype.initfunc = file_ns::node_shader_init_tex_randomizer;
  ntype.gpu_fn = file_ns::node_shader_gpu_tex_randomizer;
  ntype.default_width = bke::NodeWidth::_240;

  bke::node_register_type(ntype);
}

}  // namespace blender
