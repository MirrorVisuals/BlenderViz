/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * BlenderViz: Floor Generator node, the Floor Generator modifier (as FloorGenerator for 3ds Max)
 * in a node tree: fills the flat faces of a mesh, or closed curves, with boards or tiles.
 */

#include "BKE_curve_to_mesh.hh"
#include "BKE_curves.hh"
#include "BKE_lib_id.hh"

#include "GEO_floor_generator.hh"
#include "GEO_foreach_geometry.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_floor_generator_cc {

using geometry::FloorPattern;

static const EnumPropertyItem pattern_items[] = {
    {int(FloorPattern::Standard),
     "STANDARD",
     0,
     N_("Standard"),
     N_("Rows of boards, the joints of each row shifted by the row offset (running bond)")},
    {int(FloorPattern::Stacked),
     "STACKED",
     0,
     N_("Stacked"),
     N_("Rows and columns with the joints lined up")},
    {int(FloorPattern::Herringbone),
     "HERRINGBONE",
     0,
     N_("Herringbone"),
     N_("Boards at right angles in a zigzag (2 parallel boards: double herringbone)")},
    {int(FloorPattern::Chevron),
     "CHEVRON",
     0,
     N_("Chevron"),
     N_("Boards cut at an angle, meeting in straight lines")},
    {int(FloorPattern::BasketWeave),
     "BASKET_WEAVE",
     0,
     N_("Basket Weave"),
     N_("Squares of parallel boards, every other square turned")},
    {int(FloorPattern::Hexagon), "HEXAGON", 0, N_("Hexagon"), N_("Hexagonal tiles")},
    {int(FloorPattern::Versailles),
     "VERSAILLES",
     0,
     N_("Versailles"),
     N_("Square panels: a mitred frame around a diagonal basket weave")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .supported_type({GeometryComponent::Type::Mesh, GeometryComponent::Type::Curve})
      .description(
          "Flat faces to fill (faces in the same plane share the pattern), or closed curves or "
          "edge loops as the outline of the floor");
  b.add_output<decl::Geometry>("Floor"_ustr).align_with_previous();
  b.add_input<decl::Menu>("Pattern"_ustr).static_items(pattern_items).optional_label();

  const Array<int> rows = {int(FloorPattern::Standard), int(FloorPattern::Stacked)};
  const Array<int> boards = {int(FloorPattern::Standard),
                             int(FloorPattern::Stacked),
                             int(FloorPattern::Herringbone),
                             int(FloorPattern::Chevron),
                             int(FloorPattern::BasketWeave)};
  const Array<int> widths = {int(FloorPattern::Standard),
                             int(FloorPattern::Stacked),
                             int(FloorPattern::Herringbone),
                             int(FloorPattern::Chevron),
                             int(FloorPattern::Versailles)};
  const Array<int> sub_boards = {int(FloorPattern::BasketWeave), int(FloorPattern::Versailles)};
  const char *spread_tip = N_(
      "How the values are spread between the minimum and the maximum: low for mostly the maximum, "
      "high for mostly the minimum");

  b.add_input<decl::Float>("Length"_ustr)
      .default_value(1.2f)
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Pattern"_ustr, boards)
      .description(
          "Length of the boards (the maximum when they vary); the square of Basket Weave");
  b.add_input<decl::Float>("Length Min"_ustr)
      .default_value(1.2f)
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Pattern"_ustr, rows)
      .description("Shortest board: below the length, the lengths vary");
  b.add_input<decl::Float>("Length Spread"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .usage_by_menu("Pattern"_ustr, rows)
      .description(spread_tip);
  b.add_input<decl::Float>("Width"_ustr)
      .default_value(0.2f)
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Pattern"_ustr, widths)
      .description("Width of the boards (the maximum when they vary); the frame of Versailles");
  b.add_input<decl::Float>("Width Min"_ustr)
      .default_value(0.2f)
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Pattern"_ustr, rows)
      .description("Narrowest row: below the width, the widths of the rows vary");
  b.add_input<decl::Float>("Width Spread"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .usage_by_menu("Pattern"_ustr, rows)
      .description(spread_tip);
  b.add_input<decl::Float>("Row Offset"_ustr)
      .default_value(0.33f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .usage_by_menu("Pattern"_ustr, int(FloorPattern::Standard))
      .description("Shift of each row, as a fraction of the board length");
  b.add_input<decl::Int>("Parallel Boards"_ustr)
      .default_value(1)
      .min(1)
      .max(16)
      .usage_by_menu("Pattern"_ustr, int(FloorPattern::Herringbone))
      .description("Boards side by side in each arm (2: double herringbone)");
  b.add_input<decl::Float>("Angle"_ustr)
      .default_value(0.785398f)
      .min(0.0873f)
      .max(1.4835f)
      .subtype(PROP_ANGLE)
      .usage_by_menu("Pattern"_ustr, int(FloorPattern::Chevron))
      .description("Angle of the boards to the cut line");
  b.add_input<decl::Int>("Sub Boards"_ustr)
      .default_value(3)
      .min(1)
      .max(16)
      .usage_by_menu("Pattern"_ustr, sub_boards)
      .description("Boards in each square");
  b.add_input<decl::Float>("Radius"_ustr)
      .default_value(0.15f)
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Pattern"_ustr, int(FloorPattern::Hexagon))
      .description("Size of a hexagon tile (to its corners)");
  b.add_input<decl::Float>("Panel Size"_ustr)
      .default_value(1.0f)
      .min(0.01f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Pattern"_ustr, int(FloorPattern::Versailles))
      .description("Size of a Versailles panel");
  b.add_input<decl::Float>("Grout Length"_ustr)
      .default_value(0.002f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Gap between the ends of the boards");
  b.add_input<decl::Float>("Grout Width"_ustr)
      .default_value(0.002f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Gap between the sides of the boards");

  {
    PanelDeclarationBuilder &panel = b.add_panel("Height and Bevel"_ustr).default_closed(true);
    panel.add_input<decl::Float>("Height"_ustr)
        .default_value(0.01f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Thickness of the boards (0: flat faces)");
    panel.add_input<decl::Float>("Bevel Height"_ustr)
        .default_value(0.001f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Height of the bevel along the top edges");
    panel.add_input<decl::Float>("Bevel Width"_ustr)
        .default_value(0.001f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Width of the bevel along the top edges");
  }
  {
    PanelDeclarationBuilder &panel = b.add_panel("Variation"_ustr).default_closed(true);
    auto range = [&](const char *name,
                     const char *min_name,
                     const char *spread_name,
                     const bool angle,
                     const char *description) {
      panel.add_input<decl::Float>(UString(name))
          .default_value(0.0f)
          .min(0.0f)
          .subtype(angle ? PROP_ANGLE : PROP_DISTANCE)
          .description(description);
      panel.add_input<decl::Float>(UString(min_name))
          .default_value(0.0f)
          .min(0.0f)
          .subtype(angle ? PROP_ANGLE : PROP_DISTANCE)
          .description("Smallest random amount");
      panel.add_input<decl::Float>(UString(spread_name))
          .default_value(0.5f)
          .min(0.0f)
          .max(1.0f)
          .subtype(PROP_FACTOR)
          .description(spread_tip);
    };
    range("Rotation",
          "Rotation Min",
          "Rotation Spread",
          true,
          "Random rotation of each board around itself (limited by the grout and the overlap)");
    range("Offset X",
          "Offset X Min",
          "Offset X Spread",
          false,
          "Random move of each board along its length (limited by the grout and the overlap)");
    range("Offset Y",
          "Offset Y Min",
          "Offset Y Spread",
          false,
          "Random move of each board across (limited by the grout and the overlap)");
    range("Tilt",
          "Tilt Min",
          "Tilt Spread",
          true,
          "Random tilt of the top of each board, for uneven boards");
    panel.add_input<decl::Float>("Overlap"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description(
            "How far boards may move into each other with the random rotation and offset, past "
            "the grout");
    panel.add_input<decl::Float>("Missing Boards"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .subtype(PROP_FACTOR)
        .description("Fraction of the boards left out, for a damaged floor");
  }
  {
    PanelDeclarationBuilder &panel = b.add_panel("Placement"_ustr).default_closed(true);
    panel.add_input<decl::Float>("Scale"_ustr)
        .default_value(1.0f)
        .min(0.001f)
        .description("Scale of the whole pattern");
    panel.add_input<decl::Float>("Direction"_ustr)
        .default_value(0.0f)
        .subtype(PROP_ANGLE)
        .description("Rotation of the whole pattern");
    panel.add_input<decl::Vector>("Offset"_ustr)
        .subtype(PROP_TRANSLATION)
        .description("Move the whole pattern (X and Y of the floor)");
    panel.add_input<decl::Int>("Seed"_ustr).description("Random seed of the board variation");
    panel.add_input<decl::Float>("Weld Threshold"_ustr)
        .default_value(0.001f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description(
            "Board corners closer than this are merged and smaller pieces of boards left out (at "
            "the edges of the floor)");
  }
  {
    PanelDeclarationBuilder &panel = b.add_panel("UVs and Materials"_ustr).default_closed(true);
    panel.add_input<decl::String>("UV Map"_ustr)
        .default_value("UVMap")
        .description("UV map written (a full size board spans 0 to 1); empty: none");
    panel.add_input<decl::Float>("UV Rotation"_ustr)
        .subtype(PROP_ANGLE)
        .description(
            "Turn the UVs of every board around its middle (90 degrees: across the board)");
    panel.add_input<decl::Bool>("Random UV Rotation"_ustr)
        .description("Turn the UVs of each board by a random quarter turn more");
    panel.add_input<decl::Bool>("Random Flip U"_ustr)
        .description("Mirror the UVs of random boards along their length");
    panel.add_input<decl::Bool>("Random Flip V"_ustr)
        .description("Mirror the UVs of random boards across");
    panel.add_input<decl::Bool>("Random Offset U"_ustr)
        .description("Move the UVs of each board by a random amount along its length");
    panel.add_input<decl::Bool>("Random Offset V"_ustr)
        .description("Move the UVs of each board by a random amount across");
    panel.add_input<decl::Int>("Random Materials"_ustr)
        .default_value(1)
        .min(1)
        .max(32767)
        .description(
            "Give each board a random material index from 0 to this minus 1 (1: the material of "
            "the face)");
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  using namespace geometry;
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry"_ustr);

  FloorGeneratorParams p;
  p.pattern = params.extract_input<FloorPattern>("Pattern"_ustr);
  const float length = params.extract_input<float>("Length"_ustr);
  const float width = params.extract_input<float>("Width"_ustr);
  p.length = {std::min(params.extract_input<float>("Length Min"_ustr), length),
              length,
              params.extract_input<float>("Length Spread"_ustr)};
  p.width = {std::min(params.extract_input<float>("Width Min"_ustr), width),
             width,
             params.extract_input<float>("Width Spread"_ustr)};
  p.row_offset = params.extract_input<float>("Row Offset"_ustr);
  p.parallel_boards = params.extract_input<int>("Parallel Boards"_ustr);
  p.chevron_angle = params.extract_input<float>("Angle"_ustr);
  p.sub_boards = params.extract_input<int>("Sub Boards"_ustr);
  p.hex_radius = params.extract_input<float>("Radius"_ustr);
  p.panel_size = params.extract_input<float>("Panel Size"_ustr);
  p.grout_length = params.extract_input<float>("Grout Length"_ustr);
  p.grout_width = params.extract_input<float>("Grout Width"_ustr);
  p.height = params.extract_input<float>("Height"_ustr);
  p.bevel_height = params.extract_input<float>("Bevel Height"_ustr);
  p.bevel_width = params.extract_input<float>("Bevel Width"_ustr);
  auto range = [&](const char *max_name, const char *min_name, const char *spread_name) {
    const float max = params.extract_input<float>(UString(max_name));
    return FloorRandomRange{std::min(params.extract_input<float>(UString(min_name)), max),
                            max,
                            params.extract_input<float>(UString(spread_name))};
  };
  p.rotation = range("Rotation", "Rotation Min", "Rotation Spread");
  p.offset_x = range("Offset X", "Offset X Min", "Offset X Spread");
  p.offset_y = range("Offset Y", "Offset Y Min", "Offset Y Spread");
  p.tilt = range("Tilt", "Tilt Min", "Tilt Spread");
  p.overlap = params.extract_input<float>("Overlap"_ustr);
  p.missing = params.extract_input<float>("Missing Boards"_ustr);
  p.scale = params.extract_input<float>("Scale"_ustr);
  p.direction = params.extract_input<float>("Direction"_ustr);
  const float3 offset = params.extract_input<float3>("Offset"_ustr);
  p.offset = float2(offset.x, offset.y);
  p.seed = params.extract_input<int>("Seed"_ustr);
  p.weld = params.extract_input<float>("Weld Threshold"_ustr);
  p.uv_map_name = params.extract_input<std::string>("UV Map"_ustr);
  p.random_flip_u = params.extract_input<bool>("Random Flip U"_ustr);
  p.random_flip_v = params.extract_input<bool>("Random Flip V"_ustr);
  p.random_offset_u = params.extract_input<bool>("Random Offset U"_ustr);
  p.random_offset_v = params.extract_input<bool>("Random Offset V"_ustr);
  p.uv_rotation = params.extract_input<float>("UV Rotation"_ustr);
  p.random_rotate_uv = params.extract_input<bool>("Random UV Rotation"_ustr);
  p.material_count = params.extract_input<int>("Random Materials"_ustr);

  std::string warning;
  geometry::foreach_real_geometry(geometry_set, [&](GeometrySet &geo) {
    Mesh *floor = nullptr;
    if (const Mesh *mesh = geo.get_mesh()) {
      floor = floor_generate(*mesh, p, &warning);
    }
    if (floor == nullptr) {
      if (const Curves *curves = geo.get_curves()) {
        Mesh *wire = bke::curve_to_wire_mesh(curves->geometry.wrap());
        floor = floor_generate(*wire, p, &warning);
        BKE_id_free(nullptr, wire);
      }
    }
    geo.remove(GeometryComponent::Type::Curve);
    geo.replace_mesh(floor);
  });
  if (!warning.empty()) {
    params.error_message_add(NodeWarningType::Warning, warning);
  }
  params.set_output("Floor"_ustr, std::move(geometry_set));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeFloorGenerator"_ustr);
  ntype.ui_name = "Floor Generator";
  ntype.ui_description =
      "Fill flat faces or closed curves with boards or tiles in a pattern, as the Floor Generator "
      "modifier (FloorGenerator in 3ds Max)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_floor_generator_cc
