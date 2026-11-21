/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

/** \file
 * \ingroup geo
 *
 * BlenderViz: Floor Generator, as the FloorGenerator plugin of 3ds Max. Fills the flat faces of a
 * mesh with boards or tiles in a pattern, cut to the edges of the faces. Used by the Floor
 * Generator modifier and the Floor Generator node.
 */

#include <string>

#include "BLI_math_vector_types.hh"

struct Mesh;

namespace blender::geometry {

enum class FloorPattern {
  /** Rows of boards, the joints of each row shifted by the Row Offset (running bond). */
  Standard = 0,
  /** Rows and columns, the joints lined up (tiles). */
  Stacked = 1,
  /** Boards at right angles in a zigzag; Parallel Boards 2 is a double herringbone. */
  Herringbone = 2,
  /** Boards cut at an angle, meeting in straight lines. */
  Chevron = 3,
  /** Squares of a few parallel boards, every other square turned 90 degrees. */
  BasketWeave = 4,
  /** Hexagonal tiles. */
  Hexagon = 5,
  /** Square panels: a mitred frame around a diagonal basket weave. */
  Versailles = 6,
};

/** A random amount: `min` to `max`, `spread` (0..1) low for mostly `max`, high for mostly `min`.
 */
struct FloorRandomRange {
  float min = 0.0f;
  float max = 0.0f;
  float spread = 0.5f;
};

struct FloorGeneratorParams {
  FloorPattern pattern = FloorPattern::Standard;

  /** Board length (along the boards) and width. The min values vary Standard / Stacked boards. */
  FloorRandomRange length = {1.2f, 1.2f, 0.5f};
  FloorRandomRange width = {0.2f, 0.2f, 0.5f};
  /** Gap between the ends of boards and between their sides. */
  float grout_length = 0.002f;
  float grout_width = 0.002f;
  /** Shift of each row of Standard boards, as a fraction of the length. */
  float row_offset = 0.33f;

  /** Board thickness (0: flat faces) and the bevel of the top edges. */
  float height = 0.01f;
  float bevel_height = 0.001f;
  float bevel_width = 0.001f;

  /** Herringbone: boards side by side in each arm. */
  int parallel_boards = 1;
  /** Chevron: angle of the boards to the cut line (radians). */
  float chevron_angle = 0.785398f;
  /** Basket Weave and Versailles: boards in each square. */
  int sub_boards = 3;
  /** Hexagon: circumradius of a tile. */
  float hex_radius = 0.15f;
  /** Versailles: size of a panel (the frame is a board width). */
  float panel_size = 1.0f;

  /** Random variation per board: rotation (radians), offset along and across, tilt (radians). */
  FloorRandomRange rotation;
  FloorRandomRange offset_x;
  FloorRandomRange offset_y;
  FloorRandomRange tilt;
  /** How far boards may push into the grout and past each other with rotation and offset. */
  float overlap = 0.0f;
  /** Fraction of the boards left out (0..1), for damaged floors. */
  float missing = 0.0f;

  /** The whole pattern: scale, rotation (radians), position in the plane and random seed. */
  float scale = 1.0f;
  float direction = 0.0f;
  float2 offset = float2(0.0f);
  int seed = 0;

  /** Board corners closer than this are merged, and smaller board pieces are left out. */
  float weld = 0.001f;

  /** UVs per board, a full size board spans 0..1. */
  bool random_flip_u = false;
  bool random_flip_v = false;
  bool random_offset_u = false;
  bool random_offset_v = false;

  /** Random material index per board, from 0 to count - 1 (1: the material of the face). */
  int material_count = 1;

  /** Name of the UV map written (empty: none). */
  std::string uv_map_name = "UVMap";
};

/**
 * Make the floor for the flat faces of `src` (faces in the same plane share the pattern). Writes
 * the UV map, `board_id` (int per face, the board's index), `board_random` (float per face, a
 * random value per board) and `material_index`. Returns null when there is nothing to fill.
 * `r_warning` is set for an excessive number of boards.
 */
Mesh *floor_generate(const Mesh &src, const FloorGeneratorParams &params, std::string *r_warning);

}  // namespace blender::geometry
