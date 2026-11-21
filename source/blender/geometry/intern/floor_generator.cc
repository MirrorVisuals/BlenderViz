/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * BlenderViz: Floor Generator (see GEO_floor_generator.hh).
 *
 * The faces of the mesh are grouped by plane; each plane gets a 2D frame (X from the object's X
 * axis, the origin under the object's origin), so the pattern continues across the faces of a
 * floor. The pattern makes convex boards in that frame, which are cut by the triangles of the
 * faces and the pieces of one board joined again (so a board is cut only by the edges of the
 * floor, not by its triangulation). Each board is then extruded, with a bevel along its top.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>

#include "BLI_array.hh"
#include "BLI_function_ref.hh"
#include "BLI_map.hh"
#include "BLI_math_base.hh"
#include "BLI_math_constants.h"
#include "BLI_math_vector.hh"
#include "BLI_noise.hh"
#include "BLI_offset_indices.hh"
#include "BLI_polyfill_2d.h"
#include "BLI_set.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_mesh.h"
#include "BKE_mesh.hh"

#include "DNA_mesh_types.h"

#include "GEO_floor_generator.hh"

namespace blender::geometry {

namespace floor_gen {

using Poly = Vector<float2, 8>;

/** Stop adding boards past this, with a warning (a too small board size for the floor). */
static constexpr int64_t MAX_BOARDS = 4000000;

static float cross2(const float2 &a, const float2 &b)
{
  return a.x * b.y - a.y * b.x;
}

static float2 rotate2(const float2 &p, const float angle)
{
  const float c = std::cos(angle), s = std::sin(angle);
  return float2(p.x * c - p.y * s, p.x * s + p.y * c);
}

static float2 perp(const float2 &v)
{
  return float2(-v.y, v.x);
}

static float poly_area(const Span<float2> poly)
{
  float area = 0.0f;
  for (const int i : poly.index_range()) {
    area += cross2(poly[i], poly[(i + 1) % poly.size()]);
  }
  return area * 0.5f;
}

static Poly rect_poly(const float x0, const float y0, const float x1, const float y1)
{
  return Poly({float2(x0, y0), float2(x1, y0), float2(x1, y1), float2(x0, y1)});
}

/** Clip `subject` by the convex, counter-clockwise polygon `clipper` (Sutherland-Hodgman). */
static Poly clip_convex(const Span<float2> subject, const Span<float2> clipper)
{
  Poly out(subject);
  Poly in;
  for (const int i : clipper.index_range()) {
    if (out.is_empty()) {
      break;
    }
    const float2 a = clipper[i];
    const float2 edge = clipper[(i + 1) % clipper.size()] - a;
    in = std::move(out);
    out.clear();
    for (const int j : in.index_range()) {
      const float2 p = in[j];
      const float2 q = in[(j + 1) % in.size()];
      const float sp = cross2(edge, p - a);
      const float sq = cross2(edge, q - a);
      if (sp >= 0.0f) {
        out.append(p);
      }
      if ((sp >= 0.0f) != (sq >= 0.0f)) {
        const float t = sp / (sp - sq);
        out.append(p + (q - p) * t);
      }
    }
  }
  return out;
}

/**
 * Move each edge of a counter-clockwise polygon inwards: by `d_side` for the edges along `axis`
 * (the long sides of a board), by `d_end` for the others. Empty when the polygon disappears.
 */
static Poly inset_poly(const Span<float2> poly,
                       const float2 &axis,
                       const float d_side,
                       const float d_end)
{
  const int n = poly.size();
  if (d_side <= 0.0f && d_end <= 0.0f) {
    return Poly(poly);
  }
  Array<float2, 16> dirs(n);
  Array<float, 16> dist(n);
  for (const int i : IndexRange(n)) {
    const float2 e = poly[(i + 1) % n] - poly[i];
    const float len = math::length(e);
    dirs[i] = len > 1e-12f ? e / len : float2(1.0f, 0.0f);
    dist[i] = std::abs(math::dot(dirs[i], axis)) > 0.7071f ? d_side : d_end;
  }
  Poly out;
  for (const int i : IndexRange(n)) {
    const int prev = (i + n - 1) % n;
    /* Lines: point + normal * dist, along dir. */
    const float2 p0 = poly[prev] + perp(dirs[prev]) * dist[prev];
    const float2 p1 = poly[i] + perp(dirs[i]) * dist[i];
    const float denom = cross2(dirs[prev], dirs[i]);
    if (std::abs(denom) < 1e-6f) {
      out.append(p1);
      continue;
    }
    const float t = cross2(p1 - p0, dirs[i]) / denom;
    out.append(p0 + dirs[prev] * t);
  }
  /* The polygon must keep its orientation and every edge its direction. */
  if (poly_area(out) <= 1e-12f) {
    return {};
  }
  for (const int i : IndexRange(n)) {
    if (math::dot(out[(i + 1) % n] - out[i], dirs[i]) <= 0.0f) {
      return {};
    }
  }
  return out;
}

/** Spread: low for mostly `max`, high for mostly `min`. */
static float random_in_range(const FloorRandomRange &range, const float r)
{
  if (range.max == range.min) {
    return range.max;
  }
  const float k = std::exp((range.spread - 0.5f) * 4.0f);
  return range.min + (range.max - range.min) * std::pow(r, k);
}

/** A board of the pattern, in pattern space. */
struct PatternBoard {
  Poly poly;
  float2 center;
  /** Unit vector along the board's length. */
  float2 axis;
  /** Identifies the board, for its random values. */
  uint32_t key;
  /** Size of a full board, for UVs (a full board spans 0..1). */
  float uv_length;
  float uv_width;
};

struct PatternContext {
  const FloorGeneratorParams &params;
  float2 lo, hi;
  uint32_t seed;
  FunctionRef<void(PatternBoard &&)> emit;
  bool overflow = false;
  int64_t count = 0;

  float rand(const uint32_t a, const uint32_t b, const uint32_t channel) const
  {
    return noise::hash_to_float(a, b, channel ^ (seed * 0x9E3779B9u));
  }

  void add(Poly poly, const float2 axis, const uint32_t key, const float uv_l, const float uv_w)
  {
    if (overflow) {
      return;
    }
    if (++count > MAX_BOARDS) {
      overflow = true;
      return;
    }
    /* Clipped boards can repeat a point (a corner on the clip line), which the inset can't use. In
     * float, a few meters from the origin, the copies are up to about 1e-6 apart. */
    for (int i = poly.size() - 1; i >= 0 && poly.size() >= 3; i--) {
      if (math::distance_squared(poly[i], poly[(i + 1) % poly.size()]) < 1e-10f) {
        poly.remove(i);
      }
    }
    if (poly.size() < 3) {
      return;
    }
    float2 center(0.0f);
    for (const float2 &p : poly) {
      center += p;
    }
    center /= float(poly.size());
    this->emit(PatternBoard{std::move(poly), center, axis, key, uv_l, uv_w});
  }
};

static uint32_t key2(const int a, const int b, const uint32_t salt)
{
  return noise::hash(uint32_t(a), uint32_t(b), salt);
}

/** Positions of consecutive random sizes from 0, covering lo..hi (both directions). */
static void walk_sizes(const float lo,
                       const float hi,
                       const float start,
                       FunctionRef<float(int)> size_of,
                       FunctionRef<void(int, float, float)> fn)
{
  const int max_steps = 4000000;
  /* Forwards from start. */
  float x = start;
  for (int i = 0, steps = 0; x < hi && steps < max_steps; i++, steps++) {
    const float s = size_of(i);
    if (x + s > lo) {
      fn(i, x, x + s);
    }
    x += s;
  }
  /* Backwards. */
  x = start;
  for (int i = -1, steps = 0; x > lo && steps < max_steps; i--, steps++) {
    const float s = size_of(i);
    if (x - s < hi) {
      fn(i, x - s, x);
    }
    x -= s;
  }
}

static void pattern_rows(PatternContext &ctx, const bool stacked)
{
  const FloorGeneratorParams &p = ctx.params;
  const float scale = p.scale;
  const float min_size = 1e-4f;
  auto width_of = [&](const int row) {
    return std::max(min_size, random_in_range(p.width, ctx.rand(uint32_t(row), 0x51u, 1)) * scale);
  };
  walk_sizes(
      ctx.lo.y, ctx.hi.y, 0.0f, width_of, [&](const int row, const float y0, const float y1) {
        const float length_max = std::max(min_size, p.length.max * scale);
        const float start = stacked ?
                                0.0f :
                                (p.row_offset * row - std::floor(p.row_offset * row)) * length_max;
        auto length_of = [&](const int col) {
          /* Stacked: the same length in a column of every row, so the joints line up. */
          const uint32_t r = stacked ? 0u : uint32_t(row);
          return std::max(min_size,
                          random_in_range(p.length, ctx.rand(r, uint32_t(col), 2)) * scale);
        };
        walk_sizes(ctx.lo.x,
                   ctx.hi.x,
                   start,
                   length_of,
                   [&](const int col, const float x0, const float x1) {
                     ctx.add(rect_poly(x0, y0, x1, y1),
                             float2(1.0f, 0.0f),
                             key2(row, col, 0x1000),
                             length_max,
                             std::max(min_size, p.width.max * scale));
                   });
      });
}

static void pattern_herringbone(PatternContext &ctx)
{
  const FloorGeneratorParams &p = ctx.params;
  const int n = std::max(1, p.parallel_boards);
  const float W = std::max(1e-4f, p.width.max * p.scale);
  const float G = n * W;
  const float L = std::max(G, p.length.max * p.scale);
  /* Lattice: a = (G, G) and c = (L, -L); a tile is n boards along X and n along Y. */
  float amin = FLT_MAX, amax = -FLT_MAX, cmin = FLT_MAX, cmax = -FLT_MAX;
  for (const float2 corner :
       {ctx.lo, ctx.hi, float2(ctx.lo.x, ctx.hi.y), float2(ctx.hi.x, ctx.lo.y)})
  {
    const float a = (corner.x + corner.y) / (2.0f * G);
    const float c = (corner.x - corner.y) / (2.0f * L);
    amin = std::min(amin, a);
    amax = std::max(amax, a);
    cmin = std::min(cmin, c);
    cmax = std::max(cmax, c);
  }
  const float margin_a = (L + G) / G, margin_c = (L + G) / L;
  for (int a = int(std::floor(amin - margin_a)); a <= int(std::ceil(amax + margin_a)); a++) {
    for (int c = int(std::floor(cmin - margin_c)); c <= int(std::ceil(cmax + margin_c)); c++) {
      const float ox = a * G + c * L;
      const float oy = a * G - c * L;
      if (ox > ctx.hi.x || oy > ctx.hi.y || ox + L + G < ctx.lo.x || oy + L + G < ctx.lo.y) {
        continue;
      }
      for (int k = 0; k < n; k++) {
        ctx.add(rect_poly(ox, oy + k * W, ox + L, oy + (k + 1) * W),
                float2(1.0f, 0.0f),
                noise::hash(uint32_t(a), uint32_t(c), uint32_t(k * 2)),
                L,
                W);
        ctx.add(rect_poly(ox + k * W, oy + G, ox + (k + 1) * W, oy + G + L),
                float2(0.0f, 1.0f),
                noise::hash(uint32_t(a), uint32_t(c), uint32_t(k * 2 + 1)),
                L,
                W);
      }
      if (ctx.overflow) {
        return;
      }
    }
  }
}

static void pattern_chevron(PatternContext &ctx)
{
  const FloorGeneratorParams &p = ctx.params;
  const float angle = std::clamp(p.chevron_angle, 0.0873f, 1.4835f); /* 5..85 degrees. */
  const float W = std::max(1e-4f, p.width.max * p.scale);
  const float L = std::max(1e-4f, p.length.max * p.scale);
  const float t = std::tan(angle);
  const float c = L * std::cos(angle);
  const float h = W / std::cos(angle);
  const float2 dir_right(std::cos(angle), std::sin(angle));
  const float2 dir_left(std::cos(angle), -std::sin(angle));
  for (int m = int(std::floor(ctx.lo.x / (2.0f * c))) - 1;
       m <= int(std::ceil(ctx.hi.x / (2.0f * c)));
       m++)
  {
    const float x0 = m * 2.0f * c + c;
    for (int k = int(std::floor((ctx.lo.y - c * t) / h)) - 2;
         k <= int(std::ceil((ctx.hi.y + c * t) / h)) + 1;
         k++)
    {
      const float y = k * h;
      ctx.add(Poly({float2(x0, y),
                    float2(x0 + c, y + c * t),
                    float2(x0 + c, y + c * t + h),
                    float2(x0, y + h)}),
              dir_right,
              key2(m, k, 0x2000),
              L,
              W);
      ctx.add(Poly({float2(x0 - c, y + c * t),
                    float2(x0, y),
                    float2(x0, y + h),
                    float2(x0 - c, y + c * t + h)}),
              dir_left,
              key2(m, k, 0x2001),
              L,
              W);
      if (ctx.overflow) {
        return;
      }
    }
  }
}

static void basket_square(PatternContext &ctx,
                          const float x,
                          const float y,
                          const float size,
                          const int n,
                          const bool along_x,
                          const uint32_t key)
{
  const float w = size / n;
  for (int k = 0; k < n; k++) {
    if (along_x) {
      ctx.add(rect_poly(x, y + k * w, x + size, y + (k + 1) * w),
              float2(1.0f, 0.0f),
              noise::hash(key, uint32_t(k)),
              size,
              w);
    }
    else {
      ctx.add(rect_poly(x + k * w, y, x + (k + 1) * w, y + size),
              float2(0.0f, 1.0f),
              noise::hash(key, uint32_t(k)),
              size,
              w);
    }
  }
}

static void pattern_basket(PatternContext &ctx)
{
  const FloorGeneratorParams &p = ctx.params;
  const float S = std::max(1e-4f, p.length.max * p.scale);
  const int n = std::max(1, p.sub_boards);
  for (int i = int(std::floor(ctx.lo.x / S)); i <= int(std::floor(ctx.hi.x / S)); i++) {
    for (int j = int(std::floor(ctx.lo.y / S)); j <= int(std::floor(ctx.hi.y / S)); j++) {
      basket_square(ctx, i * S, j * S, S, n, ((i + j) & 1) == 0, key2(i, j, 0x3000));
      if (ctx.overflow) {
        return;
      }
    }
  }
}

static void pattern_hexagon(PatternContext &ctx)
{
  const FloorGeneratorParams &p = ctx.params;
  const float R = std::max(1e-4f, p.hex_radius * p.scale);
  const float dx = std::sqrt(3.0f) * R;
  const float dy = 1.5f * R;
  for (int j = int(std::floor(ctx.lo.y / dy)) - 1; j <= int(std::ceil(ctx.hi.y / dy)) + 1; j++) {
    for (int i = int(std::floor(ctx.lo.x / dx)) - 1; i <= int(std::ceil(ctx.hi.x / dx)) + 1; i++) {
      const float cx = i * dx + ((j & 1) ? dx * 0.5f : 0.0f);
      const float cy = j * dy;
      Poly poly;
      for (int q = 0; q < 6; q++) {
        const float a = float(M_PI) / 6.0f + q * float(M_PI) / 3.0f;
        poly.append(float2(cx + R * std::cos(a), cy + R * std::sin(a)));
      }
      ctx.add(std::move(poly), float2(1.0f, 0.0f), key2(i, j, 0x4000), 2.0f * R, 2.0f * R);
      if (ctx.overflow) {
        return;
      }
    }
  }
}

static void pattern_versailles(PatternContext &ctx)
{
  const FloorGeneratorParams &p = ctx.params;
  const float P = std::max(1e-3f, p.panel_size * p.scale);
  const float W = std::clamp(p.width.max * p.scale, 1e-4f, P * 0.25f);
  const int n = std::max(1, p.sub_boards);
  /* The diagonal basket weave inside the frame: squares of half the inner size. */
  const float inner = P - 2.0f * W;
  const float S = inner * 0.5f;
  const float r = float(M_SQRT1_2);
  for (int i = int(std::floor(ctx.lo.x / P)); i <= int(std::floor(ctx.hi.x / P)); i++) {
    for (int j = int(std::floor(ctx.lo.y / P)); j <= int(std::floor(ctx.hi.y / P)); j++) {
      const float x = i * P, y = j * P;
      const uint32_t panel = key2(i, j, 0x5000);
      /* Mitred frame. */
      ctx.add(
          Poly({float2(x, y), float2(x + P, y), float2(x + P - W, y + W), float2(x + W, y + W)}),
          float2(1.0f, 0.0f),
          noise::hash(panel, 1u),
          P,
          W);
      ctx.add(Poly({float2(x + P, y),
                    float2(x + P, y + P),
                    float2(x + P - W, y + P - W),
                    float2(x + P - W, y + W)}),
              float2(0.0f, 1.0f),
              noise::hash(panel, 2u),
              P,
              W);
      ctx.add(Poly({float2(x + P, y + P),
                    float2(x, y + P),
                    float2(x + W, y + P - W),
                    float2(x + P - W, y + P - W)}),
              float2(1.0f, 0.0f),
              noise::hash(panel, 3u),
              P,
              W);
      ctx.add(
          Poly({float2(x, y + P), float2(x, y), float2(x + W, y + W), float2(x + W, y + P - W)}),
          float2(0.0f, 1.0f),
          noise::hash(panel, 4u),
          P,
          W);
      const Poly inner_square = rect_poly(x + W, y + W, x + P - W, y + P - W);
      const float2 center(x + P * 0.5f, y + P * 0.5f);
      auto to_panel = [&](const float u, const float v) {
        return center + float2((u - v) * r, (u + v) * r);
      };
      const float w = S / n;
      for (int a = -2; a < 2; a++) {
        for (int c = -2; c < 2; c++) {
          const bool along_u = ((a + c) & 1) == 0;
          for (int k = 0; k < n; k++) {
            Poly strip;
            if (along_u) {
              strip = {to_panel(a * S, c * S + k * w),
                       to_panel((a + 1) * S, c * S + k * w),
                       to_panel((a + 1) * S, c * S + (k + 1) * w),
                       to_panel(a * S, c * S + (k + 1) * w)};
            }
            else {
              strip = {to_panel(a * S + k * w, c * S),
                       to_panel(a * S + (k + 1) * w, c * S),
                       to_panel(a * S + (k + 1) * w, (c + 1) * S),
                       to_panel(a * S + k * w, (c + 1) * S)};
            }
            Poly piece = clip_convex(strip, inner_square);
            if (piece.size() >= 3 && poly_area(piece) > 1e-10f) {
              const float2 axis = along_u ? float2(r, r) : float2(-r, r);
              ctx.add(std::move(piece),
                      axis,
                      noise::hash(panel, uint32_t(((a + 2) * 4 + (c + 2)) * 16 + k + 8)),
                      S,
                      w);
            }
          }
        }
      }
      if (ctx.overflow) {
        return;
      }
    }
  }
}

/** The faces of one plane, as triangles in its frame. */
struct PlaneGroup {
  float3 origin, axis_x, axis_y, normal;
  Vector<std::array<float2, 3>> tris;
  Vector<int> tri_faces;
  float2 lo = float2(FLT_MAX), hi = float2(-FLT_MAX);
};

/** A finished board piece in the frame of its plane. */
struct OutBoard {
  Poly poly;
  int group;
  float2 center, axis;
  float uv_length, uv_width;
  uint32_t key;
  int src_face;
};

/** Spatial grid over the triangles of a plane group. */
struct TriGrid {
  float2 lo;
  float2 cell;
  int nx = 1, ny = 1;
  Vector<Vector<int>> cells;

  void build(const PlaneGroup &group)
  {
    lo = group.lo;
    const float2 size = math::max(group.hi - group.lo, float2(1e-6f));
    const int res = std::clamp(int(std::sqrt(float(group.tris.size()))), 1, 256);
    nx = ny = res;
    cell = size / float(res);
    cells.resize(nx * ny);
    for (const int t : group.tris.index_range()) {
      float2 tlo(FLT_MAX), thi(-FLT_MAX);
      for (const float2 &p : group.tris[t]) {
        tlo = math::min(tlo, p);
        thi = math::max(thi, p);
      }
      const int2 c0 = cell_of(tlo), c1 = cell_of(thi);
      for (int y = c0.y; y <= c1.y; y++) {
        for (int x = c0.x; x <= c1.x; x++) {
          cells[y * nx + x].append(t);
        }
      }
    }
  }
  int2 cell_of(const float2 &p) const
  {
    return int2(std::clamp(int((p.x - lo.x) / cell.x), 0, nx - 1),
                std::clamp(int((p.y - lo.y) / cell.y), 0, ny - 1));
  }
};

static bool point_in_tri(const float2 &p, const std::array<float2, 3> &tri)
{
  return cross2(tri[1] - tri[0], p - tri[0]) >= 0.0f &&
         cross2(tri[2] - tri[1], p - tri[1]) >= 0.0f &&
         cross2(tri[0] - tri[2], p - tri[2]) >= 0.0f;
}

/** Remove points closer than `weld` and points on a straight line. */
static void clean_poly(Poly &poly, const float weld)
{
  const float weld_sq = weld * weld;
  bool changed = true;
  while (changed && poly.size() >= 3) {
    changed = false;
    for (int i = 0; i < poly.size() && poly.size() >= 3; i++) {
      const int n = poly.size();
      const float2 &a = poly[(i + n - 1) % n];
      const float2 &b = poly[i];
      const float2 &c = poly[(i + 1) % n];
      const float2 ab = b - a, bc = c - b;
      const float len = math::length(ab) * math::length(bc);
      if (math::length_squared(c - b) <= weld_sq ||
          (len > 0.0f && std::abs(cross2(ab, bc)) <= 1e-5f * len && math::dot(ab, bc) > 0.0f))
      {
        poly.remove(i);
        changed = true;
        i--;
      }
    }
  }
}

/**
 * Join the pieces of one board (cut by neighboring triangles) into its outlines. Returns false
 * when they don't give simple outlines (a hole): then the pieces are used as they are.
 */
static bool merge_pieces(const Span<Poly> pieces, const float eps, Vector<Poly> &r_polys)
{
  Vector<float2> verts;
  const float eps_sq = eps * eps;
  auto vert_id = [&](const float2 &p) {
    for (const int i : verts.index_range()) {
      if (math::distance_squared(verts[i], p) <= eps_sq) {
        return i;
      }
    }
    verts.append(p);
    return int(verts.size() - 1);
  };
  Map<int64_t, int> edges;
  auto edge_key = [](const int a, const int b) {
    return (int64_t(a) << 32) | int64_t(uint32_t(b));
  };
  for (const Poly &piece : pieces) {
    Vector<int, 8> ids;
    for (const float2 &p : piece) {
      const int id = vert_id(p);
      if (ids.is_empty() || ids.last() != id) {
        ids.append(id);
      }
    }
    while (ids.size() > 1 && ids.first() == ids.last()) {
      ids.remove_last();
    }
    for (const int i : ids.index_range()) {
      const int a = ids[i], b = ids[(i + 1) % ids.size()];
      if (a == b) {
        continue;
      }
      if (edges.remove(edge_key(b, a))) {
        continue;
      }
      edges.add(edge_key(a, b), 1);
    }
  }
  Map<int, int> next;
  for (const int64_t key : edges.keys()) {
    const int a = int(key >> 32), b = int(uint32_t(key));
    if (!next.add(a, b)) {
      return false; /* Two outlines touching at a corner. */
    }
  }
  Set<int> used;
  for (const auto item : next.items()) {
    if (used.contains(item.key)) {
      continue;
    }
    Poly poly;
    int v = item.key;
    for (int guard = 0; guard <= next.size(); guard++) {
      if (!used.add(v)) {
        break;
      }
      poly.append(verts[v]);
      const int *n = next.lookup_ptr(v);
      if (n == nullptr) {
        return false;
      }
      v = *n;
    }
    if (v != item.key || poly_area(poly) <= 0.0f) {
      return false; /* Open, or a hole. */
    }
    r_polys.append(std::move(poly));
  }
  return true;
}

static int64_t plane_key(const float3 &n, const float d)
{
  const int64_t qn[3] = {int64_t(std::round(n.x * 2000.0f)),
                         int64_t(std::round(n.y * 2000.0f)),
                         int64_t(std::round(n.z * 2000.0f))};
  const int64_t qd = int64_t(std::round(d * 10000.0f));
  return (((qn[0] * 4001 + qn[1]) * 4001 + qn[2]) * 1000003) ^ qd;
}

/** The closed loops of edges of a mesh without faces (vertices with two edges each). */
static Vector<Vector<int>> closed_edge_loops(const Mesh &mesh)
{
  const Span<int2> edges = mesh.edges();
  Array<int> count(mesh.verts_num, 0);
  Array<int2> links(mesh.verts_num, int2(-1));
  for (const int2 &e : edges) {
    for (const int k : IndexRange(2)) {
      const int v = e[k], other = e[1 - k];
      if (count[v] < 2) {
        links[v][count[v]] = other;
      }
      count[v]++;
    }
  }
  Vector<Vector<int>> loops;
  Array<bool> done(mesh.verts_num, false);
  for (const int start : IndexRange(mesh.verts_num)) {
    if (done[start] || count[start] != 2) {
      continue;
    }
    Vector<int> loop;
    int prev = -1, v = start;
    bool closed = false;
    while (true) {
      if (count[v] != 2 || done[v]) {
        break;
      }
      done[v] = true;
      loop.append(v);
      const int next = links[v][0] != prev ? links[v][0] : links[v][1];
      prev = v;
      v = next;
      if (v == start) {
        closed = true;
        break;
      }
    }
    if (closed && loop.size() >= 3) {
      loops.append(std::move(loop));
    }
  }
  return loops;
}

static PlaneGroup make_group_frame(const float3 &normal, const float3 &point_on_plane)
{
  PlaneGroup group;
  group.normal = normal;
  float3 x = float3(1.0f, 0.0f, 0.0f) - normal * normal.x;
  if (math::length_squared(x) < 1e-6f) {
    x = float3(0.0f, 1.0f, 0.0f) - normal * normal.y;
  }
  group.axis_x = math::normalize(x);
  group.axis_y = math::cross(normal, group.axis_x);
  /* Under the object's origin. */
  group.origin = normal * math::dot(point_on_plane, normal);
  return group;
}

}  // namespace floor_gen

Mesh *floor_generate(const Mesh &src, const FloorGeneratorParams &params, std::string *r_warning)
{
  using namespace floor_gen;
  const Span<float3> positions = src.vert_positions();
  const Span<int> corner_verts = src.corner_verts();
  const Span<int3> corner_tris = src.corner_tris();
  const Span<int> tri_faces = src.corner_tri_faces();
  const Span<float3> face_normals = src.face_normals_true();

  /* Group the faces by plane. */
  Vector<PlaneGroup> groups;
  Map<int64_t, int> group_of_plane;
  Array<int> face_group(src.faces_num, -1);
  const OffsetIndices<int> faces = src.faces();
  for (const int face : faces.index_range()) {
    const float3 n = face_normals[face];
    if (math::length_squared(n) < 0.5f) {
      continue;
    }
    const float d = math::dot(n, positions[corner_verts[faces[face].first()]]);
    face_group[face] = group_of_plane.lookup_or_add_cb(plane_key(n, d), [&]() {
      groups.append(make_group_frame(n, n * d));
      return int(groups.size() - 1);
    });
  }
  for (const int t : corner_tris.index_range()) {
    const int face = tri_faces[t];
    const int g = face_group[face];
    if (g < 0) {
      continue;
    }
    PlaneGroup &group = groups[g];
    std::array<float2, 3> tri;
    for (const int k : IndexRange(3)) {
      const float3 rel = positions[corner_verts[corner_tris[t][k]]] - group.origin;
      tri[k] = float2(math::dot(rel, group.axis_x), math::dot(rel, group.axis_y));
      group.lo = math::min(group.lo, tri[k]);
      group.hi = math::max(group.hi, tri[k]);
    }
    const float area = cross2(tri[1] - tri[0], tri[2] - tri[0]);
    if (std::abs(area) < 1e-14f) {
      continue;
    }
    if (area < 0.0f) {
      std::swap(tri[1], tri[2]);
    }
    group.tris.append(tri);
    group.tri_faces.append(face);
  }

  /* No faces: fill the closed loops of edges (a curve outline, or edges drawn in a mesh). */
  if (src.faces_num == 0) {
    for (const Vector<int> &loop : closed_edge_loops(src)) {
      float3 n(0.0f);
      for (const int i : loop.index_range()) {
        const float3 &a = positions[loop[i]];
        const float3 &b = positions[loop[(i + 1) % loop.size()]];
        n += float3(
            (a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
      }
      if (math::length_squared(n) < 1e-20f) {
        continue;
      }
      n = math::normalize(n);
      /* A floor faces up, whichever way the outline was drawn. */
      if (n.z < -0.5f) {
        n = -n;
      }
      const float d = math::dot(n, positions[loop[0]]);
      const int g = group_of_plane.lookup_or_add_cb(plane_key(n, d), [&]() {
        groups.append(make_group_frame(n, n * d));
        return int(groups.size() - 1);
      });
      PlaneGroup &group = groups[g];
      Array<float2> coords(loop.size());
      for (const int i : loop.index_range()) {
        const float3 rel = positions[loop[i]] - group.origin;
        coords[i] = float2(math::dot(rel, group.axis_x), math::dot(rel, group.axis_y));
        group.lo = math::min(group.lo, coords[i]);
        group.hi = math::max(group.hi, coords[i]);
      }
      Array<uint3> tris(loop.size() - 2);
      BLI_polyfill_calc(reinterpret_cast<const float (*)[2]>(coords.data()),
                        uint(loop.size()),
                        0,
                        reinterpret_cast<uint(*)[3]>(tris.data()));
      for (const uint3 &t : tris) {
        std::array<float2, 3> tri = {coords[t[0]], coords[t[1]], coords[t[2]]};
        const float area = cross2(tri[1] - tri[0], tri[2] - tri[0]);
        if (std::abs(area) < 1e-14f) {
          continue;
        }
        if (area < 0.0f) {
          std::swap(tri[1], tri[2]);
        }
        group.tris.append(tri);
        group.tri_faces.append(-1);
      }
    }
  }

  /* Rotation of the pattern: Direction, and the turn that lays the zigzag of Herringbone and
   * Chevron along X. */
  float base_angle = 0.0f;
  if (params.pattern == FloorPattern::Herringbone) {
    base_angle = -float(M_PI) * 0.25f;
  }
  else if (params.pattern == FloorPattern::Chevron) {
    base_angle = -float(M_PI) * 0.5f;
  }
  const float angle = params.direction + base_angle;
  const float scale = std::max(params.scale, 1e-4f);
  const float grout_l = std::max(0.0f, params.grout_length) * scale;
  const float grout_w = std::max(0.0f, params.grout_width) * scale;
  const float margin_l = grout_l * 0.5f + std::max(0.0f, params.overlap);
  const float margin_w = grout_w * 0.5f + std::max(0.0f, params.overlap);
  const uint32_t seed = uint32_t(params.seed);
  const float weld = std::max(0.0f, params.weld);

  Vector<OutBoard> out;
  bool overflow = false;

  for (const int g : groups.index_range()) {
    PlaneGroup &group = groups[g];
    if (group.tris.is_empty()) {
      continue;
    }
    TriGrid grid;
    grid.build(group);
    Array<int> tri_stamp(group.tris.size(), -1);
    int stamp = 0;
    const float eps = std::max(1e-7f, 1e-6f * math::reduce_max(group.hi - group.lo));

    /* The plane's area in pattern space. */
    float2 plo(FLT_MAX), phi(-FLT_MAX);
    for (const float2 corner :
         {group.lo, group.hi, float2(group.lo.x, group.hi.y), float2(group.hi.x, group.lo.y)})
    {
      const float2 pp = rotate2(corner - params.offset, -angle);
      plo = math::min(plo, pp);
      phi = math::max(phi, pp);
    }

    auto handle_board = [&](PatternBoard &&board) {
      auto rnd = [&](const uint32_t channel) {
        return noise::hash_to_float(board.key, channel, seed);
      };
      if (params.missing > 0.0f && rnd(11) < params.missing) {
        return;
      }
      /* Grout. */
      Poly poly = inset_poly(board.poly, board.axis, grout_w * 0.5f, grout_l * 0.5f);
      if (poly.size() < 3) {
        return;
      }
      /* Random rotation and offset, kept within the grout (and the overlap). */
      float extent = 0.0f;
      for (const float2 &pt : poly) {
        extent = std::max(extent, std::abs(math::dot(pt - board.center, board.axis)));
      }
      float rot = random_in_range(params.rotation, rnd(1)) * (rnd(2) < 0.5f ? -1.0f : 1.0f);
      if (extent > 1e-9f) {
        const float limit = std::asin(std::min(1.0f, margin_w / extent));
        rot = std::clamp(rot, -limit, limit);
      }
      float ox = random_in_range(params.offset_x, rnd(3)) * (rnd(4) < 0.5f ? -1.0f : 1.0f);
      float oy = random_in_range(params.offset_y, rnd(5)) * (rnd(6) < 0.5f ? -1.0f : 1.0f);
      ox = std::clamp(ox, -margin_l, margin_l);
      oy = std::clamp(oy, -margin_w, margin_w);
      const float2 shift = board.axis * ox + perp(board.axis) * oy;
      for (float2 &pt : poly) {
        pt = board.center + rotate2(pt - board.center, rot) + shift;
      }
      float2 axis = rotate2(board.axis, rot);
      float2 center = board.center + shift;

      /* To the plane's frame. */
      for (float2 &pt : poly) {
        pt = rotate2(pt, angle) + params.offset;
      }
      axis = rotate2(axis, angle);
      center = rotate2(center, angle) + params.offset;

      float2 blo(FLT_MAX), bhi(-FLT_MAX);
      for (const float2 &pt : poly) {
        blo = math::min(blo, pt);
        bhi = math::max(bhi, pt);
      }
      if (blo.x > group.hi.x || blo.y > group.hi.y || bhi.x < group.lo.x || bhi.y < group.lo.y) {
        return;
      }

      /* Cut by the triangles of the faces. */
      Vector<Poly> pieces;
      Vector<int> piece_faces;
      stamp++;
      bool whole = false;
      const int2 c0 = grid.cell_of(blo), c1 = grid.cell_of(bhi);
      for (int cy = c0.y; cy <= c1.y && !whole; cy++) {
        for (int cx = c0.x; cx <= c1.x && !whole; cx++) {
          for (const int t : grid.cells[cy * grid.nx + cx]) {
            if (tri_stamp[t] == stamp) {
              continue;
            }
            tri_stamp[t] = stamp;
            const std::array<float2, 3> &tri = group.tris[t];
            float2 tlo = math::min(math::min(tri[0], tri[1]), tri[2]);
            float2 thi = math::max(math::max(tri[0], tri[1]), tri[2]);
            if (tlo.x > bhi.x || tlo.y > bhi.y || thi.x < blo.x || thi.y < blo.y) {
              continue;
            }
            bool inside = true;
            for (const float2 &pt : poly) {
              if (!point_in_tri(pt, tri)) {
                inside = false;
                break;
              }
            }
            if (inside) {
              pieces.clear();
              piece_faces.clear();
              pieces.append(poly);
              piece_faces.append(group.tri_faces[t]);
              whole = true;
              break;
            }
            Poly piece = clip_convex(poly, Span<float2>(tri.data(), 3));
            if (piece.size() >= 3 && poly_area(piece) > eps * eps) {
              pieces.append(std::move(piece));
              piece_faces.append(group.tri_faces[t]);
            }
          }
        }
      }
      if (pieces.is_empty()) {
        return;
      }
      Vector<Poly> polys;
      if (pieces.size() == 1 || !merge_pieces(pieces, eps * 4.0f, polys)) {
        polys = std::move(pieces);
      }
      for (Poly &piece : polys) {
        clean_poly(piece, weld);
        if (piece.size() < 3 || poly_area(piece) <= std::max(weld * weld, eps * eps)) {
          continue;
        }
        out.append(OutBoard{std::move(piece),
                            g,
                            center,
                            axis,
                            board.uv_length,
                            board.uv_width,
                            board.key,
                            piece_faces.first()});
      }
    };

    PatternContext ctx{params, plo, phi, seed, handle_board};
    switch (params.pattern) {
      case FloorPattern::Standard:
        pattern_rows(ctx, false);
        break;
      case FloorPattern::Stacked:
        pattern_rows(ctx, true);
        break;
      case FloorPattern::Herringbone:
        pattern_herringbone(ctx);
        break;
      case FloorPattern::Chevron:
        pattern_chevron(ctx);
        break;
      case FloorPattern::BasketWeave:
        pattern_basket(ctx);
        break;
      case FloorPattern::Hexagon:
        pattern_hexagon(ctx);
        break;
      case FloorPattern::Versailles:
        pattern_versailles(ctx);
        break;
    }
    overflow |= ctx.overflow;
  }

  if (overflow && r_warning) {
    *r_warning = "Too many boards for the size of the floor, the floor is incomplete";
  }
  if (out.is_empty()) {
    return nullptr;
  }

  /* Build the mesh. */
  const float height = std::max(0.0f, params.height);
  const float bevel_h = std::clamp(params.bevel_height, 0.0f, height);
  const float bevel_w = std::max(0.0f, params.bevel_width);
  const bool use_bevel = height > 0.0f && bevel_h > 0.0f && bevel_w > 0.0f;

  Vector<float3> verts;
  Vector<int> face_sizes;
  Vector<int> corners;
  Vector<float2> uvs;
  Vector<int> board_ids;
  Vector<float> board_randoms;
  Vector<int> materials;
  const VArray<int> src_materials = *src.attributes().lookup_or_default<int>(
      "material_index", bke::AttrDomain::Face, 0);

  for (const int b : out.index_range()) {
    const OutBoard &board = out[b];
    const PlaneGroup &group = groups[board.group];
    auto rnd = [&](const uint32_t channel) {
      return noise::hash_to_float(board.key, channel, seed);
    };
    const float board_random = rnd(20);
    const float2 side_axis = perp(board.axis);
    /* Tilt: a random direction, the top plane turned around it. */
    const float tilt = random_in_range(params.tilt, rnd(7));
    const float tilt_dir_angle = rnd(8) * 2.0f * float(M_PI);
    const float2 tilt_dir(std::cos(tilt_dir_angle), std::sin(tilt_dir_angle));
    const float tilt_slope = std::tan(std::min(tilt, 1.4f));
    const bool flip_u = params.random_flip_u && rnd(9) < 0.5f;
    const bool flip_v = params.random_flip_v && rnd(10) < 0.5f;
    const float off_u = params.random_offset_u ? rnd(12) : 0.0f;
    const float off_v = params.random_offset_v ? rnd(13) : 0.0f;
    int material = board.src_face >= 0 ? src_materials[board.src_face] : 0;
    if (params.material_count > 1) {
      material = std::min(int(rnd(14) * params.material_count), params.material_count - 1);
    }

    auto to_3d = [&](const float2 &p, float z) {
      if (z > 0.0f && tilt > 0.0f) {
        z += math::dot(p - board.center, tilt_dir) * tilt_slope;
      }
      return group.origin + group.axis_x * p.x + group.axis_y * p.y + group.normal * z;
    };
    auto top_uv = [&](const float2 &p) {
      float u = 0.5f + math::dot(p - board.center, board.axis) / board.uv_length;
      float v = 0.5f + math::dot(p - board.center, side_axis) / board.uv_width;
      if (flip_u) {
        u = 1.0f - u;
      }
      if (flip_v) {
        v = 1.0f - v;
      }
      return float2(u + off_u, v + off_v);
    };
    auto add_face = [&](const Span<int> face_verts, const Span<float2> face_uvs) {
      face_sizes.append(face_verts.size());
      corners.extend(face_verts);
      uvs.extend(face_uvs);
      board_ids.append(b);
      board_randoms.append(board_random);
      materials.append(material);
    };

    const Span<float2> outline = board.poly;
    const int n = outline.size();
    Poly top_outline;
    if (use_bevel) {
      top_outline = inset_poly(outline, board.axis, bevel_w, bevel_w);
    }
    const bool bevel_here = use_bevel && top_outline.size() == n;
    if (!bevel_here) {
      top_outline = Poly(outline);
    }

    const int top_start = verts.size();
    for (const float2 &p : top_outline) {
      verts.append(to_3d(p, height));
    }
    {
      Vector<int, 8> ids;
      Vector<float2, 8> face_uvs;
      for (const int i : IndexRange(n)) {
        ids.append(top_start + i);
        face_uvs.append(top_uv(top_outline[i]));
      }
      add_face(ids, face_uvs);
    }
    if (height <= 0.0f) {
      continue;
    }
    /* Sides, and the bevel between the sides and the top. */
    const int bottom_start = verts.size();
    for (const float2 &p : outline) {
      verts.append(to_3d(p, 0.0f));
    }
    int ring_start = top_start;
    float ring_z = height;
    if (bevel_here) {
      ring_start = verts.size();
      ring_z = height - bevel_h;
      for (const float2 &p : outline) {
        verts.append(to_3d(p, ring_z));
      }
    }
    const Span<float2> ring_outline = bevel_here ? outline : top_outline.as_span();
    for (const int i : IndexRange(n)) {
      const int j = (i + 1) % n;
      const float2 edge = outline[j] - outline[i];
      const float len = math::length(edge);
      const float2 dir = len > 0.0f ? edge / len : board.axis;
      const float along = math::dot(outline[i] - board.center, dir);
      auto side_uv = [&](const float s, const float z) {
        return float2(0.5f + (along + s) / board.uv_length + off_u, z / board.uv_width + off_v);
      };
      const std::array<int, 4> side = {
          bottom_start + i, bottom_start + j, ring_start + j, ring_start + i};
      const std::array<float2, 4> side_uvs = {
          side_uv(0.0f, 0.0f), side_uv(len, 0.0f), side_uv(len, ring_z), side_uv(0.0f, ring_z)};
      add_face(Span<int>(side), Span<float2>(side_uvs));
      if (bevel_here) {
        const std::array<int, 4> bevel = {
            ring_start + i, ring_start + j, top_start + j, top_start + i};
        const std::array<float2, 4> bevel_uvs = {top_uv(ring_outline[i]),
                                                 top_uv(ring_outline[j]),
                                                 top_uv(top_outline[j]),
                                                 top_uv(top_outline[i])};
        add_face(Span<int>(bevel), Span<float2>(bevel_uvs));
      }
    }
  }

  Mesh *mesh = BKE_mesh_new_nomain(verts.size(), 0, face_sizes.size(), corners.size());
  BKE_mesh_copy_parameters_for_eval(mesh, &src);
  mesh->vert_positions_for_write().copy_from(verts);
  MutableSpan<int> offsets = mesh->face_offsets_for_write();
  int offset = 0;
  for (const int f : face_sizes.index_range()) {
    offsets[f] = offset;
    offset += face_sizes[f];
  }
  offsets.last() = offset;
  mesh->corner_verts_for_write().copy_from(corners);
  bke::mesh_calc_edges(*mesh, false, false);
  bke::mesh_smooth_set(*mesh, false);

  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  if (!params.uv_map_name.empty()) {
    bke::SpanAttributeWriter uv = attributes.lookup_or_add_for_write_only_span<float2>(
        params.uv_map_name, bke::AttrDomain::Corner);
    uv.span.copy_from(uvs);
    uv.finish();
  }
  bke::SpanAttributeWriter ids = attributes.lookup_or_add_for_write_only_span<int>(
      "board_id", bke::AttrDomain::Face);
  ids.span.copy_from(board_ids);
  ids.finish();
  bke::SpanAttributeWriter randoms = attributes.lookup_or_add_for_write_only_span<float>(
      "board_random", bke::AttrDomain::Face);
  randoms.span.copy_from(board_randoms);
  randoms.finish();
  bke::SpanAttributeWriter material_index = attributes.lookup_or_add_for_write_only_span<int>(
      "material_index", bke::AttrDomain::Face);
  material_index.span.copy_from(materials);
  material_index.finish();
  return mesh;
}

}  // namespace blender::geometry
