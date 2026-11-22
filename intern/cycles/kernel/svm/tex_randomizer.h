/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* BlenderViz: Texture Randomizer, as the V-Ray UVW Randomizer. Random offset, rotation, scale and
 * flip of texture coordinates per object, object index, mesh element, face (material index),
 * particle, or UV tile (stochastic tiling). With Tile Blend, the Image Texture after it is sampled
 * once per neighbouring tile (see #ImageTextureNode::expand) and the results are mixed with the
 * Weight output, so the tile edges don't show. */

#pragma once

#include "kernel/svm/node_types.h"
#include "kernel/svm/util.h"

#include "util/hash.h"

CCL_NAMESPACE_BEGIN

/* A value from `from` to `to`: continuous with no steps, else one of `steps` values. A range of a
 * full turn (rotation) leaves out `to`, it is the same as `from`. */
ccl_device_inline float tex_randomizer_value(
    const float from, const float to, const float steps, const float r, const bool wrap)
{
  const int n = int(steps);
  if (n <= 0) {
    return from + (to - from) * r;
  }
  if (n == 1) {
    return from;
  }
  const int i = min(int(r * float(n)), n - 1);
  const float divisions = wrap ? float(n) : float(n - 1);
  return from + (to - from) * float(i) / divisions;
}

ccl_device_inline uint tex_randomizer_mix(const uint h, const float value)
{
  return hash_uint2(h, __float_as_uint(value));
}

/* Weight of the neighbouring tile along one axis, 0 inside the tile and 0.5 on its edge. */
ccl_device_inline float tex_randomizer_edge_weight(const float f, const float blend)
{
  if (blend <= 0.0f) {
    return 0.0f;
  }
  const float distance = min(f, 1.0f - f);
  const float s = saturatef(1.0f - distance / blend);
  return 0.5f * s * s * (3.0f - 2.0f * s);
}

/* Apply `M * (p - pivot) + pivot + offset` to the UV, the derivatives only get `M`. */
ccl_device_inline float3 tex_randomizer_apply(const float3 co,
                                              const float4 m,
                                              const float2 pivot,
                                              const float2 offset)
{
  const float x = co.x - pivot.x;
  const float y = co.y - pivot.y;
  return make_float3(
      m.x * x + m.y * y + pivot.x + offset.x, m.z * x + m.w * y + pivot.y + offset.y, co.z);
}

ccl_device_inline float3 tex_randomizer_linear(const float3 d, const float4 m)
{
  return make_float3(m.x * d.x + m.y * d.y, m.z * d.x + m.w * d.y, d.z);
}

ccl_device_inline float3 tex_randomizer_apply(
    const float3 co, const float4 m, const float2 pivot, const float2 offset, const float3 /*tag*/)
{
  return tex_randomizer_apply(co, m, pivot, offset);
}

ccl_device_inline dual3 tex_randomizer_apply(
    const dual3 co, const float4 m, const float2 pivot, const float2 offset, const dual3 /*tag*/)
{
  return dual3(tex_randomizer_apply(co.val, m, pivot, offset),
               tex_randomizer_linear(co.dx, m),
               tex_randomizer_linear(co.dy, m));
}

ccl_device_inline float3 tex_randomizer_value_of(const float3 co)
{
  return co;
}

ccl_device_inline float3 tex_randomizer_value_of(const dual3 co)
{
  return co.val;
}

ccl_device_inline void tex_randomizer_store_float(ccl_private float *stack,
                                                  const SVMStackOffset offset,
                                                  const float value,
                                                  const float3 /*tag*/)
{
  stack_store_float(stack, offset, value);
}

ccl_device_inline void tex_randomizer_store_float(ccl_private float *stack,
                                                  const SVMStackOffset offset,
                                                  const float value,
                                                  const dual3 /*tag*/)
{
  stack_store(stack, offset, dual1(value));
}

template<class Float3Type>
ccl_device_noinline void svm_node_tex_randomizer(
    ccl_private float *ccl_restrict stack,
    const ccl_global SVMNodeTexRandomizer &ccl_restrict node)
{
  const Float3Type co = stack_load<Float3Type>(stack, node.vector);
  const float3 uv = tex_randomizer_value_of(co);
  const uint modes = node.modes;

  /* The UV tile, and with Tile Blend the neighbouring tile this evaluation is for. */
  float tile_x = floorf(uv.x);
  float tile_y = floorf(uv.y);
  float weight = 1.0f;
  if ((modes & NODE_TEX_RANDOMIZER_STOCHASTIC) && node.blend_corner >= 0) {
    const float blend = 0.5f * saturatef(stack_load(stack, node.tile_blend));
    const float fx = uv.x - tile_x;
    const float fy = uv.y - tile_y;
    const float wx = tex_randomizer_edge_weight(fx, blend);
    const float wy = tex_randomizer_edge_weight(fy, blend);
    const bool shift_x = (node.blend_corner & 1) != 0;
    const bool shift_y = (node.blend_corner & 2) != 0;
    if (shift_x) {
      tile_x += (fx < 0.5f) ? -1.0f : 1.0f;
    }
    if (shift_y) {
      tile_y += (fy < 0.5f) ? -1.0f : 1.0f;
    }
    weight = (shift_x ? wx : 1.0f - wx) * (shift_y ? wy : 1.0f - wy);
  }

  /* Hash of everything the randomness depends on. */
  uint h = hash_uint(uint(int(stack_load(stack, node.seed))));
  if (modes & NODE_TEX_RANDOMIZER_OBJECT) {
    h = tex_randomizer_mix(h, stack_load(stack, node.object_random));
  }
  if (modes & NODE_TEX_RANDOMIZER_OBJECT_INDEX) {
    h = tex_randomizer_mix(h, stack_load(stack, node.object_index));
  }
  if (modes & NODE_TEX_RANDOMIZER_ELEMENT) {
    h = tex_randomizer_mix(h, stack_load(stack, node.element_random));
  }
  if (modes & NODE_TEX_RANDOMIZER_FACE) {
    h = tex_randomizer_mix(h, floorf(stack_load(stack, node.face_index) + 0.5f));
  }
  if (modes & NODE_TEX_RANDOMIZER_PARTICLE) {
    h = tex_randomizer_mix(h, stack_load(stack, node.particle_index));
  }
  if (modes & NODE_TEX_RANDOMIZER_STOCHASTIC) {
    h = hash_uint3(h, uint(int(tile_x)), uint(int(tile_y)));
  }

  Float3Type result = co;
  float random = 0.0f;
  if (modes & NODE_TEX_RANDOMIZER_ALL) {
    random = hash_uint_to_float(h);

    const float u_offset = tex_randomizer_value(stack_load(stack, node.u_offset_from),
                                                stack_load(stack, node.u_offset_to),
                                                stack_load(stack, node.u_offset_steps),
                                                hash_uint2_to_float(h, 1u),
                                                false);
    const float v_offset = tex_randomizer_value(stack_load(stack, node.v_offset_from),
                                                stack_load(stack, node.v_offset_to),
                                                stack_load(stack, node.v_offset_steps),
                                                hash_uint2_to_float(h, 2u),
                                                false);
    const float rotation_from = stack_load(stack, node.rotation_from);
    const float rotation_to = stack_load(stack, node.rotation_to);
    const float rotation = tex_randomizer_value(rotation_from,
                                                rotation_to,
                                                stack_load(stack, node.rotation_steps),
                                                hash_uint2_to_float(h, 3u),
                                                fabsf(rotation_to - rotation_from) >=
                                                    M_2PI_F - 1e-4f);
    const float u_scale = tex_randomizer_value(stack_load(stack, node.u_scale_from),
                                               stack_load(stack, node.u_scale_to),
                                               stack_load(stack, node.u_scale_steps),
                                               hash_uint2_to_float(h, 4u),
                                               false);
    const float v_scale = (node.options & NODE_TEX_RANDOMIZER_V_SCALE) ?
                              tex_randomizer_value(stack_load(stack, node.v_scale_from),
                                                   stack_load(stack, node.v_scale_to),
                                                   stack_load(stack, node.v_scale_steps),
                                                   hash_uint2_to_float(h, 5u),
                                                   false) :
                              u_scale;
    const float flip_u = ((node.options & NODE_TEX_RANDOMIZER_FLIP_U) &&
                          hash_uint2_to_float(h, 6u) >= 0.5f) ?
                             -1.0f :
                             1.0f;
    const float flip_v = ((node.options & NODE_TEX_RANDOMIZER_FLIP_V) &&
                          hash_uint2_to_float(h, 7u) >= 0.5f) ?
                             -1.0f :
                             1.0f;

    /* Scale in percent of the texture size: 200% shows the texture twice as large. */
    const float su = flip_u * 100.0f / (fabsf(u_scale) > 1e-6f ? u_scale : 1e-6f);
    const float sv = flip_v * 100.0f / (fabsf(v_scale) > 1e-6f ? v_scale : 1e-6f);
    /* Rotating the coordinates back rotates the texture by `rotation`. */
    const float c = cosf(rotation);
    const float s = sinf(rotation);
    const float4 m = make_float4(c * su, s * su, -s * sv, c * sv);

    /* Around the middle of the UV square, or of the tile with stochastic tiling. */
    const float2 pivot = (modes & NODE_TEX_RANDOMIZER_STOCHASTIC) ?
                             make_float2(tile_x + 0.5f, tile_y + 0.5f) :
                             make_float2(0.5f, 0.5f);
    result = tex_randomizer_apply(co, m, pivot, make_float2(u_offset, v_offset), co);
  }

  if (stack_valid(node.vector_offset)) {
    stack_store(stack, node.vector_offset, result);
  }
  if (stack_valid(node.random_offset)) {
    tex_randomizer_store_float(stack, node.random_offset, random, co);
  }
  if (stack_valid(node.weight_offset)) {
    tex_randomizer_store_float(stack, node.weight_offset, weight, co);
  }
}

CCL_NAMESPACE_END
