/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* BlenderViz: cirrus clouds of the Sky Texture (Single and Multiple Scattering).
 *
 * Cirrus is a thin sheet of ice cloud 6 to 12 km up, so it is drawn as a layer on a sphere around
 * the Earth rather than as a volume: noise-free, fast, and with the right perspective (the clouds
 * crowd together toward the horizon). Its shapes are made for cirrus: long torn bands along the
 * wind with fibres combed through them, or cloudlets in rows (cirrocumulus, "mackerel sky").
 * Detail smaller than a pixel is left out (and averaged), so the horizon doesn't shimmer.
 *
 * It is lit by the sky's own sun as seen from the cloud height (golden at sunset, still lit when
 * the sun has set below), with the strong forward scattering of ice crystals (bright toward the
 * sun), the sky above, and the air in front of it (fading into the haze toward the horizon).
 * All lengths in km. */

#pragma once

#include "kernel/globals.h"
#include "kernel/types.h"

#include "kernel/svm/node_types.h"

#include "kernel/util/colorspace.h"

#include "util/hash.h"
#include "util/math.h"

CCL_NAMESPACE_BEGIN

ccl_device_inline uint cirrus_hash(const float x, const float y, const uint s)
{
  uint h = (uint(int(x)) * 0x8da6b343u) ^ (uint(int(y)) * 0xd8163841u) ^ (s * 0xcb1ab31fu);
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h;
}

ccl_device_inline float cirrus_unit(const uint h)
{
  return float(h >> 8) * (1.0f / float(1 << 24));
}

ccl_device_inline float cirrus_fade(const float t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

ccl_device_inline float cirrus_grad(const uint hash, const float dx, const float dy)
{
  /* One of 8 directions. */
  const uint h = hash & 7u;
  if (h < 2u) {
    return (h & 1u) ? dx : -dx;
  }
  if (h < 4u) {
    return (h == 2u) ? dy : -dy;
  }
  return 0.70710678f * (((h & 1u) ? dx : -dx) + ((h & 2u) ? dy : -dy));
}

/* Gradient noise, about -1..1. */
ccl_device float cirrus_noise(const float x, const float y, const uint s)
{
  const float ix = floorf(x);
  const float iy = floorf(y);
  const float fx = x - ix;
  const float fy = y - iy;
  const float n00 = cirrus_grad(cirrus_hash(ix, iy, s), fx, fy);
  const float n10 = cirrus_grad(cirrus_hash(ix + 1.0f, iy, s), fx - 1.0f, fy);
  const float n01 = cirrus_grad(cirrus_hash(ix, iy + 1.0f, s), fx, fy - 1.0f);
  const float n11 = cirrus_grad(cirrus_hash(ix + 1.0f, iy + 1.0f, s), fx - 1.0f, fy - 1.0f);
  const float u = cirrus_fade(fx);
  const float v = cirrus_fade(fy);
  return 1.414f * ((n00 * (1.0f - u) + n10 * u) * (1.0f - v) + (n01 * (1.0f - u) + n11 * u) * v);
}

/* Octaves fade out where they are smaller than the footprint of a pixel (in noise units), and are
 * then not evaluated at all. */
ccl_device float cirrus_fbm(const float x,
                            const float y,
                            const int octaves,
                            const uint s,
                            const float footprint,
                            const float gain = 0.5f)
{
  float total = 0.0f;
  float amp = 1.0f;
  float norm = 0.0f;
  float freq = 1.0f;
  for (int o = 0; o < octaves; o++) {
    const float w = saturatef(1.0f / (freq * 2.5f * footprint) - 1.0f);
    if (w > 0.0f) {
      /* Turn each octave's lattice, so stretched noise doesn't line up with it in bands. */
      const float ang = 0.6f + 1.7f * float(o);
      const float co = cosf(ang), so = sinf(ang);
      total += amp * w *
               cirrus_noise((x * co - y * so) * freq + 13.1f * float(o),
                            (x * so + y * co) * freq - 7.7f * float(o),
                            s + uint(o));
    }
    norm += amp;
    amp *= gain;
    freq *= 2.0f;
  }
  return total / norm;
}

ccl_device_inline float cirrus_smoothstep(const float a, const float b, const float x)
{
  const float t = saturatef((x - a) / (b - a));
  return t * t * (3.0f - 2.0f * t);
}

/* Density (0..1) of the cirrus layer at layer position x, y (km); `footprint` is the size of a
 * pixel there (km). */
ccl_device float cirrus_density(const ccl_global SVMNodeTexSkyCloudsData &c,
                                float x,
                                float y,
                                const float footprint)
{
  const uint s = c.seed;
  x += c.offset_x;
  y += c.offset_y;
  const float ca = cosf(c.direction);
  const float sa = sinf(c.direction);
  const float scale = c.scale;
  const float u = (x * ca + y * sa) / scale;
  const float v = (-x * sa + y * ca) / scale;
  const float fp = footprint / scale;

  /* Where the bands themselves are too small to see (toward the horizon): their average, the
   * whitish haze of distant cirrus. */
  const float resolved = saturatef((log2f(5.0f / (3.0f * fp)) + 2.0f) / 3.0f);
  const float average = 0.18f * cirrus_smoothstep(0.0f, 0.8f, c.amount);
  if (resolved <= 0.0f) {
    return average;
  }

  /* Gentle large bends (turbulence), and smaller curls for hooks. */
  const float turb = c.turbulence;
  float uw = u, vw = v;
  if (turb > 0.0f) {
    uw += cirrus_fbm(u / 40.0f, v / 40.0f, 3, s + 3, fp / 40.0f) * 10.0f * turb +
          cirrus_fbm(u / 4.0f, v / 4.0f, 2, s + 7, fp / 4.0f) * 1.2f * turb;
    vw += cirrus_fbm(u / 40.0f + 5.2f, v / 40.0f - 1.3f, 3, s + 5, fp / 40.0f) * 10.0f * turb +
          cirrus_fbm(u / 4.0f - 2.1f, v / 4.0f + 4.4f, 2, s + 9, fp / 4.0f) * 1.2f * turb;
  }

  /* Cirrus forms long torn bands along the wind; cloudlet fields (dapple) broader patches. */
  const float dp = c.dapple;
  const float cover = cirrus_fbm(
      uw / (30.0f - 8.0f * dp), vw / (5.0f + 13.0f * dp), 4, s + 11, fp / 5.0f);
  const float threshold = 0.25f - 0.8f * c.amount;

  /* Fibres combed through the band at an angle to it: streaks of random width. */
  const float cg = cosf(c.comb);
  const float sg = sinf(c.comb);
  const float uf = uw * cg + vw * sg;
  const float vf = -uw * sg + vw * cg;
  const float fib = cirrus_fbm(uf / 9.0f, vf / 0.25f, 4, s + 21, fp / 0.25f);
  const float fib2 = cirrus_fbm(uw / 10.0f, vw / 0.35f, 3, s + 23, fp / 0.35f);
  const float fibres = powf(saturatef(0.5f + 1.3f * (0.8f * fib + 0.2f * fib2)), 1.5f);
  /* The edges of the band fray along the fibres. */
  const float mask = cirrus_smoothstep(
      threshold, threshold + 0.5f, cover + 0.3f * (fibres - 0.45f));
  if (mask <= 0.0f) {
    return average * (1.0f - resolved);
  }

  /* A soft veil, thickest in the middle of a band. */
  const float veil = mask *
                     (0.55f + 0.45f * cirrus_fbm(uw / 6.0f, vw / 1.5f, 3, s + 13, fp / 1.5f));

  float shape = 0.0f;
  if (dp < 1.0f) {
    const float fibrous = veil * (0.2f + 1.5f * powf(fibres, 1.6f));
    shape += fibrous * (1.0f - dp);
  }
  if (dp > 0.0f) {
    /* Mackerel sky: wavy rows across the wind, each broken at random into puffy cloudlets. */
    const float period = 1.5f;
    const float phase = uw / period +
                        0.9f * cirrus_fbm(uw / 7.0f, vw / 7.0f, 2, s + 51, fp / 7.0f) +
                        0.12f * cirrus_fbm(uw / 0.4f, vw / 0.4f, 2, s + 53, fp / 0.4f);
    const float row = floorf(phase);
    const float across = phase - row - 0.5f;
    const float erosion = 0.22f * cirrus_fbm(uw / 0.15f, vw / 0.15f, 3, s + 49, fp / 0.15f);
    /* Along each row, cloudlets of random length with gaps: soft ellipses. */
    const float along = vw / 1.1f + row * 0.37f;
    const float k0 = floorf(along);
    float lets = 0.0f;
    for (int dk = -1; dk <= 1; dk++) {
      const float k = k0 + float(dk);
      const float h = cirrus_unit(cirrus_hash(row, k, s + 61));
      const float h2 = cirrus_unit(cirrus_hash(row, k, s + 63));
      const float present = cirrus_smoothstep(-0.05f, 0.15f, h - (0.1f - 0.6f * (mask - 0.5f)));
      if (present <= 0.0f) {
        continue;
      }
      const float center = k + 0.5f + (h2 - 0.5f) * 0.4f;
      const float length = 0.45f + 0.35f * h;
      const float width = 0.3f + 0.12f * h2;
      const float dist = sqrtf(sqr(across / width) + sqr((along - center) / length));
      lets = fmaxf(lets, cirrus_smoothstep(1.05f, 0.2f, dist + erosion) * present *
                                   (0.6f + 0.4f * h2));
    }
    /* Rows smaller than a few pixels blur into their average. */
    const float rows_seen = saturatef(period / (3.0f * fp) - 1.0f);
    lets = lets * rows_seen + 0.25f * (1.0f - rows_seen);
    float dapple = cirrus_smoothstep(0.0f, 0.5f, mask) * lets * 1.6f;
    const float distant = saturatef((fp - 0.1f) / 0.3f);
    dapple = dapple * (1.0f - distant) + veil * 0.6f * distant;
    shape += dapple * dp;
  }

  return saturatef(shape * resolved + average * (1.0f - resolved));
}

ccl_device_inline float cirrus_phase_hg(const float cos_t, const float g)
{
  const float g2 = g * g;
  const float d = 1.0f + g2 - 2.0f * g * cos_t;
  return (1.0f - g2) / (4.0f * M_PI_F * d * sqrtf(d));
}

/* Add the cirrus layer to the sky. `sky_rgb` is the sky (and sun) seen in `dir`; `zenith_xyz`
 * the sky straight up (in XYZ, as the sun), for the light the clouds get from the sky above. */
ccl_device float3 sky_cirrus(KernelGlobals kg,
                             const ccl_private ShaderData *sd,
                             const uint32_t path_flag,
                             const ccl_global SVMNodeTexSkyCloudsData &c,
                             const float3 dir,
                             const float3 sun_dir,
                             const float3 sky_rgb,
                             const float3 zenith_xyz)
{
  if (dir.z <= 1e-4f || c.density <= 0.0f || c.altitude <= 0.0f) {
    return sky_rgb;
  }

  /* Where the view ray meets the layer: a sphere around the Earth, `altitude` above the eye. */
  const float earth = c.earth_radius;
  const float h = c.altitude;
  const float b = earth * dir.z;
  const float t = -b + sqrtf(b * b + 2.0f * earth * h + h * h);
  const float cos_incidence = (earth * dir.z + t) / (earth + h);
  const float lx = dir.x * t;
  const float ly = dir.y * t;

  /* Size of a pixel on the layer, from the spread of the ray: sharp for camera rays and mirror
   * reflections (glass), blurred after rough bounces. Unknown (baking the background light):
   * only the large shapes. */
  const bool unknown = (path_flag & PATH_RAY_IMPORTANCE_BAKE) || !(sd->dI > 0.0f);
  const float pixel_angle = unknown ? 0.02f : clamp(sd->dI, 1e-5f, 0.05f);
  const bool camera = pixel_angle < 0.005f;
  const float footprint = t * pixel_angle / fmaxf(cos_incidence, 0.05f);

  const float density = cirrus_density(c, lx, ly, footprint);
  if (density <= 0.0f) {
    return sky_rgb;
  }

  /* Thin layer: optical depth along the view. */
  const float tau = c.density * density / fmaxf(cos_incidence, 0.15f);
  const float alpha = 1.0f - expf(-tau);

  /* Relief: a second sample a little toward the sun along the layer. Where the cloud thins
   * toward the sun that side is lit, where it thickens, shaded. */
  float relief = 1.0f;
  if (camera) {
    const float2 sun_h = make_float2(sun_dir.x, sun_dir.y);
    const float sun_h_len = len(sun_h);
    if (sun_h_len > 1e-4f) {
      const float step = 0.06f * c.scale;
      const float toward_sun = cirrus_density(
          c, lx + sun_h.x / sun_h_len * step, ly + sun_h.y / sun_h_len * step, footprint);
      relief = clamp(1.0f + 2.5f * (density - toward_sun), 0.55f, 1.35f);
    }
  }

  /* Ice crystals scatter strongly forward (bright toward the sun), some backward, and multiple
   * scattering adds a little light in every direction. */
  const float cos_t = dot(dir, sun_dir);
  const float phase = 0.75f * cirrus_phase_hg(cos_t, 0.75f) +
                      0.25f * cirrus_phase_hg(cos_t, -0.2f);
  const float3 sun_xyz = make_float3(c.sun_x, c.sun_y, c.sun_z);
  float3 cloud_xyz = sun_xyz * (phase * 0.9f + 0.08f) + zenith_xyz * 0.8f;
  /* Thicker parts get a little grey. */
  cloud_xyz *= (1.0f - 0.35f * (1.0f - expf(-1.5f * tau))) * relief;
  const float3 cloud_rgb = xyz_to_rgb_clamped(kg, cloud_xyz);

  /* Air in front of the cloud dims and reddens it toward the horizon (relative air mass). */
  const float elevation_deg = asinf(dir.z) * (180.0f / M_PI_F);
  const float airmass = 1.0f / (dir.z + 0.15f * powf(elevation_deg + 3.885f, -1.253f));
  const float3 t_air = make_float3(expf(-c.extinction_r * airmass),
                                   expf(-c.extinction_g * airmass),
                                   expf(-c.extinction_b * airmass));

  /* The sky in front of the cloud stays, the sky behind it is dimmed by it. */
  return sky_rgb * (c.below + (1.0f - c.below) * (1.0f - alpha)) + cloud_rgb * alpha * t_air;
}

CCL_NAMESPACE_END
