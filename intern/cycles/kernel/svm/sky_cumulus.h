/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* BlenderViz: cumulus clouds of the Sky Texture (Single and Multiple Scattering).
 *
 * Unlike the cirrus (a thin sheet, drawn as a layer), cumulus are volumes: the view ray is marched
 * through the cloud layer, 1 to 4 km up. A weather map says where the clouds are and how tall
 * they grow; within that, the billowy shapes are carved from a tileable 3D Perlin-Worley noise and
 * their edges eroded by finer Worley noise (wispy at the base, billowy at the top), as in the
 * cloud systems of Horizon Zero Dawn (Schneider 2015) and Frostbite (Hillaire 2016).
 *
 * Light: the sun, dimmed by the cloud between the point and the sun (a few steps toward it), with
 * the scattering of several orders approximated by octaves of lower extinction and weaker
 * anisotropy (Wrenninge 2013), so thick clouds stay luminous; and the light of the sky around,
 * from above at the tops and from the horizon and the sunlit ground at the base. The air in front
 * of the clouds fades them into the haze toward the horizon. All lengths in km. */

#pragma once

#include "kernel/svm/sky_clouds.h"

CCL_NAMESPACE_BEGIN

/* Must match scene/image_cloud_noise.h. */
#define CUMULUS_NOISE_RES 96
#define CUMULUS_NOISE_TILE 98
#define CUMULUS_NOISE_GRID 10

ccl_device_inline float4 cumulus_slice(KernelGlobals kg,
                                       ccl_private ShaderData *sd,
                                       const int id,
                                       const float x,
                                       const float y,
                                       const int z)
{
  const int col = z % CUMULUS_NOISE_GRID;
  const int row = z / CUMULUS_NOISE_GRID;
  const float size = float(CUMULUS_NOISE_TILE * CUMULUS_NOISE_GRID);
  const float u = (float(col * CUMULUS_NOISE_TILE) + 1.0f + x * float(CUMULUS_NOISE_RES)) / size;
  const float v = (float(row * CUMULUS_NOISE_TILE) + 1.0f + y * float(CUMULUS_NOISE_RES)) / size;
  return kernel_image_interp(kg, sd, id, dual2(make_float2(u, v)));
}

/* The noise volume at `p`, in repeats of the volume (it tiles). */
ccl_device float4 cumulus_noise(KernelGlobals kg,
                                ccl_private ShaderData *sd,
                                const int id,
                                const float3 p)
{
  const float x = p.x - floorf(p.x);
  const float y = p.y - floorf(p.y);
  const float zf = (p.z - floorf(p.z)) * float(CUMULUS_NOISE_RES) - 0.5f;
  const float z0f = floorf(zf);
  const float fz = zf - z0f;
  int z0 = int(z0f);
  if (z0 < 0) {
    z0 += CUMULUS_NOISE_RES;
  }
  int z1 = z0 + 1;
  if (z1 >= CUMULUS_NOISE_RES) {
    z1 -= CUMULUS_NOISE_RES;
  }
  const float4 a = cumulus_slice(kg, sd, id, x, y, z0);
  const float4 b = cumulus_slice(kg, sd, id, x, y, z1);
  return a + (b - a) * fz;
}

ccl_device_inline float cumulus_remap(
    const float x, const float a, const float b, const float c, const float d)
{
  return c + (x - a) / (b - a) * (d - c);
}

/* Where the clouds are (x: coverage 0..1) and how tall they grow (y: share of the thickness). */
ccl_device float2 cumulus_weather(const ccl_global SVMNodeTexSkyCumulusData &c,
                                  const float x,
                                  const float y)
{
  const float s = c.weather_scale;
  const float n = cirrus_fbm(x / s, y / s, 4, c.seed + 300u, 1e-6f);
  const float n2 = cirrus_fbm(x / (0.35f * s), y / (0.35f * s), 3, c.seed + 350u, 1e-6f);
  float cov = saturatef((n * 0.8f + n2 * 0.5f) * 1.6f + (c.coverage - 0.5f) * 1.4f + 0.5f);
  cov = cirrus_smoothstep(0.35f, 0.8f, cov);
  const float tall = cirrus_fbm(x / (1.3f * s), y / (1.3f * s), 2, c.seed + 390u, 1e-6f);
  const float h = clamp(0.45f + 0.9f * tall + 0.35f * cov, 0.25f, 1.0f);
  return make_float2(cov, h);
}

/* Density (0..1) at `p` (x, y along the ground, z height above it). */
ccl_device float cumulus_density(KernelGlobals kg,
                                 ccl_private ShaderData *sd,
                                 const ccl_global SVMNodeTexSkyCumulusData &c,
                                 const float3 p,
                                 const float2 weather,
                                 const bool detail)
{
  const float h = (p.z - c.base) / (c.thickness * weather.y);
  if (h <= 0.0f || h >= 1.0f) {
    return 0.0f;
  }
  /* Domes: the cloud narrows toward its top. */
  const float cov = weather.x * (1.0f - 0.75f * h * sqrtf(h));
  if (cov <= 1e-3f) {
    return 0.0f;
  }
  const float grad = cirrus_smoothstep(0.0f, 0.1f, h) * (1.0f - cirrus_smoothstep(0.45f, 1.0f, h));
  /* The clouds lean a little with the wind higher up. */
  const float lean = p.z - c.base;
  const float3 q = make_float3(p.x + 0.4f * lean, p.y, p.z) / c.shape_scale;
  const float4 s = cumulus_noise(kg, sd, c.noise_id, q);
  float base = saturatef(cumulus_remap(s.x, -(1.0f - s.y * 0.6f - s.z * 0.4f), 1.0f, 0.0f, 1.0f));
  base = cirrus_smoothstep(0.45f, 0.95f, base);
  float d = saturatef(cumulus_remap(base * grad, 1.0f - cov, 1.0f, 0.0f, 1.0f)) * cov;
  if (detail && d > 0.0f) {
    const float3 qd = make_float3(p.x + 0.3f * lean, p.y, p.z) / c.detail_scale;
    const float dn = cumulus_noise(kg, sd, c.noise_id, qd).w;
    /* Wispy at the base, billowy at the top. */
    const float top = saturatef(h * 3.0f);
    const float mod = dn * (1.0f - top) + (1.0f - dn) * top;
    d = saturatef(cumulus_remap(d, mod * 0.3f, 1.0f, 0.0f, 1.0f));
  }
  return d;
}

/* Distance along a ray (sine of its elevation `dz`) to height `height` above the curved ground. */
ccl_device_inline float cumulus_distance_to(const float dz, const float height, const float earth)
{
  const float a = (1.0f - dz * dz) / (2.0f * earth);
  if (a < 1e-9f) {
    return height / fmaxf(dz, 1e-6f);
  }
  return (-dz + sqrtf(dz * dz + 4.0f * a * height)) / (2.0f * a);
}

ccl_device float3 sky_cumulus(KernelGlobals kg,
                              ccl_private ShaderData *sd,
                              const uint32_t path_flag,
                              const ccl_global SVMNodeTexSkyCumulusData &c,
                              const float3 dir,
                              const float3 sun_dir,
                              const float3 sky_rgb,
                              const CirrusAmbient &ambient)
{
  if (dir.z <= 0.0f) {
    return sky_rgb;
  }
  const float earth = c.earth_radius;
  const float t0 = cumulus_distance_to(dir.z, c.base, earth);
  /* Far away the layer is a thin line in the haze. */
  const float max_distance = 120.0f;
  if (t0 > max_distance) {
    return sky_rgb;
  }
  const float t1 = fminf(cumulus_distance_to(dir.z, c.base + c.thickness, earth), t0 + 25.0f);

  /* Full detail, except where the sky is baked for sampling the light it casts. */
  const bool camera = (path_flag & PATH_RAY_IMPORTANCE_BAKE) == 0;
  const int steps = camera ? 48 : 12;
  const int light_steps = camera ? 5 : 2;
  const float dt = (t1 - t0) / float(steps);
  const float jitter = cirrus_unit(
      hash_uint3(__float_as_uint(dir.x), __float_as_uint(dir.y), __float_as_uint(dir.z)));

  const float cos_t = dot(dir, sun_dir);
  float phase[4];
  for (int k = 0; k < 4; k++) {
    const float g = 1.0f / float(1 << k);
    phase[k] = 0.8f * cirrus_phase_hg(cos_t, 0.8f * g) + 0.2f * cirrus_phase_hg(cos_t, -0.3f * g);
  }
  const float3 sun_xyz = make_float3(c.sun_x, c.sun_y, c.sun_z);
  /* Light from all around: the sky above at the tops, the horizon and the sunlit ground below. */
  const float3 amb_top = ambient.above * (M_PI_F * 0.9f);
  const float3 ground = sun_xyz * (fmaxf(sun_dir.z, 0.0f) * 0.25f);
  const float3 amb_bottom = ground * 0.5f + ambient.below * (M_PI_F * 0.3f);

  float transmittance = 1.0f;
  float3 light = zero_float3();
  float t_sum = 0.0f, w_sum = 0.0f;
  for (int i = 0; i < steps; i++) {
    const float t = t0 + (float(i) + jitter) * dt;
    float3 p = dir * t;
    p.z = t * dir.z + t * t * (1.0f - dir.z * dir.z) / (2.0f * earth);
    p.x += c.offset_x;
    p.y += c.offset_y;
    const float2 weather = cumulus_weather(c, p.x, p.y);
    if (weather.x < 0.01f) {
      continue;
    }
    const float rho = cumulus_density(kg, sd, c, p, weather, camera && t < 40.0f);
    if (rho < 1e-3f) {
      continue;
    }
    /* Toward the sun: the cloud in between (its shape only), at growing steps. */
    float od = 0.0f, prev = 0.0f;
    for (int j = 0; j < light_steps; j++) {
      const float f = float(j + 1) / float(light_steps);
      const float ld = 0.04f + 1.36f * f * f;
      const float3 q = p + sun_dir * ld;
      od += cumulus_density(kg, sd, c, q, cumulus_weather(c, q.x, q.y), false) * c.sigma *
            (ld - prev);
      prev = ld;
    }
    float scatter = 0.0f;
    for (int k = 0; k < 4; k++) {
      const float g = 1.0f / float(1 << k);
      scatter += g * phase[k] * expf(-od * g * (k ? 0.25f : 1.0f));
    }
    /* Edges facing the sun are a little darker: little light has scattered in yet. */
    const float powder = 1.0f - expf(-rho * c.sigma * 0.15f);
    scatter *= M_PI_F * (0.6f + 0.4f * powder);
    const float hn = saturatef((p.z - c.base) / (c.thickness * weather.y));
    const float3 amb = amb_top * (0.35f + 0.65f * hn) + amb_bottom * ((1.0f - hn) * 0.6f);
    const float3 source = sun_xyz * scatter + amb * (1.0f / M_2PI_F);
    const float step_t = expf(-rho * c.sigma * dt);
    const float absorbed = transmittance * (1.0f - step_t);
    light += source * absorbed;
    t_sum += t * absorbed;
    w_sum += absorbed;
    transmittance *= step_t;
    if (transmittance < 0.01f) {
      break;
    }
  }
  if (w_sum <= 0.0f) {
    return sky_rgb;
  }

  /* The air in front of the clouds: their light dims and the haze takes their place. */
  const float dist = t_sum / w_sum;
  const float dz = fmaxf(dir.z, 1e-4f);
  const float column_r = 8.0f * (1.0f - expf(-dist * dz / 8.0f)) / dz;
  const float column_m = 1.2f * (1.0f - expf(-dist * dz / 1.2f)) / dz;
  const float3 t_air = make_float3(expf(-(c.rayleigh_r * column_r + c.mie * column_m)),
                                   expf(-(c.rayleigh_g * column_r + c.mie * column_m)),
                                   expf(-(c.rayleigh_b * column_r + c.mie * column_m)));
  const float3 cloud_rgb = xyz_to_rgb_clamped(kg, light);
  /* The haze glows like the sky there, but not like the sun's disc behind a cloud. */
  const float3 haze = min(sky_rgb, xyz_to_rgb_clamped(kg, ambient.below) * 4.0f);
  return sky_rgb * transmittance + cloud_rgb * t_air +
         haze * (one_float3() - t_air) * (1.0f - transmittance);
}

CCL_NAMESPACE_END
