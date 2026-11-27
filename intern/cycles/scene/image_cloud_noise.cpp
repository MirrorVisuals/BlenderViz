/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/image_cloud_noise.h"

#include "util/hash.h"
#include "util/image_metadata.h"
#include "util/math.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

namespace {

constexpr int RES = CLOUD_NOISE_RES;

inline int wrap(const int i, const int n)
{
  const int r = i % n;
  return (r < 0) ? r + n : r;
}

inline float rnd(const int x, const int y, const int z, const uint seed)
{
  return hash_uint4_to_float(uint(x), uint(y), uint(z), seed);
}

/* Inverted Worley noise (1 at the feature points), tiling with `cells` cells across. */
void worley(vector<float> &out, const int cells, const uint seed)
{
  out.resize(size_t(RES) * RES * RES);
  for (int z = 0; z < RES; z++) {
    for (int y = 0; y < RES; y++) {
      for (int x = 0; x < RES; x++) {
        const float px = (float(x) + 0.5f) / RES * cells;
        const float py = (float(y) + 0.5f) / RES * cells;
        const float pz = (float(z) + 0.5f) / RES * cells;
        const int ix = int(floorf(px)), iy = int(floorf(py)), iz = int(floorf(pz));
        float best = 1e10f;
        for (int dz = -1; dz <= 1; dz++) {
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              const int cx = ix + dx, cy = iy + dy, cz = iz + dz;
              const int wx = wrap(cx, cells), wy = wrap(cy, cells), wz = wrap(cz, cells);
              const float fx = float(cx) + rnd(wx, wy, wz, seed) - px;
              const float fy = float(cy) + rnd(wx, wy, wz, seed + 1) - py;
              const float fz = float(cz) + rnd(wx, wy, wz, seed + 2) - pz;
              best = min(best, fx * fx + fy * fy + fz * fz);
            }
          }
        }
        out[(size_t(z) * RES + y) * RES + x] = 1.0f - min(sqrtf(best), 1.0f);
      }
    }
  }
}

inline float fade(const float t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float3 gradient(const int x, const int y, const int z, const int period, const uint seed)
{
  const int wx = wrap(x, period), wy = wrap(y, period), wz = wrap(z, period);
  const float th = rnd(wx, wy, wz, seed) * M_2PI_F;
  const float gz = rnd(wx, wy, wz, seed + 7) * 2.0f - 1.0f;
  const float r = sqrtf(max(1.0f - gz * gz, 0.0f));
  return make_float3(r * cosf(th), r * sinf(th), gz);
}

/* Gradient noise tiling with `period` lattice cells across. */
float perlin(const float px, const float py, const float pz, const int period, const uint seed)
{
  const int ix = int(floorf(px)), iy = int(floorf(py)), iz = int(floorf(pz));
  const float fx = px - ix, fy = py - iy, fz = pz - iz;
  const float u = fade(fx), v = fade(fy), w = fade(fz);
  float out = 0.0f;
  for (int dz = 0; dz <= 1; dz++) {
    for (int dy = 0; dy <= 1; dy++) {
      for (int dx = 0; dx <= 1; dx++) {
        const float3 g = gradient(ix + dx, iy + dy, iz + dz, period, seed);
        const float d = g.x * (fx - dx) + g.y * (fy - dy) + g.z * (fz - dz);
        out += d * (dx ? u : 1.0f - u) * (dy ? v : 1.0f - v) * (dz ? w : 1.0f - w);
      }
    }
  }
  return out;
}

}  // namespace

bool CloudNoiseLoader::load_metadata(ImageMetaData &metadata,
                                     const ImageLoaderParams & /*params*/,
                                     Progress & /*progress*/)
{
  metadata.width = CLOUD_NOISE_TILE * CLOUD_NOISE_GRID;
  metadata.height = CLOUD_NOISE_TILE * CLOUD_NOISE_GRID;
  metadata.channels = 4;
  metadata.type = IMAGE_DATA_TYPE_FLOAT4;
  metadata.is_compressible_as_srgb = false;
  return true;
}

bool CloudNoiseLoader::load_pixels(const ImageMetaData &metadata, void *pixels)
{
  vector<float> w4, w8, w16, w32, d4, d8, d16;
  worley(w4, 4, 1u);
  worley(w8, 8, 2u);
  worley(w16, 16, 3u);
  worley(w32, 32, 4u);
  worley(d4, 4, 21u);
  worley(d8, 8, 22u);
  worley(d16, 16, 23u);

  const int width = metadata.width;
  float *out = (float *)pixels;
  memset(out, 0, sizeof(float) * 4 * size_t(width) * metadata.height);

  for (int z = 0; z < RES; z++) {
    const int col = z % CLOUD_NOISE_GRID;
    const int row = z / CLOUD_NOISE_GRID;
    /* The slice with its wrapped border: pixel j of the tile is voxel j - 1 (wrapping). */
    for (int ty = 0; ty < CLOUD_NOISE_TILE; ty++) {
      const int y = wrap(ty - 1, RES);
      for (int tx = 0; tx < CLOUD_NOISE_TILE; tx++) {
        const int x = wrap(tx - 1, RES);
        const size_t i = (size_t(z) * RES + y) * RES + x;
        float p = 0.0f, amp = 1.0f;
        for (int o = 0; o < 3; o++) {
          const int period = 4 << o;
          p += amp * perlin(float(x) / RES * period,
                            float(y) / RES * period,
                            float(z) / RES * period,
                            period,
                            11u + uint(o));
          amp *= 0.5f;
        }
        p = saturatef(p * 0.7f + 0.5f);
        const float wf = w4[i] * 0.625f + w8[i] * 0.25f + w16[i] * 0.125f;
        /* Perlin-Worley: the Perlin noise remapped by the Worley, billowy and connected. */
        const float pw = saturatef((p - (wf - 1.0f)) / (1.0f - (wf - 1.0f)));
        float *px = out + 4 * (size_t(row * CLOUD_NOISE_TILE + ty) * width +
                               size_t(col * CLOUD_NOISE_TILE + tx));
        px[0] = pw;
        px[1] = wf;
        px[2] = w8[i] * 0.625f + w16[i] * 0.25f + w32[i] * 0.125f;
        px[3] = d4[i] * 0.625f + d8[i] * 0.25f + d16[i] * 0.125f;
      }
    }
  }

  metadata.conform_pixels(pixels);
  return true;
}

string CloudNoiseLoader::name() const
{
  return "cloud_noise";
}

bool CloudNoiseLoader::equals(const ImageLoader & /*other*/) const
{
  return true;
}

CCL_NAMESPACE_END
