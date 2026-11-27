/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* BlenderViz: the tileable 3D noise the cumulus clouds of the Sky Texture are carved from
 * (kernel/svm/sky_cumulus.h). */

#pragma once

#include "scene/image_loader.h"

CCL_NAMESPACE_BEGIN

/* A RES^3 volume stored as GRID x GRID slices of TILE x TILE pixels (one pixel of wrapped border
 * around each slice, so filtering within a slice tiles seamlessly). Channels: R Perlin-Worley
 * (the billowy base shape), G and B Worley at two frequencies (to carve it), A fine Worley (the
 * erosion detail). */
#define CLOUD_NOISE_RES 96
#define CLOUD_NOISE_TILE 98
#define CLOUD_NOISE_GRID 10

class CloudNoiseLoader : public ImageLoader {
 public:
  CloudNoiseLoader() = default;
  ~CloudNoiseLoader() override = default;

  bool load_metadata(ImageMetaData &metadata,
                     const ImageLoaderParams &params,
                     Progress &progress) override;

  bool load_pixels(const ImageMetaData &metadata, void *pixels) override;

  string name() const override;

  bool equals(const ImageLoader &other) const override;
};

CCL_NAMESPACE_END
