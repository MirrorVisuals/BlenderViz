# SPDX-FileCopyrightText: 2026 BlenderViz
#
# SPDX-License-Identifier: Apache-2.0

"""BlenderViz: cloud maps for the Sky Texture, made from photos of the sky (HDRIs).

A cloud map is a fisheye image of the sky: the zenith in the middle, the horizon at the edge, the
angle from the zenith proportional to the distance from the middle, +Y (north) up and +X right.
Red is the opacity of the clouds, green the light and shade they had in the photo (0.5 neutral).
The Sky Texture lights the clouds with its own sun, so the clouds of a midday photo can be used at
sunset as well.

Only numpy (Blender ships no scipy).
"""

import math
import warnings

import numpy as np


# -----------------------------------------------------------------------------
# Small image tools.

def _blur(img, radius):
    """Box blur, three passes (close to a Gaussian), edges clamped. `radius` in pixels."""
    r = int(radius)
    if r < 1:
        return img
    out = img.astype(np.float32)
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r + 1, r)
            p = np.pad(out, pad, mode='edge')
            c = np.cumsum(p, axis=axis, dtype=np.float64)
            if axis == 0:
                out = ((c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)).astype(np.float32)
            else:
                out = ((c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)).astype(np.float32)
    return out


def _block_reduce(img, block, func, valid=None):
    """Reduce blocks of block x block pixels with func (np.nanmax, np.nanpercentile...)."""
    h, w = img.shape
    hb, wb = (h + block - 1) // block, (w + block - 1) // block
    p = np.full((hb * block, wb * block), np.nan, dtype=np.float32)
    v = img.astype(np.float32).copy()
    if valid is not None:
        v[~valid] = np.nan
    p[:h, :w] = v
    p = p.reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, block * block)
    with np.errstate(all='ignore'):
        return func(p, axis=2)


def _fill_nan(grid):
    """Fill missing cells from their neighbours, repeatedly."""
    g = grid.copy()
    for _ in range(max(g.shape) * 2):
        missing = np.isnan(g)
        if not missing.any():
            break
        p = np.pad(g, 1, mode='edge')
        stack = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        with np.errstate(all='ignore'):
            mean = np.nanmean(stack, axis=0)
        g[missing] = mean[missing]
    g[np.isnan(g)] = np.nanmean(grid) if not np.isnan(grid).all() else 0.0
    return g


def _upsample(grid, shape):
    """Bilinear upsampling of a coarse grid (cell centres) to `shape`."""
    h, w = shape
    gh, gw = grid.shape
    ys = (np.arange(h) + 0.5) / h * gh - 0.5
    xs = (np.arange(w) + 0.5) / w * gw - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, gh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, gw - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    a = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx
    b = grid[y1][:, x0] * (1 - fx) + grid[y1][:, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# -----------------------------------------------------------------------------
# Projections.

def fisheye_directions(size):
    """Directions (z up) of the pixels of a size x size cloud map, rows bottom to top (as Blender
    images), and the mask of the pixels inside the circle (above the horizon)."""
    c = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(c, c)
    r = np.sqrt(x * x + y * y)
    inside = r <= 1.0
    theta = np.minimum(r, 1.0) * (math.pi / 2.0)
    phi = np.arctan2(y, x)
    s = np.sin(theta)
    return np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)], axis=-1), inside


def sample_equirectangular(pixels, directions):
    """Bilinear lookup of an equirectangular image (rows bottom to top, as Blender images; the
    mapping of the Environment Texture) in `directions` (..., 3)."""
    h, w = pixels.shape[:2]
    d = directions
    u = 0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2.0 * math.pi)
    v = 1.0 - np.arccos(np.clip(d[..., 2], -1.0, 1.0)) / math.pi
    px = u * w - 0.5
    py = np.clip(v * h - 0.5, 0, h - 1)
    x0 = np.floor(px).astype(int)
    y0 = np.clip(np.floor(py).astype(int), 0, h - 1)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]
    x0 %= w
    x1 = (x0 + 1) % w
    y1 = np.clip(y0 + 1, 0, h - 1)
    a = pixels[y0, x0] * (1 - fx) + pixels[y0, x1] * fx
    b = pixels[y1, x0] * (1 - fx) + pixels[y1, x1] * fx
    return a * (1 - fy) + b * fy


# -----------------------------------------------------------------------------
# Clouds from a photo.

def find_sun(pixels):
    """Direction of the sun in an equirectangular HDRI: the centre of its brightest pixels."""
    h, w = pixels.shape[:2]
    lum = pixels[..., :3].mean(axis=-1)
    top = lum >= np.percentile(lum, 99.995)
    ys, xs = np.nonzero(top)
    wts = lum[ys, xs]
    v = (ys + 0.5) / h
    u = (xs + 0.5) / w
    theta = (1.0 - v) * math.pi
    phi = (0.5 - u) * 2.0 * math.pi
    d = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    d = (d * wts[:, None]).sum(axis=0)
    return d / max(np.linalg.norm(d), 1e-9)


def _binned(values, bins, nbins, func, valid):
    """func over the values in each bin (flat index), NaN where empty."""
    out = np.full(nbins, np.nan, dtype=np.float32)
    idx = bins[valid]
    val = values[valid]
    order = np.argsort(idx, kind='stable')
    idx, val = idx[order], val[order]
    starts = np.searchsorted(idx, np.arange(nbins))
    ends = np.searchsorted(idx, np.arange(nbins), side='right')
    for b in range(nbins):
        if ends[b] - starts[b] >= 8:
            out[b] = func(val[starts[b]:ends[b]])
    return out


def separate_clouds(rgb, dirs, sun_dir, valid=None, noise=0.06):
    """Opacity of the clouds and their shading in a photo of the sky (linear RGB, ... x 3), seen
    in `dirs` (z up), lit by the sun in `sun_dir`.

    A pixel is the clear sky behind the cloud, dimmed by its opacity a, plus the light of the
    cloud, which is white: R = S_R (1 - a) + c a and B = S_B (1 - a) + c a. So
    B - R = (S_B - S_R)(1 - a): the cloud dims the blueness of the sky in proportion to its opacity,
    whatever its own brightness. The clear sky S changes smoothly with the height above the horizon
    and the angle to the sun: it is estimated per pair of those angles from the bluest pixels
    anywhere in the sky (the gaps between the clouds). Then c, the light of the cloud, gives its
    shading. Things darker than the sky (land, trees, buildings) are not clouds.

    Returns (opacity, shading), shading 0.5 neutral.
    """
    shape = rgb.shape[:-1]
    if valid is None:
        valid = np.ones(shape, dtype=bool)
    rgb = np.maximum(rgb.astype(np.float32), 0.0)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    blue = rgb[..., 2] - rgb[..., 0]
    blue_s = _blur(blue, 1) if blue.ndim == 2 else blue

    # Bins of (zenith angle, angle to the sun).
    nt, ng = 30, 60
    theta = np.arccos(np.clip(dirs[..., 2], -1.0, 1.0))
    gamma = np.arccos(np.clip(dirs @ np.asarray(sun_dir, dtype=np.float32), -1.0, 1.0))
    ti = np.clip((theta / (math.pi / 2) * nt).astype(int), 0, nt - 1)
    gi = np.clip((gamma / math.pi * ng).astype(int), 0, ng - 1)
    bins = ti * ng + gi
    ok = valid & (dirs[..., 2] > 0.0)
    # The sun and its glare tell nothing about the sky or the clouds.
    near_sun = gamma < math.radians(2.5)
    ok_sky = ok & ~near_sun & (lum < 30.0 * np.median(lum[ok]))

    def smooth_grid(values):
        return _blur(_fill_nan(values.reshape(nt, ng)), 1)

    sky_blue = smooth_grid(_binned(blue_s, bins, nt * ng, lambda a: np.percentile(a, 97), ok_sky))

    def at(grid):
        tf = np.clip(theta / (math.pi / 2) * nt - 0.5, 0, nt - 1)
        gf = np.clip(gamma / math.pi * ng - 0.5, 0, ng - 1)
        t0, g0 = np.floor(tf).astype(int), np.floor(gf).astype(int)
        t1, g1 = np.minimum(t0 + 1, nt - 1), np.minimum(g0 + 1, ng - 1)
        ft, fg = tf - t0, gf - g0
        return ((grid[t0, g0] * (1 - fg) + grid[t0, g1] * fg) * (1 - ft) +
                (grid[t1, g0] * (1 - fg) + grid[t1, g1] * fg) * ft)

    ref_blue = np.maximum(at(sky_blue), 1e-6)
    gap = ok_sky & (blue_s >= 0.92 * ref_blue)
    ref_rgb = np.stack([np.maximum(at(smooth_grid(_binned(rgb[..., k], bins, nt * ng, np.median,
                                                          gap))), 1e-6) for k in range(3)], -1)
    ref_lum = 0.2126 * ref_rgb[..., 0] + 0.7152 * ref_rgb[..., 1] + 0.0722 * ref_rgb[..., 2]

    a = 1.0 - np.clip(blue_s / ref_blue, 0.0, 1.0)
    # Below `noise` it is the photo's grain and the unevenness of the clear sky.
    opacity = np.clip((a - noise) / (1.0 - noise), 0.0, 1.0)
    # Where the sky is barely blue (around the sun, a hazy horizon) its blueness can't tell clouds
    # apart: trust it less there.
    reliable = 0.5 + 0.5 * _smoothstep(0.03, 0.12, ref_blue / np.maximum(ref_lum, 1e-6))
    opacity *= reliable
    # Land, trees, buildings: darker than the sky (not near the sun, where the sky is too bright
    # to compare with).
    elevation = np.degrees(math.pi / 2 - theta)
    land = (1.0 - _smoothstep(0.35, 0.6, lum / ref_lum)) * (1.0 - _smoothstep(15.0, 25.0, elevation))
    land *= _smoothstep(8.0, 14.0, np.degrees(gamma))
    opacity *= 1.0 - land
    # Over the sun and its glare: fade into the average of the clouds around it, so no disc shows.
    ring = ok & (gamma >= math.radians(4.0)) & (gamma < math.radians(7.0))
    if ring.any():
        fill = float(np.median(opacity[ring]))
        keep = _smoothstep(math.radians(2.0), math.radians(6.0), gamma)
        opacity = opacity * keep + fill * (1.0 - keep)
    opacity[~ok] = 0.0

    # The light of the cloud: c = (R - S_R (1 - a)) / a, against the clouds around it.
    a_safe = np.maximum(a, 0.15)
    c = np.maximum((rgb[..., 0] - ref_rgb[..., 0] * (1.0 - a_safe)) / a_safe, 1e-6)
    if c.ndim == 2:
        r = max(2, min(shape) // 60)
        w = opacity.astype(np.float32)
        c_mean = np.exp(_blur(np.log(c) * w, r) / np.maximum(_blur(w, r), 1e-4))
    else:
        c_mean = c
    rel = np.log2(c / np.maximum(c_mean, 1e-9))
    shading = np.where(opacity > 0.02, np.clip(0.5 + 0.5 * rel, 0.0, 1.0), 0.5)
    return opacity.astype(np.float32), shading.astype(np.float32)


def cloud_map_from_equirectangular(pixels, size=2048, rotation=0.0, horizon_fade=4.0):
    """Cloud map (size x size x 4, rows bottom to top) from an equirectangular HDRI
    (H x W x 3 or 4, rows bottom to top). `rotation` (radians) turns the photo around the zenith;
    `horizon_fade` (degrees) fades out the clouds just above the horizon, where photos have haze
    and land."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(all='ignore'):
            return _cloud_map_from_equirectangular(pixels, size, rotation, horizon_fade)


def _cloud_map_from_equirectangular(pixels, size, rotation, horizon_fade):
    dirs, inside = fisheye_directions(size)
    if rotation:
        ca, sa = math.cos(rotation), math.sin(rotation)
        dx = dirs[..., 0] * ca - dirs[..., 1] * sa
        dy = dirs[..., 0] * sa + dirs[..., 1] * ca
        dirs = np.stack([dx, dy, dirs[..., 2]], axis=-1)
    rgb = sample_equirectangular(pixels[..., :3].astype(np.float32), dirs)
    # `dirs` are directions of the photo, so is its sun.
    sun = find_sun(pixels)
    opacity, shading = separate_clouds(rgb, dirs, sun, valid=inside)
    elevation = np.degrees(np.arcsin(np.clip(dirs[..., 2], 0.0, 1.0)))
    opacity *= _smoothstep(0.0, max(horizon_fade, 1e-3), elevation)
    out = np.zeros((size, size, 4), dtype=np.float32)
    out[..., 0] = opacity
    out[..., 1] = shading
    out[..., 3] = 1.0
    return out


# -----------------------------------------------------------------------------
# Blender.

def image_pixels(image, max_width=8192):
    """Pixels of a Blender image as a float array (H x W x 4, rows bottom to top), scaled down to
    `max_width` if larger."""
    w, h = image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    px = buf.reshape(h, w, 4)
    step = int(math.ceil(w / max_width)) if w > max_width else 1
    if step > 1:
        hh, ww = (h // step) * step, (w // step) * step
        px = px[:hh, :ww].reshape(hh // step, step, ww // step, step, 4).mean(axis=(1, 3))
    return px


def make_cloud_map(source_image, name=None, size=2048, rotation=0.0, horizon_fade=4.0):
    """New Blender image (packed into the file) with the cloud map of an equirectangular HDRI."""
    import bpy

    pixels = image_pixels(source_image)
    data = cloud_map_from_equirectangular(pixels, size=size, rotation=rotation,
                                          horizon_fade=horizon_fade)
    name = name or "Clouds " + source_image.name
    image = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True,
                                is_data=True)
    image.colorspace_settings.is_data = True
    image.pixels.foreach_set(data.ravel())
    image.pack()
    return image
