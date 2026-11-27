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

def find_sun(pixels, with_color=False):
    """Direction of the sun in an equirectangular HDRI: the centre of its brightest pixels (and
    their color, scaled to luminance 1, with `with_color`)."""
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
    d = d / max(np.linalg.norm(d), 1e-9)
    if not with_color:
        return d
    rgb = pixels[ys, xs, :3].astype(np.float64).mean(axis=0)
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    color = rgb / lum if lum > 0 else np.ones(3)
    # A clipped sun reads white: then assume the usual midday color.
    if np.ptp(color) < 0.02:
        color = np.array([1.08, 0.99, 0.86])
    return d, color


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


def separate_clouds(rgb, dirs, sun_dir, valid=None, noise=0.06, sun_color=(1.0, 1.0, 1.0),
                    reference=None, lum_blur=None, around=None, details=False):
    """Opacity of the clouds and their shading in a photo of the sky (linear RGB, ... x 3), seen
    in `dirs` (z up), lit by the sun in `sun_dir`.

    A pixel is the clear sky behind the cloud, dimmed by its opacity a, plus the light of the
    cloud, which has the color w of the sun (luminance 1): R = S_R (1 - a) + c a w_R,
    B = S_B (1 - a) + c a w_B, L = S_L (1 - a) + c a. With q = w_B - w_R,
    B - R - q L = (S_B - S_R - q S_L)(1 - a): the cloud dims this blueness of the sky in proportion
    to its opacity, whatever its own brightness. (Sunlight is warm: without q, bright thin veils
    near the sun would read as opaque.) The clear sky S changes smoothly with the height above the horizon
    and the angle to the sun: it is estimated per pair of those angles from the bluest pixels
    anywhere in the sky (the gaps between the clouds). Then c, the light of the cloud, gives its
    shading. Things darker than the sky (land, trees, buildings) are not clouds.

    Returns (opacity, shading), shading 0.5 neutral.

    For large maps the slowly changing parts are worked out on a smaller map first: with
    `details`, (opacity, shading, extra) is returned, and `extra` (the clear sky per pair of
    angles, the blurred brightness of the photo and of the clouds around each point) can be passed
    back as `reference`, `lum_blur` and `around` (the latter two resampled to the larger map).
    """
    shape = rgb.shape[:-1]
    if valid is None:
        valid = np.ones(shape, dtype=bool)
    rgb = np.maximum(rgb.astype(np.float32), 0.0)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # The light on the clouds is never as deep orange as the sun's disc at sunset seen through
    # the whole air (and a much warmer guess would make bright clouds read as clear sky).
    q = float(np.clip(sun_color[2] - sun_color[0], -0.3, 0.0))
    blue = rgb[..., 2] - rgb[..., 0] - q * lum
    blue_s = _blur(blue, 1) if blue.ndim == 2 else blue

    # Bins of (zenith angle, angle to the sun).
    nt, ng = 30, 60
    theta = np.arccos(np.clip(dirs[..., 2], -1.0, 1.0))
    gamma = np.arccos(np.clip(dirs @ np.asarray(sun_dir, dtype=np.float32), -1.0, 1.0))
    ti = np.clip((theta / (math.pi / 2) * nt).astype(int), 0, nt - 1)
    gi = np.clip((gamma / math.pi * ng).astype(int), 0, ng - 1)
    bins = ti * ng + gi
    ok = valid & (dirs[..., 2] > 0.0)

    def smooth_grid(values):
        return _blur(_fill_nan(values.reshape(nt, ng)), 1)

    def at(grid):
        tf = np.clip(theta / (math.pi / 2) * nt - 0.5, 0, nt - 1)
        gf = np.clip(gamma / math.pi * ng - 0.5, 0, ng - 1)
        t0, g0 = np.floor(tf).astype(int), np.floor(gf).astype(int)
        t1, g1 = np.minimum(t0 + 1, nt - 1), np.minimum(g0 + 1, ng - 1)
        ft, fg = tf - t0, gf - g0
        return ((grid[t0, g0] * (1 - fg) + grid[t0, g1] * fg) * (1 - ft) +
                (grid[t1, g0] * (1 - fg) + grid[t1, g1] * fg) * ft)

    if reference is None:
        # The sun and its glare tell nothing about the sky or the clouds.
        near_sun = gamma < math.radians(2.5)
        ok_sky = ok & ~near_sun & (lum < 30.0 * np.median(lum[ok]))
        sky_blue = smooth_grid(_binned(blue_s, bins, nt * ng, lambda a: np.percentile(a, 97),
                                       ok_sky))

        gap = ok_sky & (blue_s >= 0.92 * np.maximum(at(sky_blue), 1e-6))
        sky_rgb = [smooth_grid(_binned(rgb[..., k], bins, nt * ng, np.median, gap))
                   for k in range(3)]
    else:
        sky_blue, sky_rgb = reference
    ref_blue = np.maximum(at(sky_blue), 1e-6)
    ref_rgb = np.stack([np.maximum(at(g), 1e-6) for g in sky_rgb], -1)
    ref_lum = 0.2126 * ref_rgb[..., 0] + 0.7152 * ref_rgb[..., 1] + 0.0722 * ref_rgb[..., 2]

    a = 1.0 - np.clip(blue_s / ref_blue, 0.0, 1.0)
    # Below `noise` it is the photo's grain and the unevenness of the clear sky: fade in softly, no
    # hard edge.
    opacity = _smoothstep(0.5 * noise, 2.5 * noise, a) * a
    # Where all the blue is hidden the blueness can't tell thick from thicker: there the brightness
    # of the photo can (brighter, thicker). Without it these parts would be flat.
    if lum_blur is None:
        lum_blur = _blur(lum, max(2, min(shape) // 80)) if lum.ndim == 2 else lum
    lum_rel = np.log2(np.maximum(lum, 1e-9) / np.maximum(lum_blur, 1e-9))
    full = _smoothstep(0.6, 1.0, a)
    opacity *= 1.0 - full * np.clip(0.35 - 0.6 * lum_rel, 0.0, 0.6)
    # Thin cirrus hardly dims the sky behind it but adds light to it, a lot toward the sun (ice
    # scatters forward): that light, against the clear sky there, tells where it is too. A cloud
    # counts as full where it adds about the sky's own light, more toward the sun. Below that it is
    # the unevenness of the clear sky.
    gain = lum - ref_lum * (1.0 - np.clip(a, 0.0, 1.0))
    rel_light = gain / ref_lum
    if rel_light.ndim == 2:
        rel_light = _blur(rel_light, 1)
    full_light = 1.5 * (1.0 + 3.0 * np.exp(-np.degrees(gamma) / 12.0))
    # Bright but translucent: light alone never makes a cloud more than about half opaque.
    lit = 0.6 * _smoothstep(0.08, 0.3, rel_light) * np.clip(rel_light / full_light, 0.0, 1.0)
    opacity = np.maximum(opacity, lit)
    # A soft top: even thick cirrus lets a little sky through.
    opacity = np.where(opacity < 0.7, opacity, 0.7 + 0.25 * np.tanh((opacity - 0.7) / 0.25))
    # Land, trees, buildings: darker than the sky (not near the sun, where the sky is too bright
    # to compare with).
    elevation = np.degrees(math.pi / 2 - theta)
    land = (1.0 - _smoothstep(0.35, 0.6, lum / ref_lum)) * (1.0 - _smoothstep(15.0, 25.0, elevation))
    land *= _smoothstep(8.0, 14.0, np.degrees(gamma))
    opacity *= 1.0 - land
    opacity[~ok] = 0.0

    # The light and shade inside the clouds: their brightness against the clouds around them,
    # gently (0.5 neutral). Thin parts show mostly the sky, so they count less.
    log_lum = np.log2(np.maximum(lum, 1e-9))
    if around is None and lum.ndim == 2:
        r = max(2, min(shape) // 60)
        w = np.clip(opacity, 0.0, 1.0).astype(np.float32)
        around = _blur(log_lum * w, r) / np.maximum(_blur(w, r), 1e-4)
    if around is not None:
        rel = (log_lum - around) * _smoothstep(0.05, 0.4, opacity)
    else:
        rel = np.zeros_like(lum)
    shading = 0.5 + 0.25 * np.tanh(1.5 * rel)
    if details:
        extra = {'reference': (sky_blue, sky_rgb), 'lum_blur': lum_blur, 'around': around}
        return opacity.astype(np.float32), shading.astype(np.float32), extra
    return opacity.astype(np.float32), shading.astype(np.float32)


def _map_pixel(d, size):
    """Pixel (x, y, rows bottom to top) of the cloud map in direction d (map frame)."""
    rho = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    r = np.arccos(np.clip(d[..., 2], -1.0, 1.0)) * (2.0 / math.pi)
    k = np.where(rho > 1e-9, 0.5 * r / np.maximum(rho, 1e-9), 0.0)
    return (0.5 + d[..., 0] * k) * size - 0.5, (0.5 + d[..., 1] * k) * size - 0.5


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def _fill_glare(opacity, shading, dirs, sun, inside, rotation, radius_deg=16.0):
    """The sun and its glare hide the clouds around it: fill that circle with real clouds from
    beside it at the same height above the horizon (so with the same perspective), like a clone
    stamp, blended in softly. `dirs` and `sun` are in the frame of the photo, the maps in the
    frame of the cloud map."""
    sun = np.asarray(sun, dtype=np.float32)
    gamma = np.arccos(np.clip(dirs @ sun, -1.0, 1.0))
    radius = math.radians(radius_deg)
    hole = inside & (gamma < radius)
    if not hole.any():
        return opacity, shading
    d = dirs[hole]
    g = gamma[hole]
    # The turn around the zenith whose clouds best match those just around the glare (a ring
    # around it), clear of the glare itself.
    ca, sa = math.cos(rotation), math.sin(rotation)

    def to_map(v, angle):
        c, s_ = math.cos(angle + rotation), math.sin(angle + rotation)
        return np.stack([v[:, 0] * c + v[:, 1] * s_, -v[:, 0] * s_ + v[:, 1] * c, v[:, 2]], 1)

    ring = inside & (gamma >= radius) & (gamma < 1.5 * radius)
    ring_dirs = dirs[ring]
    step = max(1, len(ring_dirs) // 4000)
    ring_dirs = ring_dirs[::step]
    rx, ry = _map_pixel(to_map(ring_dirs, 0.0), opacity.shape[0])
    ring_op = _bilinear(opacity, rx, ry)
    best, best_err = None, None
    for degrees in list(range(25, 180, 5)) + list(range(-25, -180, -5)):
        angle = math.radians(degrees)
        c, s_ = math.cos(-angle), math.sin(-angle)
        # The source of the glare circle must lie outside the glare.
        moved_sun = np.array([sun[0] * c - sun[1] * s_, sun[0] * s_ + sun[1] * c, sun[2]])
        if math.acos(float(np.clip(moved_sun @ sun, -1.0, 1.0))) < 2.2 * radius:
            continue
        sx, sy = _map_pixel(to_map(ring_dirs, angle), opacity.shape[0])
        src_op = _bilinear(opacity, sx, sy)
        err = float(np.mean((src_op - ring_op) ** 2))
        if best_err is None or err < best_err:
            best, best_err = angle, err
            gain = float(np.clip(ring_op.mean() / max(src_op.mean(), 1e-3), 0.6, 2.0))
    _fill_glare.last = (best, gain) if best is not None else None
    if best is None:
        return opacity, shading
    return _clone_glare(opacity, shading, d, g, hole, sun, rotation, radius, best, gain)


def _clone_glare(opacity, shading, d, g, hole, sun, rotation, radius, best, gain):
    """Fill the glare circle (`hole`, directions `d` at angles `g` from the sun) with the clouds
    turned by `best` around the zenith, scaled by `gain`."""
    turn = best + rotation
    ct, st = math.cos(turn), math.sin(turn)
    # The source in the frame of the cloud map (photo frame turned by `rotation`, plus `turn`).
    src = np.stack([d[:, 0] * ct + d[:, 1] * st, -d[:, 0] * st + d[:, 1] * ct, d[:, 2]], axis=1)
    px, py = _map_pixel(src, opacity.shape[0])
    keep = _smoothstep(0.4 * radius, radius, g)
    opacity = opacity.copy()
    shading = shading.copy()
    clone = np.clip(_bilinear(opacity, px, py) * gain, 0.0, 0.95)
    opacity[hole] = opacity[hole] * keep + clone * (1.0 - keep)
    shading[hole] = shading[hole] * keep + _bilinear(shading, px, py) * (1.0 - keep)
    return opacity, shading


def cloud_map_from_equirectangular(pixels, size=2048, rotation=0.0, horizon_fade=7.0):
    """Cloud map (size x size x 4, rows bottom to top) from an equirectangular HDRI
    (H x W x 3 or 4, rows bottom to top). `rotation` (radians) turns the photo around the zenith;
    `horizon_fade` (degrees) fades out the clouds just above the horizon, where photos have haze
    and land."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(all='ignore'):
            opacity, shading = _cloud_map_from_equirectangular(pixels, size, rotation,
                                                               horizon_fade)
    out = np.zeros((size, size, 4), dtype=np.float32)
    out[..., 0] = opacity
    out[..., 1] = shading
    out[..., 3] = 1.0
    return out


def _reduce(pixels, max_width):
    """`pixels` scaled down by whole blocks (averaged) to at most `max_width` wide."""
    h, w = pixels.shape[:2]
    step = int(math.ceil(w / max_width)) if w > max_width else 1
    if step == 1:
        return pixels
    hh, ww = (h // step) * step, (w // step) * step
    return pixels[:hh, :ww].reshape(hh // step, step, ww // step, step, -1).mean(axis=(1, 3))


def _map_directions(size, rows, rotation):
    """Directions (frame of the photo) of the rows `rows` (a slice) of a size x size cloud map,
    and the mask of those above the horizon."""
    c = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(c, c[rows])
    r = np.sqrt(x * x + y * y)
    inside = r <= 1.0
    theta = np.minimum(r, 1.0) * (math.pi / 2.0)
    phi = np.arctan2(y, x)
    s = np.sin(theta)
    dx, dy = s * np.cos(phi), s * np.sin(phi)
    if rotation:
        ca, sa = math.cos(rotation), math.sin(rotation)
        dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
    return np.stack([dx, dy, np.cos(theta)], axis=-1).astype(np.float32), inside


# Size of the map on which the clear sky, the sun's glare and the light around the clouds are
# worked out. Larger maps take these from it and only add the finer detail of the photo.
ANALYSIS_SIZE = 2048


def _cloud_map_from_equirectangular(pixels, size, rotation, horizon_fade):
    rgb_full = pixels[..., :3]
    # `dirs` are directions of the photo, so is its sun.
    sun, sun_color = find_sun(pixels, with_color=True)
    small = min(size, ANALYSIS_SIZE)
    # The photo at about the detail of the small map (sampling it finer would only alias).
    source = _reduce(rgb_full, 4 * small) if size > small else rgb_full
    dirs, inside = _map_directions(small, slice(None), rotation)
    rgb = sample_equirectangular(source.astype(np.float32), dirs)
    opacity, shading, extra = separate_clouds(rgb, dirs, sun, valid=inside, sun_color=sun_color,
                                              details=True)
    del rgb, source
    _fill_glare.last = None
    opacity, shading = _fill_glare(opacity, shading, dirs, sun, inside, rotation)
    glare = _fill_glare.last

    if size > small:
        # The fine detail: the same separation of every part of the large map, with the clear
        # sky and the light around the clouds of the small map.
        lum_blur, around = extra['lum_blur'], extra['around']
        scale = small / size
        big_opacity = np.zeros((size, size), dtype=np.float32)
        big_shading = np.full((size, size), 0.5, dtype=np.float32)
        strip, pad = 256, 2
        xs = (np.arange(size, dtype=np.float32) + 0.5) * scale - 0.5
        for y0 in range(0, size, strip):
            y1 = min(size, y0 + strip)
            a0, a1 = max(0, y0 - pad), min(size, y1 + pad)
            d, ins = _map_directions(size, slice(a0, a1), rotation)
            if not ins.any():
                continue
            ys = (np.arange(a0, a1, dtype=np.float32) + 0.5) * scale - 0.5
            gx, gy = np.meshgrid(xs, ys)
            part = sample_equirectangular(rgb_full, d)
            op, sh = separate_clouds(part, d, sun, valid=ins, sun_color=sun_color,
                                     reference=extra['reference'],
                                     lum_blur=_bilinear(lum_blur, gx, gy),
                                     around=_bilinear(around, gx, gy))
            big_opacity[y0:y1] = op[y0 - a0:y0 - a0 + (y1 - y0)]
            big_shading[y0:y1] = sh[y0 - a0:y0 - a0 + (y1 - y0)]
        if glare is not None:
            big_opacity, big_shading = _fill_glare_large(big_opacity, big_shading, sun, rotation,
                                                         glare)
        opacity, shading = big_opacity, big_shading

    for y0 in range(0, size, 512):
        rows = slice(y0, min(size, y0 + 512))
        d, _ins = _map_directions(size, rows, 0.0)
        elevation = np.degrees(np.arcsin(np.clip(d[..., 2], 0.0, 1.0)))
        opacity[rows] *= _smoothstep(0.0, max(horizon_fade, 1e-3), elevation)
    return opacity, shading


def _fill_glare_large(opacity, shading, sun, rotation, glare, radius_deg=16.0):
    """The glare fill of a large map, with the turn and gain found on the small map: only the
    rows of the map that the glare circle touches are visited."""
    size = opacity.shape[0]
    radius = math.radians(radius_deg)
    best, gain = glare
    for y0 in range(0, size, 512):
        rows = slice(y0, min(size, y0 + 512))
        dirs, inside = _map_directions(size, rows, rotation)
        gamma = np.arccos(np.clip(dirs @ np.asarray(sun, dtype=np.float32), -1.0, 1.0))
        hole_rows = inside & (gamma < radius)
        if not hole_rows.any():
            continue
        hole = np.zeros(opacity.shape, dtype=bool)
        hole[rows] = hole_rows
        opacity, shading = _clone_glare(opacity, shading, dirs[hole_rows], gamma[hole_rows],
                                        hole, sun, rotation, radius, best, gain)
    return opacity, shading


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


def auto_size(width):
    """Cloud map size for an HDRI `width` pixels wide: 8192 (about 45 pixels per degree, the
    detail of a 16K HDRI), larger never (the memory when rendering)."""
    return 8192


def make_cloud_map(source_image, name=None, size=2048, rotation=0.0, horizon_fade=7.0):
    """New Blender image (packed into the file) with the cloud map of an equirectangular HDRI.
    8 bits per channel: plenty for opacity and shading, and a quarter of the memory of floats."""
    import bpy

    if size <= 0:
        size = auto_size(source_image.size[0])
    pixels = image_pixels(source_image, max_width=max(8192, 4 * size))
    data = cloud_map_from_equirectangular(pixels, size=size, rotation=rotation,
                                          horizon_fade=horizon_fade)
    del pixels
    name = name or "Clouds " + source_image.name
    image = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False,
                                is_data=True)
    image.colorspace_settings.is_data = True
    image.pixels.foreach_set(data.ravel())
    del data
    image.pack()
    return image


# -----------------------------------------------------------------------------
# Clouds from satellite images (top-down maps of a cloud layer).

def _sat_pyramid(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels):
        a = pyr[-1]
        h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
        pyr.append(a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))
    return pyr

def _sat_mirror(u):
    """Mirrored repeat of 0..1 (so the map tiles without seams)."""
    u = np.mod(u, 2.0)
    return np.where(u > 1.0, 2.0 - u, u)

def _sat_sample(img, u, v):
    """Bilinear lookup at u, v in 0..1 (mirrored repeat), rows bottom to top."""
    h, w = img.shape
    x = _sat_mirror(u) * (w - 1)
    y = _sat_mirror(v) * (h - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 2); y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 2)
    fx = x - x0; fy = y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)

def cirrus_opacity_from_band(c):
    """Opacity of the cirrus from Sentinel-2 band 10 (top of atmosphere reflectance at 1375 nm,
    where the water vapour below hides the sea and the low clouds)."""
    floor = np.percentile(c, 5) + 0.0006
    x = np.maximum(c - floor, 0.0)
    # Thin cirrus reflects about 1-2 %, thick 5 % and more.
    op = 1.0 - np.exp(-x * 45.0)
    return _smoothstep(0.0, 0.08, op) * op

def cloud_map_from_planar(op_src, pixel_m, size=2048, altitude_m=9000.0, rotation=0.0,
                          horizon_fade=3.0, shading_src=None):
    """Fisheye cloud map (as from the photos) of a top-down map of a cloud layer at altitude_m:
    the map is laid out over the sky at that height, repeated mirrored beyond its edges, and
    filtered more where it is seen from far away (toward the horizon)."""
    h, w = op_src.shape
    width_km = w * pixel_m / 1000.0
    height_km = h * pixel_m / 1000.0
    alt = altitude_m / 1000.0
    pyr = _sat_pyramid(op_src, 7)
    out = np.zeros((size, size, 2), np.float32)
    c = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    pixel_angle = (math.pi / 2) / (size / 2)
    for y0 in range(0, size, 512):
        X, Y = np.meshgrid(c, c[y0:y0 + 512])
        r = np.sqrt(X * X + Y * Y)
        inside = r < 1.0
        theta = np.minimum(r, 0.9999) * (math.pi / 2)
        phi = np.arctan2(Y, X) + rotation
        dz = np.cos(theta)
        # distance to the layer (curved earth) and where it is
        a = (1 - dz * dz) / (2 * 6371.0)
        t = (-dz + np.sqrt(dz * dz + 4 * a * alt)) / np.maximum(2 * a, 1e-12)
        t = np.where(a < 1e-12, alt / np.maximum(dz, 1e-6), t)
        ground = t * np.sin(theta)
        px = ground * np.cos(phi); py = ground * np.sin(phi)
        u = 0.5 + px / width_km
        v = 0.5 + py / height_km
        # footprint of a map pixel on the layer (km), stretched along the view toward the horizon
        foot = t * pixel_angle / np.maximum(dz + 0.02, 0.02)
        level = np.clip(np.log2(np.maximum(foot / (pixel_m / 1000.0), 1e-6)), 0, len(pyr) - 1.001)
        l0 = np.floor(level).astype(int); fl = level - l0
        res = np.zeros_like(X)
        for L in range(len(pyr) - 1):
            m = l0 == L
            if not m.any():
                continue
            a0 = _sat_sample(pyr[L], u[m], v[m]); a1 = _sat_sample(pyr[L + 1], u[m], v[m])
            res[m] = a0 * (1 - fl[m]) + a1 * fl[m]
        el = 90.0 - np.degrees(theta)
        res *= _smoothstep(0.0, horizon_fade, el)
        res[~inside] = 0
        out[y0:y0 + 512, :, 0] = res
    out[..., 1] = 0.5
    return out


def make_cloud_map_from_satellite(band_image, pixel_m=20.0, name=None, size=8192,
                                  altitude_m=9000.0, rotation=0.0):
    """New Blender image (packed) with the cirrus of a Sentinel-2 band 10 image (the cirrus band),
    laid out over the sky at altitude_m."""
    import bpy

    w, h = band_image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    band_image.pixels.foreach_get(buf)
    band = buf.reshape(h, w, 4)[..., 0].copy()
    del buf
    op = cirrus_opacity_from_band(band)
    data = cloud_map_from_planar(op, pixel_m, size=size, altitude_m=altitude_m, rotation=rotation)
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[..., :2] = data
    rgba[..., 3] = 1.0
    del data
    name = name or "Clouds " + band_image.name
    image = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False, is_data=True)
    image.colorspace_settings.is_data = True
    image.pixels.foreach_set(rgba.ravel())
    del rgba
    image.pack()
    return image
