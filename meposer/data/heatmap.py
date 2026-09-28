import numpy as np


def render_heatmaps(p2d, src_size, out_size, sigma):
    w_src, h_src = src_size
    w, h = out_size
    scale = np.array([w / w_src, h / h_src], np.float32)
    vis = (p2d[:, 0] >= 0) & (p2d[:, 0] < w_src) & (p2d[:, 1] >= 0) & (p2d[:, 1] < h_src)
    xs = np.arange(w, dtype=np.float32)[None, None, :]
    ys = np.arange(h, dtype=np.float32)[None, :, None]
    centers = p2d * scale
    dx = xs - centers[:, 0, None, None]
    dy = ys - centers[:, 1, None, None]
    hm = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)).astype(np.float32)
    hm[~vis] = 0.0
    return hm, vis


def heatmap_argmax(hm):
    j, h, w = hm.shape[-3:]
    flat = hm.reshape(*hm.shape[:-2], -1)
    idx = flat.argmax(-1)
    conf = flat.max(-1)
    return np.stack([idx % w, idx // w], -1).astype(np.float32), conf


def heatmap_subpixel(hm):
    """Argmax refined by the confidence-weighted mean over the 3x3 neighbourhood (sub-cell accuracy)."""
    peaks, conf = heatmap_argmax(hm)
    h, w = hm.shape[-2:]
    flat = hm.reshape(-1, h, w)
    pk = peaks.reshape(-1, 2).astype(int)
    out = np.empty_like(pk, dtype=np.float32)
    for i, (x, y) in enumerate(pk):
        x0, x1, y0, y1 = max(0, x - 1), min(w, x + 2), max(0, y - 1), min(h, y + 2)
        patch = np.clip(flat[i, y0:y1, x0:x1], 0, None)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        s = patch.sum()
        out[i] = (x, y) if s <= 0 else ((patch * xs).sum() / s, (patch * ys).sum() / s)
    return out.reshape(peaks.shape), conf
