import struct

import numpy as np


def _blocks(img):
    """(h, w, 4) uint8 -> (nby, nbx, 16, 4) float, padded by edge replication to multiples of 4"""
    h, w = img.shape[:2]
    H, Wd = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    if (H, Wd) != (h, w):
        img = np.pad(img, ((0, H - h), (0, Wd - w), (0, 0)), mode='edge')
    b = img.reshape(H // 4, 4, Wd // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(H // 4, Wd // 4, 16, 4)
    return b.astype(np.float64)


def _565(c):
    c = np.clip(np.rint(c), 0, 255).astype(np.int64)
    return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)


def _rgb(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.float64)


def _color_block(b):
    """(n, 16, 3) -> (n,) u16 c0, c1, (n,) u32 indices (4-colour mode, c0 > c1)"""
    mx, mn = b.max(1), b.min(1)
    c0, c1 = _565(mx), _565(mn)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    p0, p1 = _rgb(c0), _rgb(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)
    d = ((b[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    idx = d.argmin(-1)
    idx = np.where((c0 == c1)[:, None], 0, idx)
    bits = (idx.astype(np.uint64) << (2 * np.arange(16, dtype=np.uint64))).sum(1).astype(np.uint32)
    return c0.astype(np.uint16), c1.astype(np.uint16), bits


def _alpha_block(a):
    """(n, 16) -> (n, 8) bytes of a DXT5 alpha block (8-level mode, a0 > a1)"""
    a0 = np.clip(np.rint(a.max(1)), 0, 255).astype(np.int64)
    a1 = np.clip(np.rint(a.min(1)), 0, 255).astype(np.int64)
    a1 = np.where(a0 == a1, np.maximum(a0 - 1, 0), a1)
    a0 = np.where(a0 == a1, a1 + 1, a0)
    k = np.arange(8)
    lv = np.where(k[None, :] == 0, a0[:, None], np.where(k[None, :] == 1, a1[:, None],
                  ((8 - k[None, :]) * a0[:, None] + (k[None, :] - 1) * a1[:, None]) / 7.0))
    idx = np.abs(a[:, :, None] - lv[:, None, :]).argmin(-1).astype(np.uint64)
    bits = (idx << (3 * np.arange(16, dtype=np.uint64))).sum(1)
    out = np.zeros((len(a), 8), np.uint8)
    out[:, 0], out[:, 1] = a0, a1
    for i in range(6):
        out[:, 2 + i] = ((bits >> np.uint64(8 * i)) & np.uint64(255)).astype(np.uint8)
    return out


def encode_level(img, dxt5):
    b = _blocks(img)
    nby, nbx = b.shape[:2]
    b = b.reshape(-1, 16, 4)
    c0, c1, bits = _color_block(b[:, :, :3])
    col = np.zeros((len(b), 8), np.uint8)
    col[:, 0:2] = c0.view(np.uint8).reshape(-1, 2)
    col[:, 2:4] = c1.view(np.uint8).reshape(-1, 2)
    col[:, 4:8] = bits.view(np.uint8).reshape(-1, 4)
    if dxt5:
        return np.concatenate([_alpha_block(b[:, :, 3]), col], 1).tobytes()
    return col.tobytes()


def encode(rgba, mipmaps=False, force_dxt5=None):
    """(h, w, 4) uint8, top row first -> DDS bytes"""
    rgba = np.ascontiguousarray(np.asarray(rgba, np.uint8))
    h, w = rgba.shape[:2]
    dxt5 = bool((rgba[..., 3] < 255).any()) if force_dxt5 is None else force_dxt5
    levels = [rgba]
    if mipmaps:
        cur = rgba.astype(np.float64)
        while max(cur.shape[:2]) > 1:
            hh, ww = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
            c = cur[:hh * 2 if cur.shape[0] > 1 else 1, :ww * 2 if cur.shape[1] > 1 else 1]
            c = c.reshape(hh, -1, ww, c.shape[1] // ww, 4).mean((1, 3))
            cur = c
            levels.append(np.clip(np.rint(c), 0, 255).astype(np.uint8))
    data = b''.join(encode_level(l, dxt5) for l in levels)
    top = ((w + 3) // 4) * ((h + 3) // 4) * (16 if dxt5 else 8)
    flags = 0x81007 | (0x20000 if mipmaps else 0)
    caps = 0x401008 if mipmaps else 0x1000
    hdr = struct.pack('<4s7I44x', b'DDS ', 124, flags, h, w, top, 0, len(levels) if mipmaps else 0)
    hdr += struct.pack('<2I4s5I', 32, 4, b'DXT5' if dxt5 else b'DXT1', 0, 0, 0, 0, 0)
    hdr += struct.pack('<5I', caps, 0, 0, 0, 0)
    assert len(hdr) == 128
    return hdr + data


def has_mipmaps(dds):
    return dds is not None and len(dds) >= 128 and struct.unpack_from('<I', dds, 28)[0] > 1
