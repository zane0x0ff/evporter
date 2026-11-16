import os
import struct
import numpy as np

from . import formats as F

MAGIC = 0x316f6d74
NAMES = {0x736f70: 'pos', 0x6d6961: 'aim', 0x61636f66: 'foca'}
CODES = {v: k for k, v in NAMES.items()}
A3 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
FILM_IN = 0.9449
DEFAULT_BLOCK = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1080, 1920, 1920 / 1080, 0.1, 6000, FILM_IN, 24, 24, 0, 0]


def is_camera(b):
    if len(b) < 0x60 or struct.unpack_from('<I', b, 0)[0] != MAGIC:
        return False
    body, blk = struct.unpack_from('<I', b, 4)[0], struct.unpack_from('<H', b, 10)[0]
    return blk != 0 and body != 0x10 and blk + 0x50 <= body and body + 0x60 <= len(b)


class CameraTmo:
    """a parsed camera motion: block (20 floats), num_frames, unit (track units -> metres), curves
    {track: [9 x (frames, values) or None]}"""

    def __init__(self, b, name=''):
        if not is_camera(b):
            raise ValueError('%s: not a camera motion (tmo1 with a camera block)' % name)
        self.name = name
        flags = struct.unpack_from('<H', b, 8)[0]
        tscale = struct.unpack_from('<f', b, 12)[0] or 1.0
        body, blk = struct.unpack_from('<I', b, 4)[0], struct.unpack_from('<H', b, 10)[0]
        self.block = list(struct.unpack_from('<20f', b, blk))
        q = lambda o: struct.unpack_from('<Q', b, o)[0]
        ofs_mot, ofs_hash, ofs_cur, ofs_key = q(body + 8), q(body + 0x28), q(body + 0x38), q(body + 0x48)
        self.num_frames, ntracks = struct.unpack_from('<hh', b, ofs_mot)
        first = struct.unpack_from('<I', b, ofs_mot + 8)[0]
        ks = 16 if flags & 2 else 8 if flags & 1 else 4
        hashes = struct.unpack_from('<%dI' % ntracks, b, ofs_hash) if ntracks > 0 else ()
        self.curves = {}
        for t, h in enumerate(hashes):
            name_t = NAMES.get(h)
            if name_t is None and t < 3:
                name_t = ('pos', 'aim', 'foca')[t]
            if name_t is None or name_t in self.curves:
                continue
            chans = []
            for c in range(9):
                f0, n, _, _ = struct.unpack_from('<QHHH', b, ofs_cur + 16 * (first + 9 * t + c))
                if n == 0:
                    chans.append(None)
                    continue
                o = ofs_key + ks * f0
                fr = np.array([struct.unpack_from('<H', b, o + ks * k)[0] for k in range(n)], float)
                if ks == 4:
                    va = np.array([struct.unpack_from('<h', b, o + 4 * k + 2)[0] for k in range(n)], float) / tscale
                else:
                    va = np.array([struct.unpack_from('<f', b, o + ks * k + 4)[0] for k in range(n)], float)
                chans.append((fr, va))
            self.curves[name_t] = chans
        if 'pos' not in self.curves or 'aim' not in self.curves:
            raise ValueError('%s: camera motion without pos / aim tracks' % name)
        p0 = self.vec('pos', np.array([0.0]), unit=1.0)[0]
        d = np.abs(p0 - np.array(self.block[3:6])).sum()
        n = np.abs(np.array(self.block[3:6])).sum()
        self.unit = 0.01 if d <= 0.05 * n + 1.0 else 1.0

    @staticmethod
    def sample(curve, frames, default=0.0):
        """the game's linear interpolation (before the first key: its value; after the last: the last)"""
        frames = np.asarray(frames, float)
        if curve is None:
            return np.full(frames.shape, default)
        fr, va = curve
        return np.interp(frames, fr, va)

    def vec(self, track, frames, unit=None):
        """(n, 3) game-space translate of a track, metres"""
        u = self.unit if unit is None else unit
        ch = self.curves.get(track) or [None] * 9
        return np.stack([self.sample(ch[k], frames) for k in range(3)], -1) * u

    def focal(self, frames):
        """focal length (mm) per frame"""
        ch = (self.curves.get('foca') or [None] * 9)[2]
        if ch is None:
            return np.full(len(frames), self.block[17] if self.block[17] > 0 else 24.0)
        f = self.sample(ch, frames) * self.unit * 100.0
        return np.where(f > 0, f, self.block[17] if self.block[17] > 0 else 24.0)

    @property
    def film_mm(self):
        return (self.block[15] if self.block[15] > 0 else FILM_IN) * 25.4

    def key_frames(self, track, axis):
        ch = (self.curves.get(track) or [None] * 9)[axis]
        return None if ch is None else ch[0]


def cameras_in(data, filename=''):
    """[(name, CameraTmo)] of a stage catalogue (.cat) or a single camera .tmo2"""
    if is_camera(data):
        return [(os.path.splitext(os.path.basename(filename))[0] or 'camera', CameraTmo(data, filename))]
    out = []
    try:
        entries = F.parse_cat(data)
    except Exception:
        return out
    for n, k, o, s, _ in entries:
        b = data[o:o + s]
        if is_camera(b):
            try:
                out.append((n, CameraTmo(b, n)))
            except ValueError:
                pass
    return out


def simplify(frames, values, tol):
    """keys (frames, values) that reproduce the samples within tol under linear interpolation"""
    n = len(values)
    if n <= 2:
        return list(range(n))
    keep = [0]
    s = 0
    e = 2
    while e < n:
        t = (frames[s + 1:e] - frames[s]) / (frames[e] - frames[s])
        lin = values[s] + t * (values[e] - values[s])
        if np.abs(lin - values[s + 1:e]).max() > tol:
            keep.append(e - 1)
            s = e - 1
        e += 1
    keep.append(n - 1)
    return keep


def encode(frames, eye, aim, focal_mm, block, tol_m=1e-5, tol_mm=1e-4):
    """camera motion bytes. frames: game frames (0..N), eye / aim: (n, 3) game space metres, focal_mm: (n,).
    Tracks in metres with foca = focal / 100, as every stage camera but bg50's (bg00 ... bg17; the block's eye
    stays in cm, so the unit test finds metres); keys only where linear interpolation would not reproduce the
    samples."""
    frames = np.asarray(frames, float)
    tracks = []
    for code, arr in ((CODES['pos'], np.asarray(eye, float)), (CODES['aim'], np.asarray(aim, float))):
        chans = [None] * 9
        for c in range(3):
            v = arr[:, c]
            k = simplify(frames, v, tol_m)
            chans[c] = (frames[k], v[k])
        tracks.append((code, chans))
    chans = [None] * 9
    v = np.asarray(focal_mm, float) / 100.0
    k = simplify(frames, v, tol_mm / 100.0)
    chans[2] = (frames[k], v[k])
    tracks.append((CODES['foca'], chans))
    num_frames = int(frames[-1]) if len(frames) else 0

    keys, curves = bytearray(), bytearray()
    nkeys = 0
    for _, chans in tracks:
        for ch in chans:
            if ch is None:
                curves += struct.pack('<QHHHH', nkeys, 0, 0, 0, 0)
                continue
            fr, va = ch
            for f, x in zip(fr, va):
                keys += struct.pack('<HHf', int(f), 0, float(x))
            curves += struct.pack('<QHHHH', nkeys, len(fr), 0, int(fr[-1]), 0)
            nkeys += len(fr)
    BODY, o_hash = 0x60, 0xC0
    o_mot = (o_hash + 4 * len(tracks) + 15) & ~15
    o_cur = o_mot + 16
    o_key = o_cur + len(curves)
    out = bytearray(struct.pack('<IIHHf', MAGIC, BODY, 0x01, 0x10, 0.0))
    out += struct.pack('<20f', *block)
    body = bytearray(0x60)
    struct.pack_into('<I', body, 0x00, 1)
    struct.pack_into('<Q', body, 0x08, o_mot)
    struct.pack_into('<Q', body, 0x28, o_hash)
    struct.pack_into('<I', body, 0x30, len(curves) // 16)
    struct.pack_into('<Q', body, 0x38, o_cur)
    struct.pack_into('<I', body, 0x40, nkeys)
    struct.pack_into('<Q', body, 0x48, o_key)
    out += body
    out += b''.join(struct.pack('<I', h) for h, _ in tracks)
    out += b'\0' * (o_mot - len(out))
    out += struct.pack('<hhIII', num_frames, len(tracks), 0, 0, 0)
    out += curves + keys
    return bytes(out)


def make_block(eye0, aim0, focal0, res_x=1920, res_y=1080):
    """the 20-float camera block for frame 0. The loader reads only aspect, near, far, film back and focal length
    (+0x30 .. +0x44); the eye (cm) at +0x0C is what tells the track unit. Floats 0-2 are a unit vector whose meaning is
    unknown (not eye - aim of the tracks); here: normalize(eye - aim)."""
    b = list(DEFAULT_BLOCK)
    d = np.asarray(eye0) - np.asarray(aim0)
    dist = float(np.linalg.norm(d))
    b[0:3] = (d / dist).tolist() if dist > 1e-9 else [0.0, 0.0, 1.0]
    b[3:6] = (np.asarray(eye0) * 100.0).tolist()
    b[7] = dist * 100.0
    b[10], b[11], b[12] = float(res_y), float(res_x), float(res_x) / float(res_y)
    b[17] = float(focal0)
    half = np.arctan((FILM_IN * 25.4) / (2.0 * float(focal0)))
    b[18], b[19] = float(half), float(np.degrees(half))
    return b


def look_rotation(eye_b, aim_b):
    """Blender 3x3 (columns = camera X, Y, Z) looking from eye to aim with world +Z up (the game's fixed +Y up)"""
    f = np.asarray(aim_b, float) - np.asarray(eye_b, float)
    f /= max(np.linalg.norm(f), 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)
