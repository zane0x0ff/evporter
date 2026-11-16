import struct
import numpy as np

from . import formats as F

ROT4 = 2 * np.pi / 65536.0
MAGIC = 0x316f6d74


class Tmo:
    def __init__(self, data, name=''):
        if len(data) < 0x60 or struct.unpack_from('<I', data, 0)[0] != MAGIC:
            raise ValueError('%s: not a tmo1 motion' % name)
        self.name = name
        self.d = data
        self.flags = struct.unpack_from('<H', data, 8)[0]
        self.t_scale = struct.unpack_from('<f', data, 12)[0] or 1.0
        B = 0x10
        q = lambda o: struct.unpack_from('<Q', data, o)[0]
        ofs_mot = q(B + 8)
        ofs_hash, self.ofs_curves, self.ofs_keys = q(B + 0x28), q(B + 0x38), q(B + 0x48)
        self.key_size = 16 if self.flags & 2 else 8 if self.flags & 1 else 4
        self.num_frames, self.num_tracks = struct.unpack_from('<hh', data, ofs_mot)
        self.first_curve = struct.unpack_from('<I', data, ofs_mot + 8)[0]
        self.hashes = list(struct.unpack_from('<%dI' % self.num_tracks, data, ofs_hash)) if self.num_tracks > 0 else []
        self.track = {}
        for i, h in enumerate(self.hashes):
            self.track.setdefault(h, i)

    def _curve(self, c):
        o = self.ofs_curves + 16 * (self.first_curve + c)
        first, n, _, last = struct.unpack_from('<QHHH', self.d, o)
        return first, n, last

    def _keys(self, c):
        """frames (int) and values (float, raw) and the linear flag per key of curve c"""
        first, n, last = self._curve(c)
        if n == 0:
            return None
        o = self.ofs_keys + self.key_size * first
        if self.key_size == 16:
            a = np.frombuffer(self.d, dtype=np.dtype([('f', '<u2'), ('m', '<u2'), ('v', '<f4'), ('x', '<f4', 2)]), count=n, offset=o)
            return a['f'].astype(np.int64), a['v'].astype(np.float64), np.isin(a['m'] & 0xf, (2, 3)), last
        if self.key_size == 8:
            a = np.frombuffer(self.d, dtype=np.dtype([('f', '<u2'), ('m', '<u2'), ('v', '<f4')]), count=n, offset=o)
            return a['f'].astype(np.int64), a['v'].astype(np.float64), np.zeros(n, bool), last
        a = np.frombuffer(self.d, dtype=np.dtype([('f', '<u2'), ('v', '<i2')]), count=n, offset=o)
        return a['f'].astype(np.int64), a['v'].astype(np.float64), np.zeros(n, bool), last

    def sample_curve(self, c, frames, default):
        k = self._keys(c)
        if k is None:
            return np.full(len(frames), float(default))
        kf, kv, lin, last = k
        f = np.asarray(frames, np.float64)
        fi = np.floor(f).astype(np.int64) & 0xffff
        j = np.searchsorted(kf, fi, side='right') - 1
        j = np.where(j < 0, 0, j)
        j = np.where(fi >= last, len(kf) - 1, j)
        v = kv[j]
        if lin.any():
            nxt = np.minimum(j + 1, len(kf) - 1)
            ok = lin[j] & (nxt != j) & (fi < last)
            den = np.where(kf[nxt] != kf[j], kf[nxt] - kf[j], 1)
            v = np.where(ok, kv[j] + (f - kf[j]) / den * (kv[nxt] - kv[j]), v)
        return v

    def local_matrices(self, frames):
        """(len(frames), num_tracks, 4, 4) local matrices S * R * T of every track"""
        n = len(frames)
        out = np.zeros((n, self.num_tracks, 4, 4))
        k4 = self.key_size == 4
        for t in range(self.num_tracks):
            c = 9 * t
            T = np.stack([self.sample_curve(c + a, frames, 0) for a in range(3)], 1)
            R = np.stack([self.sample_curve(c + 3 + a, frames, 0) for a in range(3)], 1)
            S = np.stack([self.sample_curve(c + 6 + a, frames, 100 if k4 else 1) for a in range(3)], 1)
            if k4:
                T, R, S = T / self.t_scale, R * ROT4, S * 0.01
            out[:, t] = compose_srt_batch(T, R, S)
        return out


def compose_srt_batch(T, R, S):
    """formats.compose_srt for arrays: diag(S) @ X(rx) @ Y(ry) @ Z(rz), translation row T"""
    n = len(T)
    cx, sx, cy, sy, cz, sz = (np.cos(R[:, 0]), np.sin(R[:, 0]), np.cos(R[:, 1]), np.sin(R[:, 1]),
                              np.cos(R[:, 2]), np.sin(R[:, 2]))
    o, z = np.ones(n), np.zeros(n)
    X = np.stack([np.stack([o, z, z], 1), np.stack([z, cx, sx], 1), np.stack([z, -sx, cx], 1)], 1)
    Y = np.stack([np.stack([cy, z, -sy], 1), np.stack([z, o, z], 1), np.stack([sy, z, cy], 1)], 1)
    Z = np.stack([np.stack([cz, sz, z], 1), np.stack([-sz, cz, z], 1), np.stack([z, z, o], 1)], 1)
    M3 = X @ Y @ Z
    M3 = S[:, :, None] * M3
    M = np.zeros((n, 4, 4))
    M[:, :3, :3] = M3
    M[:, 3, :3] = T
    M[:, 3, 3] = 1.0
    return M


def motions_in(data, file_name=''):
    """[(name, Tmo)] of a motion catalogue (.cat) or a single .tmo2"""
    if len(data) >= 4 and struct.unpack_from('<I', data, 0)[0] == MAGIC:
        return [(file_name.rsplit('.', 1)[0], Tmo(data, file_name))]
    out = []
    for name, kind, o, s, _ in F.parse_cat(data):
        if kind != 'tmo':
            continue
        if s >= 0x60 and struct.unpack_from('<H', data, o + 10)[0] != 0:
            continue
        try:
            out.append((name, Tmo(data[o:o + s], name)))
        except (ValueError, struct.error):
            pass
    return out


COMPANIONS = ('_hr', '_hd', '_wp00', '_wp01', '_wp02')


def is_companion(name):
    return name.endswith(COMPANIONS)


class Rig:
    """the imported armature as the game sees it. bones: dicts in parent-first order with name, parent (index or -1),
    hash, model, rest (4x4 game world), offset (3 or None), adjust (4x4 or None)"""

    def __init__(self, bones, body_model=''):
        self.bones = bones
        self.body_model = body_model
        idx = {b['name']: i for i, b in enumerate(bones)}
        for b in bones:
            par = b.get('parent')
            b['pi'] = idx.get(par, -1) if isinstance(par, str) else (-1 if par is None else int(par))
            P = bones[b['pi']]['rest'] if b['pi'] >= 0 else np.eye(4)
            b['rest_local'] = b['rest'] @ np.linalg.inv(P)
            if b.get('offset') is None:
                b['offset'] = b['rest_local'][3, :3].copy()
                if b.get('adjust') is not None:
                    b['offset'] = b['offset'] - np.asarray(b['adjust'])[3, :3]

    def part(self, b):
        m = b['model']
        if '_weapon' in m:
            return 'weapon'
        if m.endswith('_hair'):
            return 'hair'
        if m.endswith('_face'):
            return 'face'
        return 'body'

    def drives(self, base, companions):
        """[(bone index, Tmo, track)]: the base motion on every non-weapon bone it keys; companions by suffix:
        _hr the hair, _hd the face, _wpKK the weapon model it covers at least half of (as the viewer does)"""
        out = {}
        for i, b in enumerate(self.bones):
            if self.part(b) != 'weapon' and b['hash'] in base.track:
                out[i] = (base, base.track[b['hash']])
        for suffix, mo in companions.items():
            if suffix in ('_hr', '_hd'):
                want = 'hair' if suffix == '_hr' else 'face'
                for i, b in enumerate(self.bones):
                    if self.part(b) == want and b['hash'] in mo.track:
                        out[i] = (mo, mo.track[b['hash']])
        weapons = {}
        for i, b in enumerate(self.bones):
            if self.part(b) == 'weapon':
                weapons.setdefault(b['model'], []).append(i)
        for model, ids in weapons.items():
            for suffix in ('_wp00', '_wp01', '_wp02'):
                mo = companions.get(suffix)
                if mo is None:
                    continue
                hit = sum(1 for i in ids if self.bones[i]['hash'] in mo.track)
                if 2 * hit >= len(ids):
                    for i in ids:
                        if self.bones[i]['hash'] in mo.track:
                            out[i] = (mo, mo.track[self.bones[i]['hash']])
                    break
        return out

    def evaluate(self, base, companions=None, frames=None, root_motion=True):
        """game-space world matrices (len(frames), num_bones, 4, 4) of the motion"""
        companions = companions or {}
        frames = np.arange(max(1, base.num_frames)) if frames is None else np.asarray(frames)
        nf, nb = len(frames), len(self.bones)
        drives = self.drives(base, companions)
        cache = {}
        for mo in {id(m): m for m, _ in drives.values()}.values():
            fr = frames
            if mo is not base and mo.num_frames > 0:
                fr = np.minimum(frames, mo.num_frames)
            cache[id(mo)] = mo.local_matrices(fr)
        W = np.zeros((nf, nb, 4, 4))
        for i, b in enumerate(self.bones):
            d = drives.get(i)
            if d is None:
                L = np.broadcast_to(b['rest_local'], (nf, 4, 4)).copy()
            else:
                L = cache[id(d[0])][:, d[1]].copy()
                if b.get('adjust') is not None:
                    L = np.asarray(b['adjust']) @ L
                L[:, 3, :3] += b['offset']
                if not root_motion and b['pi'] < 0:
                    L[:, 3, :3] = L[0, 3, :3]
            W[:, i] = L @ W[:, b['pi']] if b['pi'] >= 0 else L
        if not root_motion:
            root = next((i for i, b in enumerate(self.bones) if b['hash'] == F.tamsoft_hash('ido')), 0)
            shift = W[:, root, 3, :3] - W[:1, root, 3, :3]
            shift[:, 1] = 0.0
            W[:, :, 3, :3] -= shift[:, None, :]
        return W


A = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
AI = np.linalg.inv(A)


def to_blender_matrices(M):
    """game row-vector matrices (..., 4, 4) -> Blender column-vector matrices in Blender axes"""
    return A @ np.swapaxes(M, -1, -2) @ AI


def pose_basis(rig, W, matrix_local, parents):
    """Blender pose bone matrix_basis (nf, nb, 4, 4) that puts each bone where the game's world matrices W put it.
    matrix_local: (nb, 4, 4) Bone.matrix_local (armature space); parents: index per bone (-1 = none)"""
    nf, nb = W.shape[:2]
    rest = np.stack([b['rest'] for b in rig.bones])
    D = to_blender_matrices(np.linalg.inv(rest)[None] @ W)
    pose = D @ matrix_local[None]
    basis = np.zeros_like(pose)
    for i in range(nb):
        p = parents[i]
        if p < 0:
            basis[:, i] = np.linalg.inv(matrix_local[i]) @ pose[:, i]
        else:
            rel = np.linalg.inv(matrix_local[p]) @ matrix_local[i]
            basis[:, i] = np.linalg.inv(rel) @ np.linalg.inv(pose[:, p]) @ pose[:, i]
    return basis, pose


def decompose(basis):
    """(…, 4, 4) -> location (…, 3), quaternion w x y z (…, 4) with sign continuity along axis 0, scale (…, 3)"""
    loc = basis[..., :3, 3]
    M = basis[..., :3, :3]
    scale = np.linalg.norm(M, axis=-2)
    R = M / np.where(scale > 1e-12, scale, 1.0)[..., None, :]
    q = mat3_to_quat(R)
    for k in range(1, q.shape[0]):
        flip = np.sum(q[k] * q[k - 1], axis=-1) < 0
        q[k][flip] *= -1
    return loc, q, scale


def mat3_to_quat(R):
    """rotation matrices (..., 3, 3) -> unit quaternions w x y z (Shepperd: divide by the largest component)"""
    R = np.asarray(R, float)
    m = lambda a, b: R[..., a, b]
    t = np.stack([1 + m(0, 0) + m(1, 1) + m(2, 2), 1 + m(0, 0) - m(1, 1) - m(2, 2),
                  1 - m(0, 0) + m(1, 1) - m(2, 2), 1 - m(0, 0) - m(1, 1) + m(2, 2)], -1)
    k = np.argmax(t, -1)
    s = 0.5 / np.sqrt(np.maximum(np.take_along_axis(t, k[..., None], -1)[..., 0], 1e-12))
    big = 0.25 / s
    q = np.zeros(R.shape[:-2] + (4,))
    c = k == 0
    q[c] = np.stack([big[c], (m(2, 1) - m(1, 2))[c] * s[c], (m(0, 2) - m(2, 0))[c] * s[c], (m(1, 0) - m(0, 1))[c] * s[c]], -1)
    c = k == 1
    q[c] = np.stack([(m(2, 1) - m(1, 2))[c] * s[c], big[c], (m(0, 1) + m(1, 0))[c] * s[c], (m(0, 2) + m(2, 0))[c] * s[c]], -1)
    c = k == 2
    q[c] = np.stack([(m(0, 2) - m(2, 0))[c] * s[c], (m(0, 1) + m(1, 0))[c] * s[c], big[c], (m(1, 2) + m(2, 1))[c] * s[c]], -1)
    c = k == 3
    q[c] = np.stack([(m(1, 0) - m(0, 1))[c] * s[c], (m(0, 2) + m(2, 0))[c] * s[c], (m(1, 2) + m(2, 1))[c] * s[c], big[c]], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)



def pose_from_basis(basis, matrix_local, parents):
    """Blender pose matrices (armature space) from matrix_basis (nf, nb, 4, 4), constraints ignored"""
    pose = np.zeros_like(basis)
    for i, p in enumerate(parents):
        if p < 0:
            pose[:, i] = matrix_local[i] @ basis[:, i]
        else:
            rel = np.linalg.inv(matrix_local[p]) @ matrix_local[i]
            pose[:, i] = pose[:, p] @ rel @ basis[:, i]
    return pose


def world_from_pose(rig, pose, matrix_local):
    """game world matrices W (nf, nb, 4, 4) of Blender pose matrices; inverse of pose_basis()"""
    D = pose @ np.linalg.inv(matrix_local)[None]
    M = np.swapaxes(AI @ D @ A, -1, -2)
    rest = np.stack([b['rest'] for b in rig.bones])
    return rest[None] @ M


def local_tracks(rig, W):
    """per bone the track matrix L (nf, nb, 4, 4) that makes the game compute W (adjust and offset removed), and
    whether the bone differs from its rest pose at some frame (bones without a track keep their rest)"""
    nf, nb = W.shape[:2]
    L = np.zeros_like(W)
    moved = np.zeros(nb, bool)
    for i, b in enumerate(rig.bones):
        Lp = W[:, i] @ np.linalg.inv(W[:, b['pi']]) if b['pi'] >= 0 else W[:, i].copy()
        moved[i] = np.abs(Lp - b['rest_local'][None]).max() > 1e-6
        Lp[:, 3, :3] -= b['offset']
        if b.get('adjust') is not None:
            Lp = np.linalg.inv(np.asarray(b['adjust']))[None] @ Lp
        L[:, i] = Lp
    return L, moved


def srt_from_local(L):
    """L = diag(S) X(rx) Y(ry) Z(rz) with translation row T (compose_srt_batch) -> T, R (unwrapped along axis 0), S"""
    T = L[..., 3, :3].copy()
    M = L[..., :3, :3]
    S = np.linalg.norm(M, axis=-1)
    R3 = M / np.where(S > 1e-12, S, 1.0)[..., None]
    sy = np.clip(-R3[..., 0, 2], -1.0, 1.0)
    ry = np.arcsin(sy)
    gimbal = np.abs(sy) > 0.999999
    rz = np.where(gimbal, 0.0, np.arctan2(R3[..., 0, 1], R3[..., 0, 0]))
    rx = np.where(gimbal, np.arctan2(-R3[..., 2, 1], R3[..., 1, 1]), np.arctan2(R3[..., 1, 2], R3[..., 2, 2]))
    R = np.stack([rx, ry, rz], -1)
    if R.ndim >= 2 and R.shape[0] > 1:
        R = np.unwrap(R, axis=0)
    return T, R, S


def encode_tmo(tracks, num_frames):
    """tracks: [(hash, [9 curves])], a curve is None (default value) or the values at frames 0..num_frames.
    Float keys (8 bytes {u16 frame, u16 0x103, f32}), a key only where the value changes (they hold)."""
    keys = bytearray()
    curves = bytearray()
    nkeys = 0
    for _, chans in tracks:
        for vals in chans:
            if vals is None:
                curves += struct.pack('<QHHHH', nkeys, 0, 0, 0, 0)
                continue
            first, n, prev = nkeys, 0, None
            last = len(vals) - 1
            for f, v in enumerate(vals):
                v = float(np.float32(v))
                if n and v == prev and f != last:
                    continue
                keys += struct.pack('<HHf', f, 0x103, v)
                prev = v
                n += 1
                nkeys += 1
            curves += struct.pack('<QHHHH', first, n, 0, last, 0)
    hashes = b''.join(struct.pack('<I', h) for h, _ in tracks)
    o_hash = 0x70
    o_mot = (o_hash + len(hashes) + 15) & ~15
    o_cur = o_mot + 16
    o_key = o_cur + len(curves)
    out = bytearray(struct.pack('<IIHHf', MAGIC, 0x10, 0x31, 0, 1000.0))
    body = bytearray(0x60)
    struct.pack_into('<I', body, 0x00, 1)
    struct.pack_into('<Q', body, 0x08, o_mot)
    struct.pack_into('<Q', body, 0x28, o_hash)
    struct.pack_into('<I', body, 0x30, 9 * len(tracks))
    struct.pack_into('<Q', body, 0x38, o_cur)
    struct.pack_into('<I', body, 0x40, nkeys)
    struct.pack_into('<Q', body, 0x48, o_key)
    out += body + hashes
    out += b'\0' * (o_mot - len(out))
    out += struct.pack('<hhIII', num_frames, len(tracks), 0, 0, 0)
    out += curves + keys
    return bytes(out)


def tracks_of(rig, L, bone_ids, eps=1e-6):
    """[(hash, 9 curves)] for the given bones; constant curves at the default value are left out"""
    out = []
    for i in bone_ids:
        T, R, S = srt_from_local(L[:, i])
        chans = []
        for arr, default in ((T, 0.0), (R, 0.0), (S, 1.0)):
            for c in range(3):
                v = arr[:, c]
                chans.append(None if np.abs(v - default).max() < eps else [float(x) for x in v])
        out.append((rig.bones[i]['hash'], chans))
    return out
