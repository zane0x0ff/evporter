import os
import struct
import numpy as np

_CRC = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0xEDB88320 if _c & 1 else _c >> 1
    _CRC.append(_c)


def tamsoft_hash(name):
    """Bone / texture name hash: reflected CRC-32 table, initial value = length, no final XOR (sub_1402512a0)."""
    if isinstance(name, str):
        name = name.encode('latin-1', 'replace')
    h = len(name) & 0xFFFFFFFF
    for b in name:
        h = _CRC[(h ^ b) & 0xFF] ^ (h >> 8)
    return h


_BONE_NAMES = None


def bone_name(h, extra=None):
    """Known bone name for a hash (bone_names.txt, recovered by hashing candidates), else Bone_xxxxxxxx."""
    global _BONE_NAMES
    if _BONE_NAMES is None:
        _BONE_NAMES = {}
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bone_names.txt')
        if os.path.exists(p):
            for line in open(p, encoding='ascii', errors='ignore'):
                n = line.split('#')[0].strip()
                if n:
                    _BONE_NAMES[tamsoft_hash(n)] = n
    if extra and h in extra:
        return extra[h]
    return _BONE_NAMES.get(h, 'Bone_%08x' % h)


class GameData:
    """Case-insensitive file access below the GameData folder (paths as the game writes them, '/' or '\\')."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        if not os.path.exists(self.find('Fnames/E/filename.bin') or ''):
            sub = self._child(self.root, 'gamedata', True)
            if sub:
                self.root = sub
        self._dirs = {}
        self._names = None
        self._cache = {}

    def _child(self, d, name, want_dir):
        try:
            for e in os.listdir(d):
                if e.lower() == name.lower():
                    p = os.path.join(d, e)
                    if os.path.isdir(p) == want_dir:
                        return p
        except OSError:
            pass
        return None

    def find(self, rel):
        parts = [p for p in rel.replace('\\', '/').split('/') if p]
        cur = self.root
        for i, p in enumerate(parts):
            nxt = os.path.join(cur, p)
            if os.path.exists(nxt):
                cur = nxt
                continue
            nxt = self._child(cur, p, i < len(parts) - 1)
            if not nxt:
                return None
            cur = nxt
        return cur if os.path.isfile(cur) else None

    def read(self, rel):
        if not rel:
            return None
        if rel in self._cache:
            return self._cache[rel]
        p = self.find(rel)
        data = open(p, 'rb').read() if p else None
        self._cache[rel] = data
        return data

    @property
    def ok(self):
        return self.find('Fnames/E/filename.bin') is not None

    def asset_name(self, aid):
        """Fnames/E/filename.bin: {u32 1; u32 n; {u32 nameOfs; u32 packed}[n]} + names; entry i = asset id i."""
        if self._names is None:
            self._names = self.read('Fnames/E/filename.bin') or b''
        fn = self._names
        if len(fn) < 8:
            return ''
        n = struct.unpack_from('<I', fn, 4)[0]
        if aid >= n:
            return ''
        o = struct.unpack_from('<I', fn, 8 + 8 * aid)[0]
        e = fn.find(b'\0', o)
        return fn[o:e].decode('latin-1').replace('\\', '/') if o < len(fn) else ''

    def asset_path(self, aid, fallback):
        return self.asset_name(aid) or fallback


def parse_cat(d):
    """Catalogue tree -> [(name, kind, offset, size, texgroup)]; kind 'tmd' | 'tmo' | 'dds' | 'blob'.
    texgroup numbers the texture packs in file order (0 = the model's own set, 1+ shared sets)."""
    N = len(d)
    u32 = lambda o: struct.unpack_from('<I', d, o)[0] if o + 4 <= N else 0

    def is_node(o):
        k, n = u32(o), u32(o + 4)
        if k == 1 and n == 1:
            return u32(o + 12) == 0x100
        return k == 0 and 0 < n < 0x10000 and u32(o + 12) == 0x100

    tag = lambda o, s: d[o:o + len(s)] == s
    kind = lambda o: 'tmd' if tag(o, b'tmd') else 'tmo' if tag(o, b'tmo') else 'dds' if tag(o, b'DDS ') else 'blob'

    def is_pack(o):
        hs, n = u32(o), u32(o + 4)
        return 0 < n < 0x1000 and 12 + 4 * n <= hs <= 0x1000 and o + hs + 4 <= N and tag(o + hs, b'DDS ')

    def names(o, size):
        out, cur = [], b''
        for c in d[o:min(o + size, N)]:
            if c == 0:
                break
            if c == 13:
                continue
            if c >= 0x80:
                cur += bytes([c]); continue
            if c in (10, 44):
                if cur:
                    out.append(cur.decode('latin-1'))
                cur = b''; continue
            if c < 0x20 or c > 0x7e:
                return []
            cur += bytes([c])
        if cur:
            out.append(cur.decode('latin-1'))
        return out

    out, group = [], [0]

    def walk(o, end, depth):
        if depth > 16 or o + 8 > N:
            return
        k, n = u32(o), u32(o + 4)
        if k == 1:
            child, sz = o + u32(o + 12), u32(o + 16)
            walk(child, child + sz if sz and child + sz <= end else end, depth + 1)
            return
        if not 0 < n < 0x10000:
            return
        offs = [o + u32(o + 20 + 4 * i) for i in range(n)]
        sizes = [u32(o + 20 + 4 * (n + i)) for i in range(n)]
        for i in range(n):
            lim = end
            for j in range(n):
                if offs[j] > offs[i]:
                    lim = min(lim, offs[j])
            if sizes[i] == 0 or offs[i] + sizes[i] > end:
                sizes[i] = max(0, lim - offs[i])
        nm, first = [], 0
        if not is_node(offs[0]) and kind(offs[0]) == 'blob' and not is_pack(offs[0]) and not tag(offs[0], b'fx00'):
            nm, first = names(offs[0], sizes[0]), 1
        if n - first == 1 and is_pack(offs[first]):
            p = offs[first]
            hs, cnt, tot = u32(p), u32(p + 4), u32(p + 8)
            g = group[0]; group[0] += 1
            for i in range(cnt):
                a = u32(p + 12 + 4 * i)
                b = u32(p + 16 + 4 * i) if i + 1 < cnt else tot
                off = p + hs + a
                size = max(0, min(b - a, N - off))
                out.append((nm[i] if i < len(nm) else 'tex%02d' % i, 'dds', off, size, g))
            return
        for i in range(first, n):
            if is_node(offs[i]):
                walk(offs[i], offs[i] + sizes[i], depth + 1)
                continue
            kd = kind(offs[i])
            out.append((nm[i - first] if i - first < len(nm) else '%s%d' % (kd, i), kd, offs[i], sizes[i], -1))

    walk(0, N, 0)
    return [e for e in out if e[2] + e[3] <= N]


VF_POS, VF_NORMAL, VF_TANGENTS, VF_UV0, VF_UV1, VF_UV2, VF_COLOR = 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
VF_U100, VF_U200, VF_SKIN, VF_INDEX32, VF_SKELETON = 0x100, 0x200, 0x400, 0x800, 0x2000


class Tmd2:
    """A .tmd2 model (include/EVPorter/Tmd2Format.h): decoded vertices (numpy), index groups, meshes with their draw
    commands, materials with texture slots and shader channels, skeleton, per-mesh layer tags, bounding boxes."""

    def __init__(self, data, name=''):
        self.name = name
        d = self.d = bytes(data)
        q = lambda o: struct.unpack_from('<Q', d, o)[0]
        u = lambda o: struct.unpack_from('<I', d, o)[0]
        f = self.flags = struct.unpack_from('<H', d, 6)[0]
        self.skinned = struct.unpack_from('<H', d, 8)[0]
        self.version = struct.unpack_from('<H', d, 0xA)[0]
        self.anim_frames = struct.unpack_from('<H', d, 0xE)[0]
        self.bounds = struct.unpack_from('<6f', d, 0x10)
        P = self.ptr = dict(meshes=q(0x28), static=q(0x30), anim=q(0x38), commands=q(0x40), materials=q(0x48),
                            params=q(0x50), user=q(0x58), keys=q(0x60), groups=q(0x68), indices=q(0x70),
                            textable=q(0x78), slots=q(0x88), vertices=q(0x90))
        n_mesh, n_mat, user_size, n_grp = u(0xA8), u(0xB8), u(0xC0), u(0xC8)
        n_textable, n_vert = u(0xD0), u(0xDC)

        fields = [('pos', '<f4', (3,))]
        if f & VF_SKIN:
            fields += [('w', 'u1', (4,)), ('i', 'u1', (4,))]
        if f & VF_NORMAL:
            fields += [('n', 'u1', (4,))]
        if f & VF_TANGENTS:
            fields += [('t', 'u1', (4,)), ('b', 'u1', (4,))]
        if f & VF_COLOR:
            fields += [('c', 'u1', (4,))]
        if f & VF_U100:
            fields += [('x100', 'u1', (4,))]
        if f & VF_U200:
            fields += [('x200', 'u1', (4,))]
        for bit, key in ((VF_UV0, 'uv0'), (VF_UV1, 'uv1'), (VF_UV2, 'uv2')):
            if f & bit:
                fields += [(key, '<i2', (2,))]
        v = np.frombuffer(d, dtype=np.dtype(fields), count=n_vert, offset=P['vertices'])
        self.num_vertices = n_vert
        self.pos = v['pos'].astype(np.float64)
        sn = lambda a: a[:, :3].astype(np.float64) / 255.0 * 2.0 - 1.0
        self.weights = v['w'].astype(np.float64) / 255.0 if f & VF_SKIN else None
        self.bone_idx = v['i'].astype(np.int64) if f & VF_SKIN else None
        self.normal = sn(v['n']) if f & VF_NORMAL else None
        self.tangent = sn(v['t']) if f & VF_TANGENTS else None
        self.binormal = sn(v['b']) if f & VF_TANGENTS else None
        self.color = v['c'].astype(np.float64) / 255.0 if f & VF_COLOR else None
        self.uv = []
        for key, bit in (('uv0', VF_UV0), ('uv1', VF_UV1), ('uv2', VF_UV2)):
            if f & bit:
                a = v[key].astype(np.float64)
                self.uv.append(np.stack([a[:, 0] / 1024.0, (a[:, 1] + 1024.0) / 1024.0], 1))
            else:
                self.uv.append(None)

        idt = '<u4' if f & VF_INDEX32 else '<u2'
        isz = 4 if f & VF_INDEX32 else 2
        self.groups = []
        for g in range(n_grp):
            tris, _, first = struct.unpack_from('<IIQ', d, P['groups'] + 16 * g)
            a = np.frombuffer(d, dtype=idt, count=tris * 3, offset=P['indices'] + first * 3 * isz).astype(np.int64)
            a[a >= n_vert] = 0
            self.groups.append(a.reshape(-1, 3))

        self.meshes = []
        for m in range(n_mesh):
            o = P['meshes'] + 0x38 * m
            bounds = struct.unpack_from('<6f', d, o)
            ncmd, mflags, xtype = struct.unpack_from('<HBB', d, o + 0x18)
            first = q(o + 0x20)
            xidx = u(o + 0x30)
            cmds = []
            for k in range(ncmd):
                c = struct.unpack_from('<H', d, P['commands'] + 2 * (first + k))[0]
                cmds.append((c >> 12, c & 0xFFF))
            self.meshes.append(dict(bounds=bounds, flags=mflags, xform_type=xtype, xform_index=xidx, commands=cmds))

        self.materials = []
        for m in range(n_mat):
            o = P['materials'] + 20 * m
            unk0, fourcc, ft, nt, fp, npar, unk10 = struct.unpack_from('<I4sHHHHI', d, o)
            slots = []
            for s in range(nt):
                so = P['slots'] + 12 * (ft + s)
                h, tex, b6, kind, b8 = struct.unpack_from('<IHBBI', d, so)
                slots.append(dict(hash=h, texture=tex, kind=kind))
            self.materials.append(dict(fourcc=fourcc.decode('latin-1'), slots=slots, first_param=fp, num_params=npar))

        self.textable = []
        for i in range(n_textable):
            h, a, b = struct.unpack_from('<I4s4s', d, P['textable'] + 12 * i)
            self.textable.append(dict(hash=h, index=a[0], tag=b[2:4].decode('latin-1', 'replace'), raw=(a + b).hex()))

        self.bones, self.inv_bind, self.skin_batches = [], [], []
        if f & VF_SKELETON:
            ob, op, oi, obn = q(0xE0), q(0xE8), q(0xF0), q(0xF8)
            nb, ninv, nbone = u(0x100), u(0x108), u(0x10C)
            for b in range(nbone):
                h, x, y, z, parent = struct.unpack_from('<I3fi', d, obn + 24 * b)
                self.bones.append(dict(hash=h, offset=(x, y, z), parent=parent))
            if ninv:
                self.inv_bind = np.frombuffer(d, dtype='<f4', count=16 * ninv, offset=oi).astype(np.float64).reshape(ninv, 4, 4)
            for s in range(nb):
                cnt, _, off = struct.unpack_from('<iIQ', d, ob + 16 * s)
                self.skin_batches.append(list(d[op + off:op + off + max(0, cnt)]))
        self.bone_index = {}
        for i, b in enumerate(self.bones):
            self.bone_index.setdefault(b['hash'], i)

        self.tags = []
        if P['user'] and user_size and P['user'] + user_size <= len(d):
            pool = d[P['user']:P['user'] + user_size]
            tags = []
            for m in range(n_mesh):
                o = u(P['meshes'] + 0x38 * m + 0x28)
                tags.append(pool[o:pool.find(b'\0', o) if pool.find(b'\0', o) >= 0 else len(pool)].decode('latin-1')
                            if o < len(pool) else '')
            self.tags = tags
        self.textures = []

    def find_bone(self, h):
        return self.bone_index.get(h, -1)

    def channel(self, off, frame=0):
        d = self.d
        n, _, first = struct.unpack_from('<HHI', d, off)
        if n == 0:
            return struct.unpack_from('<f', d, off + 4)[0]
        k = self.ptr['keys']
        if self.version != 0x201:
            return struct.unpack_from('<f', d, k + 4 * (first + (frame % n if n > 1 else 0)))[0]
        if n == 1:
            return struct.unpack_from('<f', d, k + 8 * first + 4)[0]
        key = lambda j: struct.unpack_from('<H2xf', d, k + 8 * (first + j))
        last = key(n - 1)[0]
        t = frame % last if last else 0
        for j in range(n):
            fj, vj = key(j)
            if t < fj:
                if j == 0:
                    return key(0)[1]
                f0, v0 = key(j - 1)
                return v0 + (t - f0) / float(fj - f0) * (vj - v0)
        return 0.0

    def channel_keys(self, off):
        """(frames, values) of an animated channel (more than one key), else None. Keyframed models (version
        0x201): {u16 frame, u16 flags (interp & 0xF: 2 / 3 linear), f32}; others: one baked value per frame"""
        d = self.d
        n, _, first = struct.unpack_from('<HHI', d, off)
        if n <= 1:
            return None
        k = self.ptr['keys']
        if self.version != 0x201:
            return np.arange(n, dtype=float), np.array(struct.unpack_from('<%df' % n, d, k + 4 * first), float)
        keys = [struct.unpack_from('<H2xf', d, k + 8 * (first + j)) for j in range(n)]
        return np.array([x[0] for x in keys], float), np.array([x[1] for x in keys], float)

    def material_keys(self, mi):
        """{channel: (frames, values)} of the material's animated channels"""
        m = self.materials[mi]
        out = {}
        for k in range(min(32, m['num_params'])):
            kv = self.channel_keys(self.ptr['params'] + 8 * (m['first_param'] + k))
            if kv is not None:
                out[k] = kv
        return out

    def material_params(self, mi, frame=0):
        m = self.materials[mi]
        return [self.channel(self.ptr['params'] + 8 * (m['first_param'] + k), frame) for k in range(min(32, m['num_params']))]

    def bind_world(self, adjust=None):
        """rest pose world matrices: W = L * W[parent], L = identity (+ per-bone body adjustment) + bone offset"""
        W = []
        for i, b in enumerate(self.bones):
            L = adjust[i].copy() if adjust is not None and adjust[i] is not None else np.eye(4)
            L[3, :3] += b['offset']
            W.append(L @ W[b['parent']] if b['parent'] >= 0 else L)
        return W

    def mesh_matrix(self, mi):
        m = self.meshes[mi]
        if m['xform_type'] == 1:
            v = struct.unpack_from('<9f', self.d, self.ptr['static'] + 36 * m['xform_index'])
            return compose_srt(v[0:3], v[3:6], v[6:9])
        if m['xform_type'] == 2:
            base = self.ptr['anim'] + 80 * m['xform_index']
            c = [self.channel(base + 8 * i) for i in range(9)]
            return compose_srt(c[0:3], c[3:6], c[6:9])
        return None


def euler_xyz(x, y, z):
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    X = np.array([[1, 0, 0, 0], [0, cx, sx, 0], [0, -sx, cx, 0], [0, 0, 0, 1]])
    Y = np.array([[cy, 0, -sy, 0], [0, 1, 0, 0], [sy, 0, cy, 0], [0, 0, 0, 1]])
    Z = np.array([[cz, sz, 0, 0], [-sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return X @ Y @ Z


def compose_srt(t, r, s):
    M = np.diag([s[0], s[1], s[2], 1.0]) @ euler_xyz(*r)
    M[3, :3] = t
    return M


def model_from_cat(data, want=''):
    """(model, entries) for the model 'want' (else the first) of a catalogue; textures attached as raw DDS bytes"""
    entries = parse_cat(data)
    tmds = [e for e in entries if e[1] == 'tmd']
    e = next((x for x in tmds if x[0] == want), None) if want else (tmds[0] if tmds else None)
    if not e:
        raise ValueError('catalogue has no model ' + want if want else 'catalogue holds no model')
    m = Tmd2(data[e[2]:e[2] + e[3]], e[0])
    m.textures = cat_textures(data, entries)
    return m


def cat_textures(data, entries=None, force_group=-1):
    entries = entries if entries is not None else parse_cat(data)
    return [dict(name=n, hash=tamsoft_hash(n), group=force_group if force_group >= 0 else g, dds=data[o:o + s],
                 size=dds_size(data[o:o + s]))
            for n, k, o, s, g in entries if k == 'dds' and s >= 128]


def dds_size(dds):
    h, w = struct.unpack_from('<II', dds, 12)
    return w * h


def resolve_textures(order, sources):
    """per name hash, a shared-group texture beats a model's own, then the larger image (viewer TexturePool)"""
    best = {}
    for lst in sources:
        for t in lst or []:
            b = best.get(t['hash'])
            if b is None or (t['group'] >= 1, t['size']) > (b['group'] >= 1, b['size']):
                best[t['hash']] = t
    return [best.get(t['hash'], t) for t in order]


class CosInfo:
    """Binary/Param/Player/cosinfo.bin: {u32 1, u32 0, u32 nChara, u32 nCostume}, nChara x 28 u16, nCostume x
    {u32 braType, u16 flagA (no damage BRK0-2), u16 flagB (underwear hidden)}."""

    def __init__(self, data):
        _, _, self.nchara, self.ncostume = struct.unpack_from('<4I', data, 0)
        self.chars = []
        for c in range(self.nchara):
            r = [None if x == 0xFFFF else x for x in struct.unpack_from('<28H', data, 16 + 56 * c)]
            self.chars.append(dict(costume=r[0:2], underwear=r[2:4], acc=r[4:14], acc_aux=r[14:24], hair=r[24:27]))
        o = 16 + 56 * self.nchara
        self.costumes = [dict(zip(('bra', 'no_damage', 'no_underwear'), struct.unpack_from('<IHH', data, o + 8 * k)))
                         for k in range(self.ncostume)]

    def costume(self, cid):
        c = cid % 174
        return self.costumes[c] if c < len(self.costumes) else dict(bra=0, no_damage=0, no_underwear=0)


def body_scale(data, chara):
    """plbodyscl.bin: {u32 n; u32 nameOfs[n]} then per character n x {f32 translate[3], scale[3]} -> {hash: 4x4}"""
    if not data:
        return {}
    n = struct.unpack_from('<I', data, 0)[0]
    base = 4 + 4 * n
    first = struct.unpack_from('<I', data, 4)[0]
    if chara >= (first - base) // (24 * n):
        return {}
    out = {}
    for b in range(n):
        no = struct.unpack_from('<I', data, 4 + 4 * b)[0]
        name = data[no:data.index(b'\0', no)].decode('latin-1')
        v = struct.unpack_from('<6f', data, base + (chara * n + b) * 24)
        if v == (0, 0, 0, 1, 1, 1):
            continue
        S = np.diag([v[3], v[4], v[5], 1.0])
        S[3, :3] = v[0:3]
        out[tamsoft_hash(name)] = S
    return out


def acc_offset(data, record):
    """accoffset.bin record (48 B): f32 offset[3], f32 rotDeg[3], u16, i16, u16, u16, char bone[16]"""
    if not data or 4 + 48 * (record + 1) > len(data):
        return (0, 0, 0), (0, 0, 0), 'head'
    b = 4 + 48 * record
    off = struct.unpack_from('<3f', data, b)
    rot = struct.unpack_from('<3f', data, b + 12)
    bone = data[b + 32:b + 48].split(b'\0')[0].decode('latin-1').strip()
    return off, rot, bone or 'head'


def accessory_ref(aid):
    """accessory id -> (accoffset record, asset id, file stem) (sub_14006a590)"""
    if aid is None or aid < 0 or aid >= 1500:
        return None
    kinds = [('acce_head', 40), ('acce_face', 30), ('acce_body', 20), ('acce_arm', 20), ('acce_leg', 20),
             ('richacce_head', 20), ('richacce_body', 20), ('extacce', 30)]
    if aid < 1000:
        record, variant = aid % 200, aid // 200
        asset = 0xC51 + variant * 200 + record
    else:
        record, variant = 200 + (aid - 1000) % 100, (aid - 1000) // 100
        asset = 0x1E79 + variant * 100 + record - 200
    if record >= 200:
        return record, asset, 'canbadge%02d_%02d' % (record - 200, variant)
    n = record
    for k, c in kinds:
        if n < c:
            return record, asset, '%s%02d_%02d' % (k, n, variant)
        n -= c
    return None


def weapon_slots(data, chara):
    """weaponinfo.bin: 35 x 8 char[16] bones, A = i32[35][8] (0 = hidden), B = i16[35][8][8] hidden meshes"""
    if not data or len(data) < 10080:
        return []
    names, lists = 35 * 8 * 16, 35 * 8 * 16 + 35 * 8 * 4
    out = []
    for k in range(8):
        o = (chara * 8 + k) * 16
        bone = data[o:o + 16].split(b'\0')[0].decode('latin-1')
        if not bone or not struct.unpack_from('<i', data, names + (chara * 8 + k) * 4)[0]:
            continue
        hidden = [x for x in struct.unpack_from('<8h', data, lists + (chara * 8 + k) * 16) if x >= 0]
        out.append((k, bone, hidden))
    return out


def _rgb565(c):
    c = c.astype(np.uint32)
    return np.stack([((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31], -1)


def decode_dds(dds, start=None):
    """DXT1 / DXT3 / DXT5 / 24-32 bit DDS -> (width, height, uint8 RGBA array (h, w, 4), top row first).
    First mip level only (of the image at byte offset start: a cube face, see decode_dds_cube), vectorised over the
    4x4 blocks."""
    if len(dds) < 128 or dds[:4] != b'DDS ':
        raise ValueError('not a DDS file')
    h, w = struct.unpack_from('<II', dds, 12)
    pf, four, bits = struct.unpack_from('<I4sI', dds, 80)
    o = 128 + (20 if pf & 4 and four == b'DX10' else 0)
    if start is not None:
        o = start
    if pf & 4:
        mode = {b'DXT1': 1, b'DXT3': 3, b'DXT5': 5}.get(four)
        if not mode:
            raise ValueError('unsupported DDS format %r' % four)
        bs = 8 if mode == 1 else 16
        bw, bh = (w + 3) // 4, (h + 3) // 4
        blk = np.frombuffer(dds, np.uint8, bw * bh * bs, o).reshape(-1, bs)
        cb = blk if mode == 1 else blk[:, 8:]
        c0 = cb[:, 0].astype(np.uint16) | (cb[:, 1].astype(np.uint16) << 8)
        c1 = cb[:, 2].astype(np.uint16) | (cb[:, 3].astype(np.uint16) << 8)
        p0, p1 = _rgb565(c0), _rgb565(c1)
        four_col = (c0 > c1) | (mode != 1)
        pal = np.zeros((len(blk), 4, 4), np.uint32)
        pal[:, 0, :3], pal[:, 1, :3] = p0, p1
        pal[:, 2, :3] = np.where(four_col[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
        pal[:, 3, :3] = np.where(four_col[:, None], (p0 + 2 * p1) // 3, 0)
        pal[:, :, 3] = 255
        pal[:, 3, 3] = np.where(four_col, 255, 0)
        idx = cb[:, 4:8].astype(np.uint32)
        bitsv = idx[:, 0] | (idx[:, 1] << 8) | (idx[:, 2] << 16) | (idx[:, 3] << 24)
        sel = (bitsv[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
        px = np.take_along_axis(pal, sel[:, :, None].astype(np.int64).repeat(4, 2), 1)
        if mode == 3:
            a = blk[:, :8]
            nib = np.stack([a & 15, a >> 4], -1).reshape(-1, 16)
            px[:, :, 3] = nib.astype(np.uint32) * 17
        elif mode == 5:
            a0, a1 = blk[:, 0].astype(np.int32), blk[:, 1].astype(np.int32)
            ap = np.zeros((len(blk), 8), np.int32)
            ap[:, 0], ap[:, 1] = a0, a1
            big = a0 > a1
            for i in range(1, 7):
                ap[:, i + 1] = np.where(big, ((7 - i) * a0 + i * a1) // 7, 0)
            for i in range(1, 5):
                ap[:, i + 1] = np.where(big, ap[:, i + 1], ((5 - i) * a0 + i * a1) // 5)
            ap[:, 6] = np.where(big, ap[:, 6], 0)
            ap[:, 7] = np.where(big, ap[:, 7], 255)
            ab = blk[:, 2:8].astype(np.uint64)
            v = np.zeros(len(blk), np.uint64)
            for i in range(6):
                v |= ab[:, i] << np.uint64(8 * i)
            asel = ((v[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
            px[:, :, 3] = np.take_along_axis(ap, asel, 1)
        img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)[:h, :w]
        return w, h, img.astype(np.uint8)
    if bits not in (24, 32):
        raise ValueError('unsupported uncompressed DDS (%d bpp)' % bits)
    bpp = bits // 8
    raw = np.frombuffer(dds, np.uint8, w * h * bpp, o).reshape(h * w, bpp)
    val = np.zeros(h * w, np.uint32)
    for i in range(bpp):
        val |= raw[:, i].astype(np.uint32) << (8 * i)
    masks = struct.unpack_from('<4I', dds, 92)
    out = np.full((h * w, 4), 255, np.uint8)
    for k, mask in enumerate(masks):
        if not mask or (k == 3 and not pf & 1):
            continue
        sh = (mask & -mask).bit_length() - 1
        m = mask >> sh
        out[:, k] = (((val >> sh) & m) * 255 // m).astype(np.uint8)
    return w, h, out.reshape(h, w, 4)


def dds_is_cube(dds):
    return dds is not None and len(dds) >= 128 and dds[:4] == b'DDS ' and bool(struct.unpack_from('<I', dds, 112)[0] & 0x200)


def decode_dds_cube(dds):
    """a DDS cube map (caps2 & 0x200, bg17_envcube_* & co) -> 6 faces (+X, -X, +Y, -Y, +Z, -Z), each (h, w, 4) uint8,
    top row first; each face is followed by its mip chain (viewer/Assets.cpp DecodeDDS)"""
    h, w = struct.unpack_from('<II', dds, 12)
    pf, four, bits = struct.unpack_from('<I4sI', dds, 80)
    flags, mips = struct.unpack_from('<I', dds, 8)[0], struct.unpack_from('<I', dds, 28)[0]
    mips = mips if flags & 0x20000 and mips > 0 else 1
    o = 128 + (20 if pf & 4 and four == b'DX10' else 0)
    chain = 0
    for m in range(mips):
        mw, mh = max(1, w >> m), max(1, h >> m)
        if pf & 4:
            chain += ((mw + 3) // 4) * ((mh + 3) // 4) * (8 if four == b'DXT1' else 16)
        else:
            chain += mw * mh * (bits // 8)
    return [decode_dds(dds, o + chain * f)[2] for f in range(6)]


def cube_lookup(faces, d):
    """nearest texel of the cube faces for game-space directions d (n, 3), the GL / D3D cube convention
    (s, t from the major axis; t = 0 is the face's first row) -> (n, 4) uint8"""
    d = np.asarray(d, np.float64)
    ax = np.abs(d)
    major = np.argmax(ax, 1)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    face = np.where(major == 0, np.where(x > 0, 0, 1), np.where(major == 1, np.where(y > 0, 2, 3), np.where(z > 0, 4, 5)))
    sc = np.choose(face, [-z, z, x, x, x, -x])
    tc = np.choose(face, [-y, -y, z, -z, -y, -y])
    ma = np.maximum(ax[np.arange(len(d)), major], 1e-12)
    s, t = (sc / ma + 1) * 0.5, (tc / ma + 1) * 0.5
    h, w = faces[0].shape[:2]
    i = np.clip((s * w).astype(np.int64), 0, w - 1)
    j = np.clip((t * h).astype(np.int64), 0, h - 1)
    out = np.empty((len(d), 4), np.uint8)
    for f in range(6):
        k = face == f
        out[k] = faces[f][j[k], i[k]]
    return out


def cube_to_equirect(faces, width=None):
    """the cube map as a latitude / longitude image (h = w / 2, top row first) in game space: column u -> longitude
    atan2(x, z) = (u - 0.5) * 2 pi, row -> latitude asin(y) from +pi/2 (top) to -pi/2; the Blender material reads it
    with the same mapping (stage.py)"""
    w = width or max(64, min(2048, 4 * faces[0].shape[1]))
    h = w // 2
    lon = ((np.arange(w) + 0.5) / w - 0.5) * 2 * np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    d = np.stack([np.sin(LON) * np.cos(LAT), np.sin(LAT), np.cos(LON) * np.cos(LAT)], -1).reshape(-1, 3)
    return cube_lookup(faces, d).reshape(h, w, 4)


TOUCH_KINDS = {'touch_kabe': 'wall', 'touch_yuka': 'floor', 'touch_switch': 'switch wall', 'touch_water': 'water'}


def touch_kind(name):
    for k, v in TOUCH_KINDS.items():
        if name.endswith(k):
            return v
    return None


def parse_touch(b):
    """stage collision mesh (*_touch_kabe / _yuka / _switch / _water, docs 3.11; python/evporter.py Touch):
    returns (vertices (n, 3) game space metres, [(part id, [polygon index tuples])])"""
    flags, nid, nv = struct.unpack_from('<3I', b, 0)
    w, wf = (2, 'H') if flags & 1 else (4, 'I')
    o = 12 + 8 * nid
    ng = struct.unpack_from('<H', b, o)[0]
    group_end = struct.unpack_from('<%dH' % ng, b, o + 2)
    nparts = group_end[-1] if ng else 0
    o = (o + 2 + 2 * ng + 3) & ~3
    table = [struct.unpack_from('<2I', b, o + 8 * i) for i in range(nparts)]
    base = o + 8 * nparts
    dt = np.dtype('<u2' if w == 2 else '<u4')
    parts, start = [], 0
    for idx, end in table:
        p = base + start
        n = struct.unpack_from('<' + wf, b, p)[0]
        p += w
        face_end = np.frombuffer(b, dt, n, p); p += w * n
        npoly = int(face_end[-1]) if n else 0
        poly_end = np.frombuffer(b, dt, npoly, p).astype(np.int64); p += w * npoly
        nidx = int(poly_end[-1]) if npoly else 0
        index = np.frombuffer(b, dt, nidx, p).astype(np.int64)
        starts = np.r_[0, poly_end[:-1]] if npoly else np.zeros(0, np.int64)
        parts.append((idx, [tuple(index[s:e]) for s, e in zip(starts, poly_end) if e - s >= 3]))
        start = end
    o = (base + start + 3) & ~3
    verts = np.frombuffer(b, '<f4', 3 * nv, o).reshape(nv, 3).astype(np.float64)
    return verts, parts
