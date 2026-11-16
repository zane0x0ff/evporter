import struct

import numpy as np

from . import formats as F

A16 = lambda n: (n + 15) & ~15

PTRS = [(0x28, 'meshes'), (0x30, 'static'), (0x38, 'anim'), (0x40, 'commands'), (0x48, 'materials'), (0x50, 'params'),
        (0x58, 'user'), (0x60, 'keys'), (0x68, 'groups'), (0x70, 'indices'), (0x78, 'textable'), (0x80, 'unk80'),
        (0x88, 'slots'), (0x90, 'vertices')]
EXT = [(0xE0, 'skinb'), (0xE8, 'palette'), (0xF0, 'invbind'), (0xF8, 'bones')]
RANK = ['meshes', 'static', 'anim', 'commands', 'materials', 'params', 'user', 'keys', 'skinb', 'palette', 'groups',
        'indices', 'textable', 'unk80', 'slots', 'invbind', 'bones', 'vertices']


def vertex_dtype(f):
    fields = [('pos', '<f4', (3,))]
    if f & F.VF_SKIN:
        fields += [('w', 'u1', (4,)), ('i', 'u1', (4,))]
    if f & F.VF_NORMAL:
        fields += [('n', 'u1', (4,))]
    if f & F.VF_TANGENTS:
        fields += [('t', 'u1', (4,)), ('b', 'u1', (4,))]
    if f & F.VF_COLOR:
        fields += [('c', 'u1', (4,))]
    if f & F.VF_U100:
        fields += [('x100', 'u1', (4,))]
    if f & F.VF_U200:
        fields += [('x200', 'u1', (4,))]
    for bit, key in ((F.VF_UV0, 'uv0'), (F.VF_UV1, 'uv1'), (F.VF_UV2, 'uv2')):
        if f & bit:
            fields += [(key, '<i2', (2,))]
    return np.dtype(fields)


class Raw:
    """a .tmd2 split into its sections (raw bytes) + the decoded geometry the writer rebuilds"""

    def __init__(self, data):
        d = self.d = bytes(data)
        q = lambda o: struct.unpack_from('<Q', d, o)[0]
        u = lambda o: struct.unpack_from('<I', d, o)[0]
        self.flags = struct.unpack_from('<H', d, 6)[0]
        self.skel = bool(self.flags & F.VF_SKELETON)
        self.hlen = 0x110 if self.skel else 0xE0
        self.header = bytearray(d[:self.hlen])
        self.ptr = {n: q(o) for o, n in PTRS + (EXT if self.skel else [])}
        n_mesh, n_static, n_anim, n_cmd, n_mat, n_par, user_size, n_keys, n_grp, n_tri, n_tex, n_80, n_slot, n_vert = \
            struct.unpack_from('<14I', d, 0xA8)
        self.n_vert = n_vert
        key_size = 8 if struct.unpack_from('<H', d, 0xA)[0] == 0x201 else 4
        ext = struct.unpack_from('<4I', d, 0x100) if self.skel else (0, 0, 0, 0)
        self.n_skinb, self.palette_size = ext[0], ext[1]
        sizes = dict(meshes=0x38 * n_mesh, static=0x24 * n_static, anim=0x50 * n_anim, materials=0x14 * n_mat,
                     params=8 * n_par, user=user_size, keys=key_size * n_keys, textable=12 * n_tex, unk80=20 * n_80,
                     slots=12 * n_slot, invbind=64 * ext[2], bones=24 * ext[3])
        self.sec = {k: d[self.ptr[k]:self.ptr[k] + v] if self.ptr.get(k) else b'' for k, v in sizes.items()}
        self.dtype = vertex_dtype(self.flags)
        self.vertices = np.frombuffer(d, self.dtype, n_vert, self.ptr['vertices']).copy()
        self.idx32 = bool(self.flags & F.VF_INDEX32)
        self.groups, self.group_pad = [], []
        for g in range(n_grp):
            tris, pad, first = struct.unpack_from('<IIQ', d, self.ptr['groups'] + 16 * g)
            self.group_pad.append(pad)
            self.groups.append(np.frombuffer(d, '<u4' if self.idx32 else '<u2', tris * 3,
                                             self.ptr['indices'] + first * 3 * (4 if self.idx32 else 2)).reshape(-1, 3).copy())
        self.meshes = []
        for m in range(n_mesh):
            o = self.ptr['meshes'] + 0x38 * m
            ncmd = struct.unpack_from('<H', d, o + 0x18)[0]
            first = q(o + 0x20)
            cmds = list(struct.unpack_from('<%dH' % ncmd, d, self.ptr['commands'] + 2 * first)) if ncmd else []
            self.meshes.append(dict(raw=bytearray(d[o:o + 0x38]), commands=cmds))
        self.skin_batches, self.skin_pad = [], []
        for s in range(self.n_skinb):
            cnt, pad, off = struct.unpack_from('<iIQ', d, self.ptr['skinb'] + 16 * s)
            self.skin_batches.append(list(d[self.ptr['palette'] + off:self.ptr['palette'] + off + max(0, cnt)]))
            self.skin_pad.append(pad)
        self.user_raw = self.sec['user']
        self.tags = None
        self.update_bounds = False

    def build(self):
        """the model bytes: sections in the template's order (16-byte aligned), geometry from the fields"""
        out = bytearray(self.header)
        P = {}
        cmds, mesh_bytes = [], bytearray()
        for m in self.meshes:
            r = bytearray(m['raw'])
            struct.pack_into('<H', r, 0x18, len(m['commands']))
            struct.pack_into('<Q', r, 0x20, len(cmds))
            if m.get('bounds') is not None:
                struct.pack_into('<6f', r, 0, *m['bounds'])
            cmds += m['commands']
            mesh_bytes += r
        sec = dict(self.sec)
        sec['meshes'] = bytes(mesh_bytes)
        sec['commands'] = struct.pack('<%dH' % len(cmds), *cmds) if cmds else b''
        sec['user'] = self.user_raw if self.tags is None else b''.join(t.encode('latin-1') + b'\0' for t in self.tags)
        pal = bytearray()
        if self.skel:
            sb = bytearray()
            for k, b in enumerate(self.skin_batches):
                pad = self.skin_pad[k] if k < len(self.skin_pad) else (self.skin_pad[0] if self.skin_pad else 0)
                sb += struct.pack('<iIQ', len(b), pad, len(pal))
                pal += bytes(b)
            sec['skinb'], sec['palette'] = bytes(sb), bytes(pal)
        idt = '<u4' if self.idx32 else '<u2'
        grp, ind, first = bytearray(), [], 0
        for k, g in enumerate(self.groups):
            pad = self.group_pad[k] if k < len(self.group_pad) else 0
            grp += struct.pack('<IIQ', len(g), pad, first)
            ind.append(np.asarray(g).astype(idt).ravel())
            first += len(g)
        sec['groups'] = bytes(grp)
        sec['indices'] = np.concatenate(ind).astype(idt).tobytes() if ind else b''
        sec['vertices'] = self.vertices.astype(self.dtype).tobytes()
        names = [n for _, n in PTRS + (EXT if self.skel else [])]
        rank = {n: i for i, n in enumerate(RANK)}
        placed = sorted((n for n in names if self.ptr.get(n)), key=lambda n: (self.ptr[n], rank[n]))
        placed += [n for n in sorted(names, key=lambda n: rank[n]) if not self.ptr.get(n) and sec.get(n)]
        for n in placed:
            out += b'\0' * (A16(len(out)) - len(out))
            P[n] = len(out)
            out += sec.get(n, b'')
        for o, n in PTRS + (EXT if self.skel else []):
            struct.pack_into('<Q', out, o, P.get(n, 0))
        struct.pack_into('<I', out, 0xA8, len(self.meshes))
        struct.pack_into('<I', out, 0xB8, len(sec['materials']) // 0x14)
        struct.pack_into('<I', out, 0xBC, len(sec['params']) // 8)
        struct.pack_into('<I', out, 0xD0, len(sec['textable']) // 12)
        struct.pack_into('<I', out, 0xD8, len(sec['slots']) // 12)
        struct.pack_into('<I', out, 0xB4, len(cmds))
        struct.pack_into('<I', out, 0xC0, len(sec['user']))
        struct.pack_into('<I', out, 0xC8, len(self.groups))
        struct.pack_into('<I', out, 0xCC, first)
        struct.pack_into('<I', out, 0xDC, len(self.vertices))
        if self.skel:
            struct.pack_into('<I', out, 0x100, len(self.skin_batches))
            struct.pack_into('<I', out, 0x104, len(pal))
        if self.update_bounds and len(self.vertices):
            p = self.vertices['pos'].astype(np.float32)
            struct.pack_into('<6f', out, 0x10, *p.min(0), *p.max(0))
        return bytes(out)

    @property
    def num_materials(self):
        return len(self.sec['materials']) // 0x14

    def material_slots(self, mi):
        """[(hash, texture index, raw 12 bytes)] of material mi"""
        r = self.sec['materials'][0x14 * mi:0x14 * mi + 0x14]
        first, n = struct.unpack_from('<HH', r, 8)
        sl = self.sec['slots']
        return [(struct.unpack_from('<I', sl, 12 * (first + k))[0], struct.unpack_from('<H', sl, 12 * (first + k) + 4)[0],
                 sl[12 * (first + k):12 * (first + k) + 12]) for k in range(n)]

    def remap_textures(self, new_index):
        """new texture indices by name hash ({hash: index}, the catalogue's texture group after an insertion):
        TexSlot +4 (u16) and the texture table's +4 (u8)"""
        sl = bytearray(self.sec['slots'])
        for k in range(len(sl) // 12):
            h = struct.unpack_from('<I', sl, 12 * k)[0]
            if h in new_index:
                struct.pack_into('<H', sl, 12 * k + 4, new_index[h])
        tt = bytearray(self.sec['textable'])
        for k in range(len(tt) // 12):
            h = struct.unpack_from('<I', tt, 12 * k)[0]
            if h in new_index:
                if new_index[h] > 255:
                    raise ValueError('more than 256 textures in the texture group')
                tt[12 * k + 4] = new_index[h]
        self.sec['slots'], self.sec['textable'] = bytes(sl), bytes(tt)

    def add_texture_entry(self, h, index, width, height, tag):
        """texture table record {u32 hash, u8 index, u8 0, u16 width, u16 height, char tag[2]} (tag: the address
        modes, 'RR' repeat, 'EE' clamp ...), kept sorted by index"""
        tt = self.sec['textable']
        recs = [tt[12 * k:12 * k + 12] for k in range(len(tt) // 12)]
        if any(struct.unpack_from('<I', r, 0)[0] == h for r in recs):
            return
        recs.append(struct.pack('<IBBHH2s', h, index, 0, width, height, tag.encode('latin-1')[:2].ljust(2, b'R')))
        recs.sort(key=lambda r: r[4])
        self.sec['textable'] = b''.join(recs)

    def texture_entry(self, h):
        tt = self.sec['textable']
        for k in range(len(tt) // 12):
            if struct.unpack_from('<I', tt, 12 * k)[0] == h:
                return tt[12 * k:12 * k + 12]
        return None

    def add_material(self, base, slot_changes):
        """a copy of material base (record, channels, texture slots) with some slots pointing elsewhere:
        slot_changes {slot: (hash, texture index)}; returns the new material's index"""
        mats = self.sec['materials']
        r = bytearray(mats[0x14 * base:0x14 * base + 0x14])
        first_slot, n_slot, first_par, n_par = struct.unpack_from('<4H', r, 8)
        sl = bytearray(self.sec['slots'][12 * first_slot:12 * (first_slot + n_slot)])
        for k, (h, idx) in slot_changes.items():
            if 0 <= k < n_slot:
                struct.pack_into('<I', sl, 12 * k, h)
                struct.pack_into('<H', sl, 12 * k + 4, idx)
        pars = self.sec['params'][8 * first_par:8 * (first_par + n_par)]
        new_slot, new_par = len(self.sec['slots']) // 12, len(self.sec['params']) // 8
        if new_slot + n_slot > 0xFFFF or new_par + n_par > 0xFFFF:
            raise ValueError('too many texture slots / channels')
        struct.pack_into('<HH', r, 8, new_slot, n_slot)
        struct.pack_into('<HH', r, 12, new_par, n_par)
        self.sec['slots'] += bytes(sl)
        self.sec['params'] += pars
        self.sec['materials'] += bytes(r)
        return len(self.sec['materials']) // 0x14 - 1

    def _key_size(self):
        return 8 if struct.unpack_from('<H', self.d, 0xA)[0] == 0x201 else 4

    def channel_range(self, mi):
        """(first record, count) of material mi's channel records in the params section"""
        r = self.sec['materials'][0x14 * mi:0x14 * mi + 0x14]
        return struct.unpack_from('<HH', r, 12)

    def channels(self, mi):
        """[(number of keys, value at frame 0)] of material mi. A record is {u16 keys, u16 pad, u32}: no keys ->
        the u32 is the float value; else the index of its first key (v0x201 keys {u16 frame, pad, float}, older
        versions plain floats)"""
        first, n = self.channel_range(mi)
        pars, keys, ks = self.sec['params'], self.sec['keys'], self._key_size()
        out = []
        for k in range(n):
            nk, _, v = struct.unpack_from('<HHI', pars, 8 * (first + k))
            if nk == 0:
                out.append((0, struct.unpack_from('<f', pars, 8 * (first + k) + 4)[0]))
            else:
                o = ks * v + (4 if ks == 8 else 0)
                out.append((nk, struct.unpack_from('<f', keys, o)[0] if o + 4 <= len(keys) else 0.0))
        return out

    def set_channels(self, mi, values, eps=1e-6):
        """write channel values of material mi (values[k] None: unchanged). A changed channel becomes a constant
        (a single-key channel is constant anyway); an animated one (2+ keys) is left alone. When another material
        shares the records they are copied first. Returns ([(k, old, new)] written, [(k, keys)] skipped)"""
        cur = self.channels(mi)
        todo = [(k, float(v)) for k, v in enumerate(values) if k < len(cur) and v is not None and abs(float(v) - cur[k][1]) > eps]
        written, skipped = [], []
        todo2 = []
        for k, v in todo:
            if cur[k][0] > 1:
                skipped.append((k, cur[k][0]))
            else:
                todo2.append((k, v))
        if not todo2:
            return written, skipped
        first, n = self.channel_range(mi)
        shared = any(j != mi and self.channel_range(j)[0] < first + n and first < sum(self.channel_range(j))
                     for j in range(self.num_materials))
        if shared:
            new_first = len(self.sec['params']) // 8
            if new_first + n > 0xFFFF:
                raise ValueError('too many channel records')
            self.sec['params'] += self.sec['params'][8 * first:8 * (first + n)]
            mats = bytearray(self.sec['materials'])
            struct.pack_into('<H', mats, 0x14 * mi + 12, new_first)
            self.sec['materials'] = bytes(mats)
            first = new_first
        pars = bytearray(self.sec['params'])
        for k, v in todo2:
            struct.pack_into('<HHf', pars, 8 * (first + k), 0, 0, v)
            written.append((k, cur[k][1], v))
        self.sec['params'] = bytes(pars)
        return written, skipped

    def mesh_bounds(self, mi):
        """bind-pose box of the vertices mesh mi draws (its index groups)"""
        used = [self.groups[c & 0xFFF].ravel() for c in self.meshes[mi]['commands'] if c >> 12 == 3 and (c & 0xFFF) < len(self.groups)]
        if not used:
            return None
        p = self.vertices['pos'][np.unique(np.concatenate(used))].astype(np.float32)
        return tuple(p.min(0)) + tuple(p.max(0))


def enc_unit(v, w=None):
    """[-1, 1] vectors -> u8x4 as the game stores them (b / 255 * 2 - 1); w: the 4th byte"""
    v = np.asarray(v, np.float64)
    b = np.clip(np.rint((v + 1.0) * 0.5 * 255.0), 0, 255).astype(np.uint8)
    out = np.zeros((len(v), 4), np.uint8)
    out[:, :3] = b
    if w is not None:
        out[:, 3] = w
    return out


def enc_uv(uv):
    """D3D uv (v down) -> s16x2: u = s / 1024, v = (t + 1024) / 1024"""
    uv = np.asarray(uv, np.float64)
    s = np.rint(uv[:, 0] * 1024.0)
    t = np.rint(uv[:, 1] * 1024.0 - 1024.0)
    if np.abs(s).max(initial=0) > 32767 or np.abs(t).max(initial=0) > 32767:
        raise ValueError('UV coordinates outside the range the game stores (|u|, |v - 1| < 32)')
    return np.stack([s, t], 1).astype(np.int16)


def enc_weights(w):
    """weights (n, 4) summing to 1 -> u8x4 summing to exactly 255"""
    w = np.clip(np.asarray(w, np.float64), 0, None)
    s = w.sum(1, keepdims=True)
    w = np.where(s > 0, w / np.where(s > 0, s, 1), np.array([[1.0, 0, 0, 0]]))
    b = np.floor(w * 255.0).astype(np.int64)
    rest = 255 - b.sum(1)
    frac = w * 255.0 - b
    order = np.argsort(-frac, 1)
    for k in range(4):
        add = rest > k
        rows = np.nonzero(add)[0]
        b[rows, order[rows, k]] += 1
    return b.astype(np.uint8)
