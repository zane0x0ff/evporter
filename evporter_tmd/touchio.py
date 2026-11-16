import math
import struct

import numpy as np


def _a4(n):
    return (n + 3) & ~3


class Touch:
    def __init__(self, b=None):
        self.flags, self.ids, self.group_end, self.parts, self.verts, self.cell = 3, [], [], [], np.zeros((0, 3)), 3.0
        if b is not None:
            self._read(bytes(b))

    def _read(self, d):
        u = lambda f, o: struct.unpack_from(f, d, o)
        self.flags, nid, nv = u('<3I', 0)
        w, f = (2, 'H') if self.flags & 1 else (4, 'I')
        self.ids = [u('<2I', 12 + 8 * i) for i in range(nid)]
        o = 12 + 8 * nid
        ng = u('<H', o)[0]
        self.group_end = list(u('<%dH' % ng, o + 2)) if ng else []
        nparts = self.group_end[-1] if ng else 0
        o = _a4(o + 2 + 2 * ng)
        table = [u('<2I', o + 8 * i) for i in range(nparts)]
        base = o + 8 * nparts
        self.parts, start = [], 0
        for idx, end in table:
            p = base + start
            n = u('<' + f, p)[0]; p += w
            face_end = u('<%d%s' % (n, f), p) if n else (); p += w * n
            npoly = face_end[-1] if n else 0
            poly_end = u('<%d%s' % (npoly, f), p) if npoly else (); p += w * npoly
            nidx = poly_end[-1] if npoly else 0
            index = u('<%d%s' % (nidx, f), p) if nidx else ()
            polys = [tuple(index[a:b]) for a, b in zip((0,) + tuple(poly_end[:-1]), poly_end)]
            faces = [polys[a:b] for a, b in zip((0,) + tuple(face_end[:-1]), face_end)]
            self.parts.append((idx, faces))
            start = end
        o = _a4(base + start)
        self.verts = np.frombuffer(d, '<f4', 3 * nv, o).reshape(nv, 3).copy()
        o += 12 * nv
        self.cell = u('<f', o + 8)[0] if self.flags & 2 and o + 24 <= len(d) else 3.0

    def grid(self):
        v = self.verts.astype(np.float32)
        if not len(v):
            return struct.pack('<3f3I', 0, 0, self.cell, 0, 0, 0)
        mn, mx = v.min(0), v.max(0)
        offx, offz = np.float32(-mn[0]), np.float32(-mn[2])
        cs = np.float32(self.cell)
        nx = int(np.floor((mx[0] + offx) / cs)) + 1
        nz = int(np.floor((mx[2] + offz) / cs)) + 1
        cells = [[] for _ in range(nx * nz)]
        for pi, (_, faces) in enumerate(self.parts):
            qi = 0
            for face in faces:
                for poly in face:
                    pts = v[list(poly)]
                    x0 = int(np.floor((pts[:, 0].min() + offx) / cs)); x1 = int(np.floor((pts[:, 0].max() + offx) / cs))
                    z0 = int(np.floor((pts[:, 2].min() + offz) / cs)); z1 = int(np.floor((pts[:, 2].max() + offz) / cs))
                    for iz in range(max(0, z0), min(nz - 1, z1) + 1):
                        for ix in range(max(0, x0), min(nx - 1, x1) + 1):
                            cells[iz * nx + ix].append((pi, qi))
                    qi += 1
        ends, ent, n = [], [], 0
        for c in cells:
            n += len(c)
            ends.append(n)
            ent += c
        return (struct.pack('<3f3I', float(offx), float(offz), float(cs), nx, nz, nx * nz) + struct.pack('<%dI' % len(ends), *ends) +
                b''.join(struct.pack('<2I', a, b) for a, b in ent))

    def bytes(self):
        nidx = max((max(p) for _, faces in self.parts for face in faces for p in face), default=-1) + 1
        if nidx > len(self.verts):
            raise ValueError('collision polygon index beyond the vertices')
        counts = [sum(len(p) for face in faces for p in face) for _, faces in self.parts]
        if len(self.verts) > 0xFFFF or max(counts + [0]) > 0xFFFF:
            self.flags &= ~1
        w, f = (2, 'H') if self.flags & 1 else (4, 'I')
        out = bytearray(struct.pack('<3I', self.flags, len(self.ids), len(self.verts)))
        for a, b in self.ids:
            out += struct.pack('<2I', a, b)
        out += struct.pack('<H', len(self.group_end)) + struct.pack('<%dH' % len(self.group_end), *self.group_end)
        out += b'\0' * (_a4(len(out)) - len(out))
        data, table = bytearray(), []
        for idx, faces in self.parts:
            polys = [p for face in faces for p in face]
            face_end = np.cumsum([len(face) for face in faces]).tolist()
            poly_end = np.cumsum([len(p) for p in polys]).tolist()
            index = [i for p in polys for i in p]
            pd = struct.pack('<' + f, len(faces)) + struct.pack('<%d%s' % (len(face_end), f), *face_end)
            pd += struct.pack('<%d%s' % (len(poly_end), f), *poly_end) + struct.pack('<%d%s' % (len(index), f), *index)
            data += pd
            table.append((idx, len(data)))
        for idx, end in table:
            out += struct.pack('<2I', idx, end)
        out += data
        out += b'\0' * (_a4(len(out)) - len(out))
        out += self.verts.astype('<f4').tobytes()
        if self.flags & 2:
            out += self.grid()
        return bytes(out)

    def polygons(self):
        """[(part index, polygon tuple)] in file order"""
        return [(pi, p) for pi, (_, faces) in enumerate(self.parts) for face in faces for p in face]

    def set_polygons(self, verts, polys, part_of):
        """replace the geometry: verts (n, 3) game space, polys [index tuples], part_of [part index per polygon]
        (parts keep their ids; one face per polygon, as the shipped files)"""
        self.verts = np.asarray(verts, np.float64).reshape(-1, 3)
        n = len(self.parts)
        if n == 0:
            self.parts, self.group_end = [(0, [])], [1]
            self.ids = self.ids or [(0, 0)]
            n = 1
        faces = [[] for _ in range(n)]
        for p, k in zip(polys, part_of):
            faces[k if 0 <= k < n else 0].append([tuple(int(i) for i in p)])
        self.parts = [(idx, faces[i]) for i, (idx, _) in enumerate(self.parts)]
