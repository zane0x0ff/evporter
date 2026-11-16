import struct

import numpy as np

from . import formats as F, catio, tmdwrite as W


class ExportError(ValueError):
    pass


def mesh_in(source, pos, nrm, uvs, col, bones, weights, tris, tri_mat, tags=None, new=False, name='', flags=None):
    return dict(flags=flags, source=source, pos=np.asarray(pos, np.float64), nrm=np.asarray(nrm, np.float64),
                uvs=[None if u is None else np.asarray(u, np.float64) for u in uvs],
                col=None if col is None else np.asarray(col, np.float64),
                bones=None if bones is None else np.asarray(bones, np.int64),
                weights=None if weights is None else np.asarray(weights, np.float64),
                tris=np.asarray(tris, np.int64).reshape(-1, 3), tri_mat=np.asarray(tri_mat, np.int64),
                tags=tags, new=new, name=name)


def tag_of(R, mi):
    """the tag string of template mesh mi (None: no tag, Mesh +0x28 = 0xFFFFFFFF)"""
    o = struct.unpack_from('<I', R.meshes[mi]['raw'], 0x28)[0]
    pool = R.user_raw
    if o == 0xFFFFFFFF or o >= len(pool):
        return None
    e = pool.find(b'\0', o)
    return pool[o:e if e >= 0 else len(pool)].decode('latin-1')


def _batches(tris_bones, limit):
    """greedy split of triangles (in order) into batches whose bone sets stay within limit;
    tris_bones: list of sets per triangle -> list of (triangle indices, sorted bones)"""
    out, cur, bones = [], [], set()
    for t, b in enumerate(tris_bones):
        if len(b) > limit:
            raise ExportError('a triangle uses %d bones, more than the %d a skin batch holds' % (len(b), limit))
        if cur and len(bones | b) > limit:
            out.append((cur, sorted(bones)))
            cur, bones = [], set()
        cur.append(t)
        bones |= b
    if cur:
        out.append((cur, sorted(bones)))
    return out


def _tangents(pos, nrm, uv, tris):
    """per-vertex tangent (dP/du, area weighted, orthogonalised to N) and binormal = cross(N, T)"""
    T = np.zeros_like(pos)
    if uv is not None and len(tris):
        p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
        t0, t1, t2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
        e1, e2, d1, d2 = p1 - p0, p2 - p0, t1 - t0, t2 - t0
        det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        ok = np.abs(det) > 1e-12
        r = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tt = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
        for k in range(3):
            np.add.at(T, tris[:, k], tt)
    T = T - nrm * (T * nrm).sum(1, keepdims=True)
    ln = np.linalg.norm(T, axis=1, keepdims=True)
    alt = np.cross(nrm, np.where(np.abs(nrm[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    T = np.where(ln > 1e-12, T / np.where(ln > 1e-12, ln, 1), alt / np.maximum(np.linalg.norm(alt, axis=1, keepdims=True), 1e-12))
    B = np.cross(nrm, T)
    return T, B


def _unit(v):
    v = np.asarray(v, np.float64)
    ln = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(ln > 1e-12, v / np.where(ln > 1e-12, ln, 1), np.array([[0.0, 1.0, 0.0]]))


def build_model(template, meshes_in, keep_missing=True, raw=None):
    """rebuilt model bytes and a report dict; raw: a tmdwrite.Raw of the template already given new materials /
    texture indices (export_catalogue)"""
    R = raw if raw is not None else W.Raw(template)
    M = F.Tmd2(template)
    flags = R.flags
    dt = R.dtype
    palette_limit = struct.unpack_from('<H', template, 8)[0]
    has_skin = bool(flags & F.VF_SKIN)
    nb = len(M.bones)
    if has_skin and not palette_limit and nb > 256:
        raise ExportError('model without skin batches but %d bones (vertex bone indices are bytes)' % nb)

    out_meshes = []
    by_src = {}
    extra = []
    for m in meshes_in:
        if m['source'] is None or not 0 <= m['source'] < len(R.meshes):
            raise ExportError('mesh %r has no source mesh in %s' % (m['name'], M.name))
        if m['new'] or m['source'] in by_src:
            extra.append(m)
        else:
            by_src[m['source']] = m
    for mi in range(len(R.meshes)):
        out_meshes.append((mi, by_src.get(mi)))
    for m in extra:
        out_meshes.append((m['source'], m))

    verts = []
    nvert = 0
    groups, skin_batches, meshes, tags = [], [], [], []
    template_vertices = R.vertices
    kept_map = None

    def keep_vertices(idx):
        nonlocal nvert, kept_map
        if kept_map is None:
            kept_map = np.full(len(template_vertices), -1, np.int64)
        new = np.unique(idx[kept_map[idx] < 0])
        kept_map[new] = nvert + np.arange(len(new))
        verts.append(template_vertices[new])
        nvert += len(new)
        return kept_map[idx]

    report = dict(replaced=0, kept=0, emptied=0, added=0, vertices=0, triangles=0, batches=0)
    for out_i, (src, m) in enumerate(out_meshes):
        raw = bytearray(R.meshes[src]['raw'])
        if m is not None and m.get('flags') is not None:
            raw[0x1A] = int(m['flags']) & 0xFF
        tcmds = R.meshes[src]['commands']
        set_type = next((c >> 12 for c in tcmds if c >> 12 in (2, 5, 6)), 6)
        ends = bool(tcmds) and tcmds[-1] >> 12 == 1
        tag = tag_of(R, src) if m is None or m['tags'] is None else m['tags']
        cmds = []
        if m is None:
            if keep_missing:
                for c in tcmds:
                    t, a = c >> 12, c & 0xFFF
                    if t == 3 and a < len(R.groups):
                        groups.append(keep_vertices(R.groups[a].astype(np.int64).ravel()).reshape(-1, 3))
                        cmds.append((3 << 12) | (len(groups) - 1))
                    elif t == 4 and a < len(R.skin_batches):
                        skin_batches.append(list(R.skin_batches[a]))
                        cmds.append((4 << 12) | (len(skin_batches) - 1))
                    else:
                        cmds.append(c)
                report['kept'] += 1
            else:
                cmds = [(1 << 12)] if ends else []
                report['emptied'] += 1
        else:
            cmds = _encode_mesh(m, R, M, dt, flags, palette_limit, groups, skin_batches, verts, nvert, set_type, ends, report)
            nvert = sum(len(v) for v in verts)
            report['added' if (m['new'] or out_i >= len(R.meshes)) else 'replaced'] += 1
        if any((c >> 12) == 3 for c in cmds) and len(groups) > 0xFFF:
            raise ExportError('more than 4095 index groups')
        meshes.append(dict(raw=raw, commands=cmds))
        tags.append(tag)

    allv = np.concatenate(verts) if verts else np.zeros(0, dt)
    if len(allv):
        key = np.ascontiguousarray(allv).view(np.dtype((np.void, dt.itemsize))).ravel()
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        allv = allv[first[order]]
        groups = [rank[inv.ravel()[g]] for g in groups]
    if not (flags & F.VF_INDEX32) and len(allv) > 0xFFFF:
        raise ExportError('%d vertices: this model stores 16-bit indices (at most 65535 vertices); '
                          'reduce the mesh or split fewer UV / normal seams' % len(allv))
    R.vertices = allv
    R.groups = groups
    R.group_pad = [R.group_pad[0] if R.group_pad else 0] * len(groups)
    R.skin_batches = skin_batches
    R.meshes = meshes
    if R.user_raw or any(t for t in tags):
        R.tags = [t for t in tags if t is not None]
        o = 0
        for mm, t in zip(R.meshes, tags):
            if t is None:
                struct.pack_into('<I', mm['raw'], 0x28, 0xFFFFFFFF)
            else:
                struct.pack_into('<I', mm['raw'], 0x28, o)
                o += len(t.encode('latin-1')) + 1
    for k, mm in enumerate(R.meshes):
        mm['bounds'] = R.mesh_bounds(k) or tuple(struct.unpack_from('<6f', mm['raw'], 0))
    R.update_bounds = True
    report['vertices'], report['triangles'] = len(allv), int(sum(len(g) for g in groups))
    report['batches'] = len(skin_batches)
    report['meshes'] = len(R.meshes)
    return R.build(), report


def _encode_mesh(m, R, M, dt, flags, palette_limit, groups, skin_batches, verts, nvert, set_type, ends, report):
    L = len(m['pos'])
    tris, tri_mat = m['tris'], m['tri_mat']
    if len(tris) == 0:
        return [(1 << 12)] if ends else []
    if tris.max() >= L:
        raise ExportError('mesh %s: corner index out of range' % m['name'])
    nmat = len(R.sec['materials']) // 0x14
    if tri_mat.min() < 0 or tri_mat.max() >= nmat:
        raise ExportError('mesh %s: material index outside the model\'s %d materials' % (m['name'], nmat))
    rec = np.zeros(L, dt)
    rec['pos'] = m['pos'].astype(np.float32)
    nrm = np.asarray(m['nrm'], np.float64)
    ln = np.linalg.norm(nrm, axis=1)
    off = ((ln < 0.97) & (ln > 0.1)) | (ln > 1.03)
    nrm[off] = _unit(nrm[off])
    if 'n' in dt.names:
        rec['n'] = W.enc_unit(nrm, 0)
    if 'c' in dt.names:
        col = m['col'] if m['col'] is not None else np.ones((L, 4))
        rec['c'] = np.clip(np.rint(col * 255.0), 0, 255).astype(np.uint8)
    for k, key in enumerate(('uv0', 'uv1', 'uv2')):
        if key in dt.names:
            uv = m['uvs'][k] if k < len(m['uvs']) and m['uvs'][k] is not None else (m['uvs'][0] if m['uvs'][0] is not None else np.zeros((L, 2)))
            rec[key] = W.enc_uv(uv)
    gb = None
    if 'w' in dt.names:
        if m['bones'] is None:
            raise ExportError('mesh %s: the model is skinned but the mesh has no bone weights' % m['name'])
        gb = m['bones'].copy()
        wq = W.enc_weights(m['weights'])
        gb[wq == 0] = -1
        if (gb >= len(M.bones)).any():
            raise ExportError('mesh %s: bone index outside the skeleton' % m['name'])
        rec['w'] = wq
    cmds = []
    mats = list(dict.fromkeys(tri_mat.tolist()))
    for mat in mats:
        sel = np.nonzero(tri_mat == mat)[0]
        cmds.append((set_type << 12) | mat)
        if gb is not None and palette_limit:
            tb = [set(int(b) for b in gb[tris[t]].ravel() if b >= 0) for t in sel]
            parts = _batches(tb, palette_limit)
        else:
            parts = [(list(range(len(sel))), None)]
        for tri_local, pal in parts:
            ts = tris[sel[tri_local]]
            corners = np.unique(ts.ravel())
            r = rec[corners].copy()
            if gb is not None:
                g = gb[corners]
                if pal is not None:
                    lut = {b: i for i, b in enumerate(pal)}
                    loc = np.vectorize(lambda b: lut.get(int(b), 0))(np.maximum(g, 0)) if len(g) else g
                    loc = np.where(g >= 0, loc, 0)
                else:
                    loc = np.maximum(g, 0)
                r['i'] = loc.astype(np.uint8)
            if 't' in dt.names:
                r['t'] = 0
                r['b'] = 0
            key = r.view(np.dtype((np.void, dt.itemsize)))
            uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
            inv = inv.ravel()
            v = r[first]
            remap = np.full(L, -1, np.int64)
            remap[corners] = inv
            local_tris = remap[ts]
            if 't' in dt.names:
                pos = v['pos'].astype(np.float64)
                n = v['n'][:, :3].astype(np.float64) / 255.0 * 2 - 1
                uv = None
                if 'uv0' in dt.names:
                    a = v['uv0'].astype(np.float64)
                    uv = np.stack([a[:, 0] / 1024.0, (a[:, 1] + 1024.0) / 1024.0], 1)
                Tn, Bn = _tangents(pos, _unit(n), uv, local_tris)
                v['t'] = W.enc_unit(Tn, 0)
                v['b'] = W.enc_unit(Bn, 0)
            base = sum(len(x) for x in verts)
            verts.append(v)
            groups.append((local_tris + base).astype(np.int64))
            if pal is not None:
                skin_batches.append(pal)
                cmds.append((4 << 12) | (len(skin_batches) - 1))
            cmds.append((3 << 12) | (len(groups) - 1))
    if ends:
        cmds.append(1 << 12)
    return cmds


def parse_mtl(b):
    n = struct.unpack_from('<I', b, 0)[0]
    parts = list(struct.unpack_from('<%dI' % n, b, 4))
    o = 4 + 4 * n
    nm = struct.unpack_from('<I', b, o)[0]
    idx = list(struct.unpack_from('<%db' % nm, b, o + 4))
    cls = list(struct.unpack_from('<%dI' % nm, b, o + 4 + nm))
    return parts, idx, cls, b[o + 4 + 5 * nm:]


def write_mtl(parts, cls, tail=b''):
    idx = [parts.index(c) if c in parts else -1 for c in cls]
    return (struct.pack('<I%dI' % len(parts), len(parts), *parts) + struct.pack('<I', len(cls)) +
            struct.pack('<%db' % len(cls), *idx) + struct.pack('<%dI' % len(cls), *cls) + tail)


def texture_family(model, mesh_commands, tex_names):
    """the family (texture name up to the first '_') of the colour texture (slot 0) of the mesh's first material"""
    for c in mesh_commands:
        if c >> 12 in (2, 5, 6) and (c & 0xFFF) < len(model.materials):
            sl = model.materials[c & 0xFFF]['slots']
            if sl:
                name = tex_names.get(sl[0]['hash'], '')
                return name.split('_')[0] if name else None
    return None


def rebuild_mtl(table, template_model, new_model_bytes, tex_names, sources):
    """mtl.csv for the rebuilt model: the class of each mesh follows its colour texture family (the rule of the
    shipped costumes, docs 3.7); a family the template does not have takes the class of the mesh's source"""
    parts, idx, cls, tail = parse_mtl(table)
    T = template_model
    fam_cls = {}
    for mi, mm in enumerate(T.meshes):
        if mi < len(cls):
            f = texture_family(T, [(t << 12) | a for t, a in mm['commands']], tex_names)
            fam_cls.setdefault(f, cls[mi])
    N = F.Tmd2(new_model_bytes)
    out = []
    for mi, mm in enumerate(N.meshes):
        f = texture_family(N, [(t << 12) | a for t, a in mm['commands']], tex_names)
        src = sources[mi] if mi < len(sources) else 0
        out.append(fam_cls.get(f, cls[src] if src < len(cls) else (cls[0] if cls else 0)))
    return write_mtl(parts, out, tail)


def model_leaves(root):
    """{model name: (list node, child index)} of the tmd2 leaves; {model name: (list, index)} of the mtl tables"""
    models, tables = {}, {}
    for path, ch, parent, i, label in catio.walk(root):
        if not isinstance(ch, catio.Leaf):
            continue
        nm = parent.names()
        if ch.data[:3] == b'tmd' and label:
            models[label] = (parent, i)
        elif nm and nm[0] == 'mtl.csv' and len(nm) > 1 and nm[1].endswith('.bin'):
            tables[nm[1][:-4]] = (parent, i)
    return models, tables


def replace_leaves(root, leaves):
    """replace named leaves (collision entries ...) of a catalogue tree: {name: bytes}; returns the names done"""
    named, _ = catio.leaves(root)
    done = []
    for name, data in leaves.items():
        if name in named:
            parent, i = named[name]
            parent.children[i] = catio.Leaf(data)
            done.append(name)
    return done


def _sorted_names(names):
    b = [n.encode('latin-1') for n in names]
    return b == sorted(b)


def texture_groups(root):
    """[(list node, pack)] of the catalogue's texture groups (a name list + a texture pack)"""
    out = []
    for path, ch, parent, i, label in catio.walk(root):
        if isinstance(ch, catio.Pack):
            out.append((parent, ch))
    return out


def model_group(root, raw):
    """the texture group a model's indices point into: the group whose names agree with the model's texture table
    (name hash at each index) and slots; None if no group fits"""
    tt = raw.sec['textable']
    pairs = [struct.unpack_from('<IB', tt, 12 * k) for k in range(len(tt) // 12)]
    sl = raw.sec['slots']
    pairs += [struct.unpack_from('<IH', sl, 12 * k) for k in range(len(sl) // 12)]
    best, score = None, -1
    for parent, pack in texture_groups(root):
        hashes = [F.tamsoft_hash(n) for n in parent.names()]
        ok = sum(1 for h, i in pairs if i < len(hashes) and hashes[i] == h)
        if ok > score:
            best, score = (parent, pack), ok
    return best


def add_textures(root, new_textures, groups):
    """insert new textures into texture groups. new_textures: {name: dict(dds=bytes, model=model name)}; groups:
    {model name: (list node, pack)}. Each texture goes into its model's group at its place in name order (the group
    is sorted by name; TexSlot / texture table indices are positions in it). Returns ({id(pack): {hash: new index}}
    for the changed groups, report lines)"""
    remap, lines = {}, []
    by_pack = {}
    for name, nt in new_textures.items():
        g = groups.get(nt.get('model'))
        if g is None:
            raise ExportError('texture %s: no texture group found for model %s' % (name, nt.get('model')))
        parent, pack = g
        if name in parent.names():
            raise ExportError('texture %s already exists in the group: give the new texture another name' % name)
        by_pack.setdefault(id(pack), (parent, pack, []))[2].append((name, nt['dds']))
    for key, (parent, pack, add) in by_pack.items():
        names = parent.names()
        files = list(pack.files)
        keep_sorted = _sorted_names(names)
        for name, dds in add:
            k = len(names)
            if keep_sorted:
                k = 0
                while k < len(names) and names[k].encode('latin-1') < name.encode('latin-1'):
                    k += 1
            names.insert(k, name)
            files.insert(k, bytes(dds))
            lines.append('texture %s added (%dx%d)' % (name, *struct.unpack_from('<II', dds, 12)[::-1]))
        if len(names) > 256:
            raise ExportError('a texture group holds at most 256 textures')
        pack.files = files
        parent.children[0] = catio.Leaf(catio.names_bytes(names))
        remap[key] = {F.tamsoft_hash(n): k for k, n in enumerate(names)}
    return remap, lines


def _twin_key(T, mi, tex_names):
    """what makes two materials copies of each other: shader, channel values, texture names without the outfit
    state (_brk0, _brk1 ...) and wet suffixes"""
    import re
    m = T.materials[mi]
    names = []
    for sl in m['slots']:
        n = tex_names.get(sl['hash'], '%08x' % sl['hash'])
        names.append(re.sub(r'_(brk\d+|wet)(?=_|$)', '', n))
    return (m['fourcc'], tuple(round(float(x), 6) for x in T.material_params(mi)), tuple(names))


def _apply_channels(name, R, T, chans, twins, lines, tex_names):
    """write the scene's channel values into R; twins: give materials identical to an edited one (the outfit-state
    copies) the same changes when the scene does not edit them itself. Returns whether R changed"""
    touched = False
    n_tpl = len(T.materials)
    given = dict(chans)

    def edited(mi):
        vals = chans.get(mi)
        if vals is None or mi >= R.num_materials:
            return False
        cur = [v for _, v in R.channels(mi)]
        return any(v is not None and abs(float(v) - cur[k]) > 1e-6 for k, v in enumerate(vals) if k < len(cur))
    if twins:
        keys = {mi: _twin_key(T, mi, tex_names) for mi in range(n_tpl)}
        for mi, vals in list(chans.items()):
            if mi >= n_tpl:
                continue
            cur = [v for _, v in R.channels(mi)]
            changed = {k: v for k, v in enumerate(vals) if k < len(cur) and v is not None and abs(float(v) - cur[k]) > 1e-6}
            if not changed:
                continue
            for mj in range(n_tpl):
                if mj != mi and not edited(mj) and keys[mj] == keys[mi]:
                    given[mj] = [changed.get(k) for k in range(len(cur))]
                    lines.append('%s: material %d gets the changes of material %d (an identical copy: %s)'
                                 % (name, mj, mi, T.materials[mj]['fourcc']))
    for mi in sorted(given):
        if not 0 <= mi < R.num_materials:
            continue
        written, skipped = R.set_channels(mi, given[mi])
        if written:
            touched = True
            lines.append('%s: material %d channels %s' % (name, mi, ', '.join('ch%d %.6g -> %.6g' % w for w in written)))
        for k, nk in skipped:
            lines.append('%s: material %d ch%d is animated (%d keys): left unchanged' % (name, mi, k, nk))
    return touched


def export_catalogue(template_cat, edits, keep_missing=True, dds=None, leaves=None, new_materials=None,
                     new_textures=None, channels=None, twins=True):
    """template_cat: the original .cat bytes; edits: {model name: [mesh_in ...]}; dds: {texture name: DDS bytes};
    leaves: {entry name: bytes} other entries to replace;
    new_textures: {name: dict(dds=bytes, like=existing texture)} textures to add (add_textures);
    new_materials: {model name: [dict(base=material index, slots={slot: texture name})]}: copies of materials with
    other textures, numbered after the model's own materials in list order (mesh_in tri_mat may use them);
    channels: {model name: {material index: [ch0 .. chN]}} the scene's channel values (new materials included);
    twins: identical copies of an edited material (outfit states) get the same changes
    -> (new catalogue bytes, report lines)"""
    root, tail = catio.read(template_cat)
    lines = []
    models, tables = model_leaves(root)
    raws = {name: W.Raw(parent.children[i].data) for name, (parent, i) in models.items()}
    groups = {name: model_group(root, R) for name, R in raws.items()}
    remap = {}
    if new_textures:
        remap, tl = add_textures(root, new_textures, groups)
        lines += tl
    _, textures = catio.leaves(root)
    tex_names = {F.tamsoft_hash(n): n for n in textures}
    new_materials = new_materials or {}
    for name in new_materials:
        if name not in models:
            raise ExportError('model %s is not in the template catalogue' % name)
    for name, (parent, i) in models.items():
        tpl = parent.children[i].data
        R = raws[name]
        g = groups[name]
        gnames = g[0].names() if g is not None else []
        gfiles = g[1].files if g is not None else []
        index_of = {F.tamsoft_hash(n): k for k, n in enumerate(gnames)}
        dims = {F.tamsoft_hash(n): struct.unpack_from('<II', f, 12)[::-1] for n, f in zip(gnames, gfiles) if f[:4] == b'DDS '}
        touched = False
        if g is not None and id(g[1]) in remap:
            R.remap_textures(remap[id(g[1])])
            touched = True
        for nm in new_materials.get(name, []):
            base = nm['base']
            if not 0 <= base < R.num_materials:
                raise ExportError('%s: material %d does not exist' % (name, base))
            old_slots = R.material_slots(base)
            changes = {}
            for k, tname in nm['slots'].items():
                h = F.tamsoft_hash(tname)
                if h not in index_of:
                    raise ExportError('%s: texture %s is not in the model\'s texture group' % (name, tname))
                idx = index_of[h]
                changes[int(k)] = (h, idx)
                if R.texture_entry(h) is None:
                    like = R.texture_entry(old_slots[int(k)][0]) if int(k) < len(old_slots) else None
                    tag = like[10:12].decode('latin-1') if like is not None else 'RR'
                    w, hh = dims.get(h, (0, 0))
                    R.add_texture_entry(h, idx, w, hh, tag)
            mi = R.add_material(base, changes)
            lines.append('%s: material %d = copy of %d (%s) with %s' % (name, mi, base, F.Tmd2(tpl).materials[base]['fourcc'],
                                                                     ', '.join('t%s %s' % kv for kv in sorted(nm['slots'].items()))))
            touched = True
        if (channels or {}).get(name):
            if _apply_channels(name, R, F.Tmd2(tpl), channels[name], twins, lines, tex_names):
                touched = True
        if name in edits:
            new, rep = build_model(tpl, edits[name], keep_missing, raw=R)
            parent.children[i] = catio.Leaf(new)
            lines.append('%s: %d meshes (%d replaced, %d added, %d kept, %d emptied), %d vertices, %d triangles, '
                         '%d skin batches' % (name, rep['meshes'], rep['replaced'], rep['added'], rep['kept'],
                                              rep['emptied'], rep['vertices'], rep['triangles'], rep['batches']))
            if name in tables:
                tp, ti = tables[name]
                T = F.Tmd2(tpl)
                n_tpl = len(T.meshes)
                mins = edits[name]
                sources = list(range(n_tpl)) + [m['source'] for m in mins if m['new'] or
                                                sum(1 for x in mins if x['source'] == m['source']) > 1][:max(0, rep['meshes'] - n_tpl)]
                tp.children[ti] = catio.Leaf(rebuild_mtl(tp.children[ti].data, T, new, tex_names, sources))
                lines.append('%s.bin (mtl.csv): rebuilt for %d meshes' % (name, rep['meshes']))
        elif touched:
            parent.children[i] = catio.Leaf(R.build())
            lines.append('%s: materials updated' % name)
    for name in edits:
        if name not in models:
            raise ExportError('model %s is not in the template catalogue (models: %s)' % (name, ', '.join(models)))
    for tname, data in (dds or {}).items():
        if tname not in textures:
            raise ExportError('texture %s is not in the catalogue' % tname)
        if data[:4] != b'DDS ':
            raise ExportError('%s: not a DDS file' % tname)
        pack, k = textures[tname]
        old = pack.files[k]
        pack.files[k] = bytes(data)
        lines.append('texture %s replaced (%d -> %d bytes)%s' % (
            tname, len(old), len(data), '' if old[12:20] == data[12:20] else ', new size %dx%d' % struct.unpack_from('<II', data, 16)[::-1]))
    for name in replace_leaves(root, leaves or {}):
        lines.append('%s replaced' % name)
    return catio.write(root, tail), lines


def meshes_from_model(M, which=None):
    """MeshIn per mesh of a parsed Tmd2, from its own data (global bones): the identity input, for tests"""
    out = []
    for mi, mesh in enumerate(M.meshes):
        if which is not None and mi not in which:
            continue
        tris, tmat, pal_of = [], [], []
        mat, pal = 0, None
        for t, a in mesh['commands']:
            if t in (2, 5, 6):
                mat = a
            elif t == 4 and a < len(M.skin_batches):
                pal = a
            elif t == 3 and a < len(M.groups):
                tris.append(M.groups[a]); tmat.append(np.full(len(M.groups[a]), mat)); pal_of += [pal] * len(M.groups[a])
        if not tris:
            continue
        tris = np.concatenate(tris)
        corners = tris.ravel()
        bones = None
        if M.bone_idx is not None:
            loc = M.bone_idx[corners]
            pals = np.repeat(np.array([-1 if p is None else p for p in pal_of]), 3)
            bones = loc.copy()
            for k, p in enumerate(pals):
                if p >= 0:
                    tb = M.skin_batches[p]
                    bones[k] = [tb[x] if x < len(tb) else 0 for x in loc[k]]
        out.append(mesh_in(mi, M.pos[corners], M.normal[corners], [u[corners] if u is not None else None for u in M.uv],
                           M.color[corners] if M.color is not None else None, bones,
                           M.weights[corners] if M.weights is not None else None,
                           np.arange(len(corners)).reshape(-1, 3), np.concatenate(tmat), name='%s_%02d' % (M.name, mi)))
    return out
