import os
import struct

import numpy as np
import bpy

from . import formats as F, costume as C, catio

A3_GAME = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def _to_game(v):
    return np.asarray(v, np.float64) @ A3_GAME.T


def _is_outline(o):
    if o.get('tmd_outline_of') or o.data.get('tmd_outline_of'):
        return True
    mats = [s.material for s in o.material_slots if s.material is not None]
    return bool(mats) and all(m.get('tmd_outline_of') for m in mats)


def _belongs(o, arm):
    return o.parent == arm or any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)


def character_objects(arm, skipped=None):
    """the mesh objects the export writes: deformed by / parented to the armature, with TMD data (Mesh.tmd_material
    from the import or Make costume mesh) and Export on; not outline hulls, physics bodies or other helpers.
    skipped: a list that receives (object name, reason) for the character's other mesh objects"""
    out = []
    for o in bpy.data.objects:
        if o.type != 'MESH' or not _belongs(o, arm):
            continue
        t = getattr(o.data, 'tmd_material', None)
        why = None
        if _is_outline(o):
            why = 'outline hull'
        elif o.get('tmd_physics') or o.get('tmd_label') is not None and not (t is not None and t.model):
            why = 'physics body'
        elif t is None or not t.model:
            why = 'no TMD data'
        elif not getattr(t, 'export', True):
            why = 'Export off'
        if why is None:
            out.append(o)
        elif skipped is not None:
            skipped.append((o.name, why))
    return out


def _evaluated_mesh(obj, apply_modifiers):
    """(mesh, cleanup) of the object without its Armature modifiers. The evaluated mesh is copied into a standalone
    datablock (new_from_object) before the modifiers are switched back on: re-evaluating the depsgraph frees the
    evaluated object's own mesh ("StructRNA of type Mesh has been removed")."""
    if not apply_modifiers:
        return obj.data, lambda: None
    saved = [(m, m.show_viewport) for m in obj.modifiers if m.type == 'ARMATURE']
    for m, _ in saved:
        m.show_viewport = False
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev = obj.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    finally:
        for m, v in saved:
            m.show_viewport = v
        bpy.context.evaluated_depsgraph_get().update()
    return me, lambda: bpy.data.meshes.remove(me)


def _corner_normals(me):
    n = len(me.loops)
    a = np.zeros(n * 3, np.float32)
    if hasattr(me, 'corner_normals'):
        me.corner_normals.foreach_get('vector', a)
    else:
        me.calc_normals_split()
        me.loops.foreach_get('normal', a)
    return a.reshape(n, 3).astype(np.float64)


def _colors(me, loop_v):
    attr = me.color_attributes.get('Col') if hasattr(me, 'color_attributes') else None
    if attr is None:
        return None
    a = np.zeros(len(attr.data) * 4, np.float32)
    attr.data.foreach_get('color', a)
    a = a.reshape(-1, 4).astype(np.float64)
    if attr.domain == 'POINT':
        return a[loop_v]
    return a


def skin_matrices(model, arm):
    """per bone of the model: invBind @ rest (game space, row vectors) - the matrix the import applied"""
    rest = {}
    for b in arm.data.bones:
        h, r = b.get('tmd_hash'), b.get('tmd_rest')
        if h is not None and r is not None:
            rest[int(h, 16) if isinstance(h, str) else int(h)] = np.array(list(r), float).reshape(4, 4)
    S, missing = [], 0
    for i, b in enumerate(model.bones):
        W = rest.get(b['hash'])
        ib = model.inv_bind[i] if i < len(model.inv_bind) else np.eye(4)
        if W is None:
            missing += 1
            S.append(np.eye(4))
        else:
            S.append(ib @ W)
    return np.array(S), missing


def mesh_from_object(obj, arm, model, S, apply_modifiers=True, new_index=None):
    """costume.mesh_in for one object; also returns a list of warnings"""
    warn = []
    me, cleanup = _evaluated_mesh(obj, apply_modifiers)
    try:
        me.calc_loop_triangles()
        nl = len(me.loops)
        loop_v = np.zeros(nl, np.int32)
        me.loops.foreach_get('vertex_index', loop_v)
        nv = len(me.vertices)
        co = np.zeros(nv * 3, np.float32)
        me.vertices.foreach_get('co', co)
        co = co.reshape(nv, 3).astype(np.float64)
        M = np.array(arm.matrix_world.inverted() @ obj.matrix_world) if arm is not None else np.array(obj.matrix_world)
        co = co @ M[:3, :3].T + M[:3, 3]
        pos = _to_game(co)
        nrm = _corner_normals(me) @ np.linalg.inv(M[:3, :3])
        nrm = _to_game(nrm)
        uvs = []
        for k, name in enumerate(('UVMap', 'UV1', 'UV2')):
            lay = me.uv_layers.get(name) or (me.uv_layers[0] if k == 0 and len(me.uv_layers) else None)
            if lay is None:
                uvs.append(None)
                continue
            a = np.zeros(nl * 2, np.float32)
            lay.data.foreach_get('uv', a)
            a = a.reshape(nl, 2).astype(np.float64)
            a[:, 1] = 1.0 - a[:, 1]
            uvs.append(a)
        col = _colors(me, loop_v)
        bones = weights = None
        if model.weights is None:
            Mm = model.mesh_matrix(obj.data.tmd_material.mesh_index) if obj.data.tmd_material.mesh_index < len(model.meshes) else None
            if Mm is not None:
                Mi = np.linalg.inv(Mm)
                pos = (np.c_[pos, np.ones(len(pos))] @ Mi)[:, :3]
                nrm = nrm @ Mi[:3, :3]
        elif model.bones:
            if arm is None or S is None:
                raise C.ExportError('%s: the model is skinned; import it with its armature' % obj.name)
            gname = {g.index: g.name for g in obj.vertex_groups}
            ghash = {}
            for gi, name in gname.items():
                b = arm.data.bones.get(name)
                h = b.get('tmd_hash') if b is not None else None
                h = int(h, 16) if isinstance(h, str) else F.tamsoft_hash(name)
                if h in model.bone_index:
                    ghash[gi] = model.bone_index[h]
            vb = np.zeros((nv, 4), np.int64)
            vw = np.zeros((nv, 4), np.float64)
            dropped = 0
            for v in me.vertices:
                inf = sorted(((g.weight, ghash[g.group]) for g in v.groups if g.group in ghash and g.weight > 0), reverse=True)
                if len(inf) > 4:
                    dropped += 1
                for k, (w, b) in enumerate(inf[:4]):
                    vb[v.index, k], vw[v.index, k] = b, w
            unweighted = int((vw.sum(1) <= 0).sum())
            if unweighted:
                warn.append('%s: %d vertices without a bone weight (bound to %s)' % (obj.name, unweighted, F.bone_name(model.bones[0]['hash'])))
            if dropped:
                warn.append('%s: %d vertices had more than 4 bones (the 4 strongest kept)' % (obj.name, dropped))
            s = vw.sum(1, keepdims=True)
            vw = np.where(s > 0, vw / np.where(s > 0, s, 1), np.array([[1.0, 0, 0, 0]]))
            Mv = np.einsum('nk,nkij->nij', vw, S[vb])
            Minv = np.linalg.inv(Mv)
            pos = np.einsum('ni,nij->nj', np.c_[pos, np.ones(nv)], Minv)[:, :3]
            Ml = Minv[loop_v, :3, :3]
            nrm = np.einsum('ni,nij->nj', nrm, Ml)
            bones, weights = vb[loop_v], vw[loop_v]
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.where(ln > 1e-12, nrm / np.where(ln > 1e-12, ln, 1), nrm)
        zat = me.attributes.get('tmd_zero_normal')
        if zat is not None and zat.domain == 'POINT' and len(zat.data) == nv:
            z = np.zeros(nv, bool)
            zat.data.foreach_get('value', z)
            nrm[z[loop_v]] = 127.0 / 255.0 * 2.0 - 1.0
        nt = len(me.loop_triangles)
        tl = np.zeros(nt * 3, np.int32)
        me.loop_triangles.foreach_get('loops', tl)
        tp = np.zeros(nt, np.int32)
        me.loop_triangles.foreach_get('polygon_index', tp)
        pm = np.zeros(len(me.polygons), np.int32)
        me.polygons.foreach_get('material_index', pm)
        slot_mat = []
        fallback = None
        for slot in obj.material_slots:
            m = slot.material
            idx = m.get('tmd_index') if m is not None and m.get('tmd_model', model.name) == model.name else None
            if m is not None and new_index and m.name in new_index:
                idx = new_index[m.name]
            slot_mat.append(None if idx is None else int(idx))
            if idx is not None and fallback is None:
                fallback = int(idx)
        if fallback is None:
            t = obj.data.tmd_material
            fallback = next((it.index for it in t.materials if it.index >= 0), None)
        if fallback is None:
            raise C.ExportError('%s: none of its materials comes from %s' % (obj.name, model.name))
        bad = sum(1 for s in slot_mat if s is None)
        if bad:
            warn.append('%s: %d material slot(s) without a %s material use material %d' % (obj.name, bad, model.name, fallback))
        smap = np.array([fallback if s is None else s for s in slot_mat] or [fallback], np.int64)
        tri_mat = smap[np.clip(pm[tp], 0, len(smap) - 1)]
        corners_pos = pos[loop_v]
        t = obj.data.tmd_material
        tris = tl.reshape(-1, 3)
        if np.linalg.det(M[:3, :3]) < 0:
            tris = tris[:, ::-1]
        if len(tris):
            p = corners_pos[tris]
            fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            against = int(((fn * nrm[tris].sum(1)).sum(1) < 0).sum())
            if against > len(tris) // 2:
                warn.append('%s: %d of %d triangles face against their normals (the game culls them from the front): '
                            'flipped faces? Mesh > Normals > Recalculate Outside' % (obj.name, against, len(tris)))
        return C.mesh_in(t.mesh_index, corners_pos, nrm, uvs, col, bones, weights, tris, tri_mat,
                         tags=t.tags, name=obj.name, flags=getattr(t, 'mesh_flags', None)), warn
    finally:
        cleanup()


def _norm(p):
    """absolute, resolved, case-normalised path (Windows: drive letters and case)"""
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def _inside(path, folder):
    """path is folder or below it; False for another drive (os.path.commonpath raises ValueError there on Windows)"""
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


TEX_NAME_OK = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')


def texture_name(s):
    """a catalogue texture name from an image name: ASCII letters, digits, _ and -, no extension"""
    s = os.path.splitext(s)[0] if s.lower().endswith(('.png', '.dds', '.tga', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.exr')) else s
    out = ''.join(c if c in TEX_NAME_OK else '_' for c in s).strip('_')
    return out[:60] or 'texture'


def slot_roles(fourcc):
    """what each texture slot of a shader is for (viewer ports, docs 3.5 / 3.9)"""
    from . import character
    cc = character._remap(fourcc) if hasattr(character, '_remap') else fourcc
    st = character.stage_info(fourcc, '', True)
    if st is not None:
        k = st['kind']
        base = ['colour', 'specular', 'lightmap (multiply, UV1)', 'lightmap (add, UV1)']
        if k in ('MT00', 'MUF0'):
            return base + ['colour B (UV2)', 'blend mask (UV1)', 'specular B (UV2)']
        if k == 'MTA1':
            return base + ['cloud layer 2 (UV2)']
        if k in ('LMC0', 'LMP0', 'LMB0'):
            return base + ['emissive (UV2)', 'cube map', 'cube map 2']
        if k in ('NEO0', 'NEC0'):
            return base + ['emissive', 'emissive mask (UV2)', 'cube map']
        if k == 'WTR0':
            return ['colour', 'specular', 'normal map A', 'normal map B']
        if k == 'BAR0':
            return ['barrier']
        return base + ['cube map', 'cube map 2', 't6']
    if cc[:2] in ('CS', 'SK', 'HR', 'EN'):
        return ['colour (top row = toon ramp)', 'normal map', 'wet', 'skin', 'underwear', 'height', 'break']
    return []


COMPANIONS = {'specular': ('_spc', '_sp', '_specular', '_spec'), 'normal map': ('_nml', '_normal', '_nrm', '_n')}


def companion_files(path, roles):
    """{slot: file} next to a colour texture file named like its specular / normal map (X_spc.png, X_nml.png ...)"""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    folder, base = os.path.split(path)
    stem, ext = os.path.splitext(base)
    files = {f.lower(): f for f in os.listdir(folder)}
    for k, role in enumerate(roles):
        for key, suffixes in COMPANIONS.items():
            if not role.startswith(key):
                continue
            for suf in suffixes:
                for e in (ext, '.png', '.dds', '.tga', '.jpg'):
                    f = files.get((stem + suf + e).lower())
                    if f:
                        out[k] = os.path.join(folder, f)
                        break
                if k in out:
                    break
    return out


def image_dds(img, mipmaps):
    """DDS bytes of a Blender image: a .dds file as it is, else its pixels encoded (DXT1, DXT5 with alpha)"""
    from . import ddsenc
    path = bpy.path.abspath(img.filepath) if img.filepath else ''
    if path.lower().endswith('.dds') and os.path.isfile(path):
        b = open(path, 'rb').read()
        if b[:4] == b'DDS ':
            return b
    w, h = img.size
    if not w or not h:
        raise C.ExportError('image %s has no pixels (not loaded?)' % img.name)
    px = np.zeros(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    rgba = np.clip(np.rint(px.reshape(h, w, 4)[::-1] * 255.0), 0, 255).astype(np.uint8)
    return ddsenc.encode(rgba, mipmaps)


def collect_new_materials(objs, model_name, template_model, textures):
    """the New game materials used by objs for this model: (new_materials [dict(base, slots)], {material name:
    index}, new_textures {name: dict(dds, like)}, report lines). textures: {name: (pack, index)} of the catalogue"""
    from . import ddsenc
    names_of = {F.tamsoft_hash(n): n for n in textures}
    mats = []
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            if m is not None and m.get('tmd_new') and m.get('tmd_model') == model_name and m not in mats:
                mats.append(m)
    n0 = len(template_model.materials)
    new_materials, index, new_textures, lines = [], {}, {}, []
    for j, m in enumerate(mats):
        base = int(m.get('tmd_base_index', m.get('tmd_index', 0)))
        if not 0 <= base < n0:
            raise C.ExportError('material %s: base material %d not in %s' % (m.name, base, model_name))
        base_slots = template_model.materials[base]['slots']
        slots = {}
        for k, tname in dict(m.get('tmd_slot_textures', {})).items():
            k = int(k)
            if k >= len(base_slots):
                continue
            tname = str(tname)
            if tname not in textures and tname not in new_textures:
                img = bpy.data.images.get(dict(m.get('tmd_slot_images', {})).get(str(k), ''))
                if img is None:
                    raise C.ExportError('material %s: the image for texture %s is missing' % (m.name, tname))
                like = names_of.get(base_slots[k]['hash'])
                if like is None:
                    raise C.ExportError('material %s: slot %d of the base material has no texture in this catalogue' % (m.name, k))
                pk = textures[like]
                new_textures[tname] = dict(dds=image_dds(img, ddsenc.has_mipmaps(pk[0].files[pk[1]])), like=like,
                                           model=model_name)
            slots[k] = tname
        new_materials.append(dict(base=base, slots=slots))
        index[m.name] = n0 + j
        lines.append('%s: new material %d (%s, copy of %d)' % (model_name, n0 + j, m.name, base))
    return new_materials, index, new_textures, lines


def collect_channels(objs, model_name, new_index, lines):
    """{material index: [ch0 .. chN]} from the game materials of objs (their 'tmd_params', edited in the Material
    tab); new game materials by their export index. Two Blender materials for one game material: the first wins"""
    out, src = {}, {}
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            if m is None or m.get('tmd_model') != model_name or m.get('tmd_params') is None:
                continue
            mi = new_index.get(m.name) if m.get('tmd_new') else m.get('tmd_index')
            if mi is None or mi < 0:
                continue
            vals = [float(x) for x in m['tmd_params']]
            if mi in out:
                if src[mi] != m.name and any(abs(a - b) > 1e-6 for a, b in zip(out[mi], vals)):
                    lines.append('%s: material %d has different channels in %s and %s: %s used'
                                 % (model_name, mi, src[mi], m.name, src[mi]))
                continue
            out[mi], src[mi] = vals, m.name
    return out


def check_paths(template_path, out_path, gamedata_root=''):
    if not os.path.isfile(template_path):
        raise C.ExportError('template catalogue not found: %s' % template_path)
    a, b = _norm(template_path), _norm(out_path)
    if a == b:
        raise C.ExportError('the output would overwrite the original catalogue; choose another file')
    if gamedata_root and _inside(b, _norm(gamedata_root)):
        raise C.ExportError('the output is inside the GameData folder; export elsewhere and copy it in yourself')


def _dds_folder(root, dds_dir):
    dds = {}
    if dds_dir:
        _, textures = catio.leaves(root)
        for f in sorted(os.listdir(dds_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() == '.dds' and stem in textures:
                dds[stem] = open(os.path.join(dds_dir, f), 'rb').read()
    return dds


def _write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    F.parse_cat(data)


def export(arm, template_path, out_path, apply_modifiers=True, keep_missing=True, dds_dir='', gamedata_root='',
           twins=True):
    """write the edited catalogue; returns report lines"""
    check_paths(template_path, out_path, gamedata_root)
    data = open(template_path, 'rb').read()
    root, _ = catio.read(data)
    models, _ = C.model_leaves(root)
    _, textures = catio.leaves(root)
    new_mats, new_tex, chans = {}, {}, {}
    skipped = []
    objs = character_objects(arm, skipped)
    edits, lines = {}, []
    if skipped:
        kinds = {}
        for n, why in skipped:
            kinds.setdefault(why, []).append(n)
        for why, names in sorted(kinds.items()):
            if why != 'outline hull':
                lines.append('not exported (%s): %s%s' % (why, ', '.join(names[:6]), ' ...' if len(names) > 6 else ''))
    for name, (parent, i) in models.items():
        mine = [o for o in objs if o.data.tmd_material.model == name]
        if not mine:
            continue
        model = F.Tmd2(parent.children[i].data, name)
        if model.weights is None:
            raise C.ExportError('%s has no skin weights (a rigid accessory / weapon): not supported by the export yet' % name)
        S, missing = skin_matrices(model, arm) if model.bones else (None, 0)
        if missing:
            lines.append('%s: %d bones of the model are not in the armature (identity used)' % (name, missing))
        nm_list, new_index, nt, nl = collect_new_materials(mine, name, model, textures)
        new_mats[name] = nm_list
        new_tex.update(nt)
        lines += nl
        chans[name] = collect_channels(mine, name, new_index, lines)
        mine.sort(key=lambda o: (o.name != '%s_%02d' % (name, o.data.tmd_material.mesh_index), o.name))
        meshes = []
        for o in mine:
            m, warn = mesh_from_object(o, arm, model, S, apply_modifiers, new_index)
            src = model.tags[m['source']] if model.tags and m['source'] < len(model.tags) else None
            if m['tags'] == src or (not m['tags'] and not src):
                m['tags'] = None
            meshes.append(m)
            lines += warn
        edits[name] = meshes
    if not edits:
        raise C.ExportError('no mesh of this character comes from %s (models: %s)' % (os.path.basename(template_path), ', '.join(models)))
    out, rep = C.export_catalogue(data, edits, keep_missing, _dds_folder(root, dds_dir), None, new_mats, new_tex,
                                  chans, twins)
    _write(out_path, out)
    return rep + lines


def stage_objects(model_names, skipped=None):
    """the scene's mesh objects of these models (TMD data, Export on, no outline / physics helper)"""
    out = []
    for o in bpy.context.scene.objects:
        if o.type != 'MESH':
            continue
        t = getattr(o.data, 'tmd_material', None)
        if t is None or t.model not in model_names:
            continue
        if _is_outline(o) or o.get('tmd_physics'):
            continue
        if not getattr(t, 'export', True):
            if skipped is not None:
                skipped.append((o.name, 'Export off'))
            continue
        out.append(o)
    return out


def touch_objects(entry_names):
    """{collision entry name: object} of the imported collision meshes (Import stage > Collision meshes)"""
    out = {}
    for o in bpy.context.scene.objects:
        if o.type != 'MESH' or not o.get('tmd_touch'):
            continue
        name = o.get('tmd_entry') or o.name.split('.')[0]
        if name in entry_names and name not in out:
            out[name] = o
    return out


def touch_from_object(obj, template_bytes, apply_modifiers=True):
    """the collision entry rebuilt from the object: its faces (triangulated as Blender draws them) in game space,
    each in the part of its 'tmd_part_id' face attribute (parts keep the template's ids; unknown -> part 0)"""
    from . import touchio
    T = touchio.Touch(template_bytes)
    me, cleanup = _evaluated_mesh(obj, apply_modifiers) if apply_modifiers else (obj.data, lambda: None)
    try:
        me.calc_loop_triangles()
        nv = len(me.vertices)
        co = np.zeros(nv * 3, np.float32)
        me.vertices.foreach_get('co', co)
        M = np.array(obj.matrix_world)
        co = co.reshape(nv, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        verts = _to_game(co)
        nt = len(me.loop_triangles)
        tv = np.zeros(nt * 3, np.int32)
        me.loop_triangles.foreach_get('vertices', tv)
        tp = np.zeros(nt, np.int32)
        me.loop_triangles.foreach_get('polygon_index', tp)
        part = np.zeros(len(me.polygons), np.int32)
        at = me.attributes.get('tmd_part')
        if at is not None and at.domain == 'FACE':
            at.data.foreach_get('value', part)
            part_of = [int(part[p]) for p in tp]
        else:
            at = me.attributes.get('tmd_part_id')
            if at is not None and at.domain == 'FACE':
                at.data.foreach_get('value', part)
            lut = {pid: k for k, pid in reversed(list(enumerate(idx for idx, _ in T.parts)))}
            part_of = [lut.get(int(part[p]), 0) for p in tp]
        tris = tv.reshape(-1, 3)
        if np.linalg.det(M[:3, :3]) < 0:
            tris = tris[:, ::-1]
        T.set_polygons(verts, tris.tolist(), part_of)
        return T.bytes(), nt
    finally:
        cleanup()


def export_stage(template_path, out_path, apply_modifiers=True, keep_missing=True, dds_dir='', gamedata_root='',
                 collision=True, touch_template='', touch_out='', twins=True):
    """write an edited stage catalogue (models, collision entries, textures) and optionally the edited copy of the
    collision catalogue the game loads (Binary/Touch/bgNN_tch.cat); returns report lines"""
    check_paths(template_path, out_path, gamedata_root)
    data = open(template_path, 'rb').read()
    root, _ = catio.read(data)
    models, _ = C.model_leaves(root)
    named, textures = catio.leaves(root)
    new_mats, new_tex, chans = {}, {}, {}
    skipped, lines, edits = [], [], {}
    for name, (parent, i) in models.items():
        objs = stage_objects({name}, skipped)
        if not objs:
            continue
        model = F.Tmd2(parent.children[i].data, name)
        arm = next((_armature_of(o) for o in objs if _armature_of(o) is not None), None)
        S = skin_matrices(model, arm)[0] if (model.weights is not None and model.bones and arm is not None) else None
        nm_list, new_index, nt, nl = collect_new_materials(objs, name, model, textures)
        new_mats[name] = nm_list
        new_tex.update(nt)
        lines += nl
        chans[name] = collect_channels(objs, name, new_index, lines)
        objs.sort(key=lambda o: (o.name != '%s_%02d' % (name, o.data.tmd_material.mesh_index), o.name))
        meshes = []
        for o in objs:
            m, warn = mesh_from_object(o, arm, model, S, apply_modifiers, new_index)
            m['tags'] = None if (m['tags'] or '') == (model.tags[m['source']] if model.tags and m['source'] < len(model.tags) else '') else m['tags']
            meshes.append(m)
            lines += warn
        edits[name] = meshes
    touches = {}
    if collision:
        entries = [n for n in named if F.touch_kind(n)]
        for name, o in touch_objects(set(entries)).items():
            p, i = named[name]
            touches[name], nt = touch_from_object(o, p.children[i].data, apply_modifiers)
            lines.append('%s: %d triangles from %s' % (name, nt, o.name))
    if not edits and not touches:
        raise C.ExportError('no object of the scene comes from %s (models: %s)' % (os.path.basename(template_path), ', '.join(models)))
    out, rep = C.export_catalogue(data, edits, keep_missing, _dds_folder(root, dds_dir), touches, new_mats, new_tex,
                                  chans, twins)
    _write(out_path, out)
    lines = rep + lines
    if touch_template and (touches or collision):
        check_paths(touch_template, touch_out, gamedata_root)
        td = open(touch_template, 'rb').read()
        troot, ttail = catio.read(td)
        done = C.replace_leaves(troot, touches)
        tout = catio.write(troot, ttail)
        _write(touch_out, tout)
        lines.append('%s: %s (the collision the game loads; rename it to %s when copying it into the game)'
                     % (os.path.basename(touch_out), ', '.join(done) + ' rebuilt' if done else 'unchanged copy',
                        os.path.basename(touch_template)))
    elif touches:
        lines.append('collision written into the stage catalogue only: the game loads it from Binary/Touch/bgNN_tch.cat')
    for n, why in skipped:
        lines.append('not exported (%s): %s' % (why, n))
    return lines


STATES = ('NTRL', 'BRK0', 'BRK1', 'BRK2', 'BRK3', 'BRKS')


def split_tags(tags):
    """(states, bust 'BOTH' | 'BAKU' | 'HIN', other tokens) of a tag string"""
    tok = tags.split()
    states = [t for t in tok if t in STATES]
    bust = 'BAKU' if 'BAKU' in tok else 'HIN' if 'HIN' in tok else 'BOTH'
    other = [t for t in tok if t not in STATES and t not in ('BAKU', 'HIN')]
    return states, bust, other


def join_tags(states, bust, other):
    tok = [s for s in STATES if s in states]
    tok += [t for t in other if t]
    if bust in ('BAKU', 'HIN'):
        tok.append(bust)
    return ' '.join(tok)


def _armature_of(obj):
    for m in obj.modifiers:
        if m.type == 'ARMATURE' and m.object is not None:
            return m.object
    p = obj.parent
    while p is not None and p.type != 'ARMATURE':
        p = p.parent
    return p


def _common_color(me):
    attr = me.color_attributes.get('Col') if hasattr(me, 'color_attributes') else None
    if attr is None or not len(attr.data):
        return None
    a = np.zeros(len(attr.data) * 4, np.float32)
    attr.data.foreach_get('color', a)
    a = np.round(a.reshape(-1, 4), 4)
    vals, cnt = np.unique(a, axis=0, return_counts=True)
    return tuple(float(x) for x in vals[np.argmax(cnt)])


def make_costume_mesh(obj, template, tags, materials='ADD', weights=True, color=True):
    """give obj the TMD data of template (an imported mesh): model, source mesh (exported as an extra mesh with
    these tags), materials, parent + Armature modifier, optionally bone weights (nearest face of the template, rest
    pose), the colour attribute and the UV maps the model uses. Returns report lines"""
    lines = []
    src, dst = template.data.tmd_material, obj.data.tmd_material
    for p in ('model', 'kind', 'mesh_index', 'xform_type', 'mesh_flags', 'vertex_flags'):
        setattr(dst, p, getattr(src, p))
    dst.bounds = src.bounds
    dst.tags = tags
    dst.hidden_reason = ''
    dst.materials.clear()
    for it in src.materials:
        n = dst.materials.add()
        for p in ('material', 'index', 'fourcc', 'num_params', 'params', 'textures'):
            setattr(n, p, getattr(it, p))
    if template.data.get('tmd_source'):
        obj.data['tmd_source'] = template.data['tmd_source']
    tmats = [s.material for s in template.material_slots if s.material is not None]
    if materials == 'REPLACE' or not any(s.material is not None and s.material.get('tmd_index') is not None
                                          for s in obj.material_slots):
        obj.data.materials.clear()
        for m in tmats:
            obj.data.materials.append(m)
        if len(obj.data.polygons):
            obj.data.polygons.foreach_set('material_index', np.zeros(len(obj.data.polygons), np.int32))
        lines.append('%s: materials of %s (all faces on %s)' % (obj.name, template.name, tmats[0].name if tmats else '-'))
    elif materials == 'ADD':
        have = {s.material for s in obj.material_slots}
        for m in tmats:
            if m not in have:
                obj.data.materials.append(m)
    for name in [l.name for l in template.data.uv_layers]:
        if obj.data.uv_layers.get(name) is None:
            base = obj.data.uv_layers.get('UVMap') or (obj.data.uv_layers[0] if len(obj.data.uv_layers) else None)
            lay = obj.data.uv_layers.new(name=name)
            if base is not None and lay is not None:
                a = np.zeros(len(base.data) * 2, np.float32)
                base.data.foreach_get('uv', a)
                lay.data.foreach_set('uv', a)
            lines.append('%s: UV map %s added (a copy of %s)' % (obj.name, name, base.name if base else 'nothing'))
    if len(obj.data.uv_layers) and obj.data.uv_layers.get('UVMap') is None:
        obj.data.uv_layers[0].name = 'UVMap'
    if color and hasattr(obj.data, 'color_attributes') and obj.data.color_attributes.get('Col') is None:
        c = _common_color(template.data)
        if c is not None:
            at = obj.data.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
            at.data.foreach_set('color', np.tile(np.array(c, np.float32), len(obj.data.vertices)))
            lines.append('%s: colour attribute Col = %s (the template\'s most common vertex colour; red = highlight mask)'
                         % (obj.name, ', '.join('%.3f' % x for x in c)))
    arm = _armature_of(template)
    if arm is not None:
        if obj.parent != arm:
            mw = obj.matrix_world.copy()
            obj.parent = arm
            obj.matrix_parent_inverse = arm.matrix_world.inverted()
            obj.matrix_world = mw
        if not any(m.type == 'ARMATURE' and m.object == arm for m in obj.modifiers):
            mod = obj.modifiers.new('Armature', 'ARMATURE')
            mod.object = arm
    if weights and arm is not None:
        lines += transfer_weights(obj, template, arm)
    return lines


def transfer_weights(obj, template, arm):
    """bone weights from the template's nearest face (interpolated), with the armature in its rest position"""
    view = bpy.context.view_layer
    prev_active, prev_sel = view.objects.active, [o for o in bpy.context.selected_objects]
    prev_pose = arm.data.pose_position
    hidden = template.hide_get()
    try:
        arm.data.pose_position = 'REST'
        template.hide_set(False)
        for o in prev_sel:
            o.select_set(False)
        template.select_set(True)
        obj.select_set(True)
        view.objects.active = template
        bones = {b.name for b in arm.data.bones}
        for g in [g for g in obj.vertex_groups if g.name in bones]:
            obj.vertex_groups.remove(g)
        bpy.ops.object.data_transfer(data_type='VGROUP_WEIGHTS', use_create=True, vert_mapping='POLYINTERP_NEAREST',
                                     layers_select_src='ALL', layers_select_dst='NAME', mix_mode='REPLACE')
        n = len([g for g in obj.vertex_groups if g.name in bones])
        return ['%s: weights of %d bones transferred from %s' % (obj.name, n, template.name)]
    except Exception as e:
        return ['%s: weights not transferred (%s); paint them or use a Data Transfer modifier' % (obj.name, e)]
    finally:
        arm.data.pose_position = prev_pose
        template.hide_set(hidden)
        template.select_set(False)
        obj.select_set(False)
        for o in prev_sel:
            if o.name in view.objects:
                o.select_set(True)
        view.objects.active = prev_active
