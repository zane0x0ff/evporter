import numpy as np
from . import formats as F

LAYER_STATES = ('NTRL', 'BRK0', 'BRK1', 'BRK2', 'BRK3', 'BRKS')


NO_OUTLINE = ('CSBB', 'CSWB', 'CSW0', 'CSTB', 'CSHB', 'CSHT')


def _remap(cc):
    return cc[:3] + '0' if cc[3] == '2' else cc[:3] + '1' if cc[3] == '3' else cc


def toon_kind(fourcc, model_name):
    """material FourCC -> game shader family (viewer Renderer.cpp / web render.js materialInfo):
    COSTUME CSB0 & co, SKIN SKC0, FACE SKC0/SKF0 on the face model, HAIR HRH0/HRL0, BREAK CSBB/CSWB, PLAIN otherwise"""
    cc = _remap(fourcc)
    if cc in ('SKC0', 'SKF0'):
        return 'FACE' if 'face' in model_name else 'SKIN'
    if cc in ('HRH0', 'HRL0'):
        return 'HAIR'
    if cc in ('CSBB', 'CSWB'):
        return 'BREAK'
    if cc.startswith('CS') or cc == 'EN00':
        return 'COSTUME'
    return 'PLAIN'


def stage_info(fourcc, model_name, has_uv2=False):
    """stage / field material (Model/Field bgNN.cat, viewer Renderer.cpp): None for the other families, else
    dict(kind = the ported program (stage.py), blend = drawn alpha blended, blend_state = 0 alpha / 1 additive /
    2 subtractive / 6 (SRC_COLOR, ONE), cull = back faces culled, fog = 'FG' for the sky's own fog)"""
    raw = fourcc
    cc = _remap(raw)
    if cc[:3] == 'BDS':
        cc = 'BGS' + cc[3]
    elif cc[:3] == 'FRD':
        cc = 'FRR' + cc[3]
    p3, p2 = cc[:3], cc[:2]
    kind = None
    if cc == 'MT00':
        kind = 'MT00'
    elif p3 == 'BGN':
        kind = 'BGN0'
    elif p3 == 'WTR':
        kind = 'WTR0'
    elif p3 in ('NRS', 'NRF', 'ADS', 'ADT'):
        kind = 'NRS0'
    elif p3 == 'BGG':
        kind = 'BGG0'
    elif p3 == 'BGC':
        kind = 'BGC0'
    elif p3 == 'LMC':
        kind = 'LMC0'
    elif p3 == 'LMP':
        kind = 'LMP0'
    elif p3 == 'NEC':
        kind = 'NEC0'
    elif p2 == 'WM':
        kind = 'WM20'
    elif p3 in ('NEO', 'NEB'):
        kind = 'NEO0'
    elif p3 == 'BAR':
        kind = 'BAR0'
    elif p3 == 'LMB':
        kind = 'LMB0'
    elif p3 == 'MTA':
        kind = 'MTA1'
    elif p3 == 'VFG':
        kind = 'VFG0'
    elif p3 == 'UFG':
        kind = 'UFG0'
    elif p3 == 'MUF':
        kind = 'MUF0'
    elif p3 == 'FRR':
        kind = 'FRR0' if has_uv2 else 'FRR0_0649'
    elif p2 in ('BG', 'FG') or p3 in ('SUN', 'BAD'):
        kind = 'BGS0'
    if kind is None:
        return None
    blend = raw[3] in '23' or kind == 'NEO0'
    state, cull = 0, False
    if p3 == 'BAR':
        blend, state = True, 2 if cc == 'BAR1' else 1
    if p3 in ('BAD', 'ADS', 'ADT'):
        blend, state = True, 1
    if p3 in ('NEB', 'LMB'):
        blend, state = True, 6
    if model_name.endswith('_a'):
        blend = True
    if kind == 'WTR0' or cc == 'NRSC':
        blend, cull = True, True
    if p3 == 'SUN':
        blend, cull = True, True
    return dict(kind=kind, blend=blend, blend_state=state, cull=cull, fog='FG' if p2 == 'FG' else '')


def bust_variant(chara, side):
    """table 0x1404ca880: 0 = small bust (BAKU meshes hidden) for pl08 and pl18's everyday costume, else 1 (HIN hidden)"""
    return 0 if chara == 8 or (chara == 18 and side == 0) else 1


def layer_hidden(model, ls):
    """costume layer pass (sub_1400e70e0) and the face / hair tags NNJ / UNI / ORIG / DIFF -> per mesh hidden + reason"""
    n = len(model.meshes)
    out = [None] * n
    if not model.tags:
        return out
    for i, s in enumerate(model.tags):
        show = True
        if 'ORIG' in s:
            show = ls['original']
        elif 'DIFF' in s:
            show = not ls['original']
        if 'NNJ' in s and ls['state_side'] != 1:
            show = False
        elif 'UNI' in s and ls['state_side'] != 0:
            show = False
        if not show:
            out[i] = 'costume tag'
    for i, s in enumerate(model.tags):
        if not s or not any(t in s for t in ('NTRL', 'BRK', 'HIN', 'BAKU', 'ACT', 'EVT')):
            continue
        if ('HIN' if ls['big_bust'] else 'BAKU') in s:
            out[i] = 'bust variant'
        elif ls['state'] not in s:
            out[i] = 'layer ' + ls['state']
        elif ('ACT' if ls['act'] else 'EVT') in s:
            out[i] = None
        elif ('EVT' if ls['act'] else 'ACT') in s:
            out[i] = 'scene tag'
        else:
            out[i] = None
    for i, s in enumerate(model.tags):
        if 'CHG1' in s:
            out[i] = 'transformation form'
    return out


class Plan:
    def __init__(self):
        self.bones = []
        self.bone_by_name = {}
        self.objects = []
        self.materials = {}
        self.notes = []
        self.name = ''
        self.head_bone = None
        self.head_point = None
        self.chara = self.costume_id = self.hair_id = None

    def add_bone(self, name, h, parent, head, model, offset=None, adjust=None):
        """head: the rest world matrix (4x4) or position; offset / adjust: the bone's tmd2 offset and the
        plbodyscl adjustment, so motions can be applied as the game does (L = adjust * SRT, + offset)"""
        base, k = name, 1
        while name in self.bone_by_name:
            name = '%s.%s' % (base, model) if k == 1 else '%s.%s.%d' % (base, model, k)
            k += 1
        head = np.asarray(head, float)
        rot = head[:3, :3] if head.ndim == 2 else np.eye(3)
        b = dict(name=name, hash=h, parent=parent, head=(head[3, :3] if head.ndim == 2 else head[:3]).copy(),
                 rot=rot.copy(), model=model, offset=None if offset is None else np.asarray(offset, float),
                 adjust=adjust)
        self.bones.append(b)
        self.bone_by_name[name] = b
        return name


class Character:
    def __init__(self, gd, chara):
        self.gd, self.chara = gd, chara
        cos = gd.read('Binary/Param/Player/cosinfo.bin')
        if not cos:
            raise ValueError('Binary/Param/Player/cosinfo.bin not found below ' + gd.root)
        self.cos = CosInfoCache.get(cos)
        if chara >= self.cos.nchara:
            raise ValueError('cosinfo.bin has %d characters' % self.cos.nchara)
        self.rec = self.cos.chars[chara]
        self.models = {}
        self.notes = []

    def model(self, path, want=''):
        key = (path, want)
        if key not in self.models:
            data = self.gd.read(path)
            m = None
            if data:
                try:
                    m = F.model_from_cat(data, want)
                    m.source_path = path
                except Exception as e:
                    self.notes.append('%s: %s' % (path, e))
            else:
                self.notes.append('missing ' + path)
            self.models[key] = m
        return self.models[key]

    def build(self, costume_id, side=None, layer='NTRL', accessories=True, weapons=False, proportions=True,
              hidden_meshes=False, face_overlays=False):
        """costume_id: global costume id (0..172). side: 0 everyday / 1 shinobi look (hair, underwear, accessories,
        NNJ / UNI tags); default = the side whose cosinfo costume is costume_id, else everyday."""
        gd, ix, rec = self.gd, self.chara, self.rec
        self.notes = []
        self.models = {k: v for k, v in self.models.items() if v is not None}
        if side is None:
            side = 1 if rec['costume'][1] == costume_id and rec['costume'][0] != costume_id else 0
        flags = self.cos.costume(costume_id)
        plan = Plan()
        plan.notes = self.notes
        plan.costume_flags = flags
        fb = ('Model/Costume/Uniform/%02d_00_H.cat' % costume_id if costume_id < 29 else
              'Model/Costume/Ninja/%02d_00_H.cat' % (costume_id - 29) if costume_id < 74 else
              'Model/Costume/FreeCos/%02d_00_H.cat' % (costume_id - 74))
        body_path = gd.asset_path(0x3CD + costume_id, fb)
        body = self.model(body_path)
        if body is None:
            raise ValueError('cannot load the costume model ' + body_path)
        plan.name = 'pl%02d_%s' % (ix, body.name)
        state = layer
        if flags['no_damage'] and layer in ('BRK0', 'BRK1', 'BRK2'):
            state = 'NTRL'
            self.notes.append('costume %d has no damage stages BRK0-BRK2 (cosinfo flag): NTRL used' % costume_id)
        ls = dict(state=state, big_bust=bust_variant(ix, side) != 0, act=True, state_side=side, original=True)

        face = self.model('Model/Playable/pl%02d_00/pl%02d_00_H.cat' % (ix, ix))
        hair_id = rec['hair'][side] if rec['hair'][side] is not None else ix
        hair = self.model('Model/Playable/hr%02d/hr%02d_H.cat' % (hair_id, hair_id))
        pants, bura = None, None
        u = rec['underwear'][side]
        if u is not None:
            if not flags['no_underwear']:
                swim = 0xB61 + u if u < 120 else 0x1C72 + u - 120
                pants = self.model(gd.asset_path(swim, 'Model/Costume/Swim/%02d_%02d_H.cat' % (u // 20, u % 20)))
            btype = flags['bra']
            bra = 0xA99 + btype * 20 + u % 20 if u < 120 else 0x1C22 + btype * 20 + (u - 120) % 20
            bdata = gd.read(gd.asset_path(bra, 'Model/Costume/Bura/%02d_%02d_H.cat' % (btype, u % 20)))
            bura = F.cat_textures(bdata, force_group=1) if bdata else None

        adjust = None
        if proportions:
            sc = F.body_scale(gd.read('Binary/Param/Player/plbodyscl.bin'), ix)
            if sc:
                adjust = [sc.get(b['hash']) for b in body.bones]
        W_body = body.bind_world(adjust)
        body_names = []
        for i, b in enumerate(body.bones):
            parent = body_names[b['parent']] if b['parent'] >= 0 else None
            body_names.append(plan.add_bone(F.bone_name(b['hash']), b['hash'], parent, W_body[i], body.name,
                                            offset=b['offset'], adjust=adjust[i] if adjust is not None else None))

        body_tex = F.resolve_textures(body.textures, [body.textures, face and face.textures, hair and hair.textures,
                                                       pants and pants.textures, bura])
        self._mesh_objects(plan, body, body_tex, body_names, W_body, layer_hidden(body, ls), hidden_meshes, 'body')

        head = body.find_bone(F.tamsoft_hash('head'))
        if head >= 0:
            plan.head_bone = body_names[head]
            plan.head_point = (np.array([0.0, 0.162, -0.05, 1.0]) @ W_body[head])[:3]
        tights = next((t for t in body_tex if t['hash'] == F.tamsoft_hash('tights00_00_EE')), None)
        for part, bone, kind in ((face, head, 'face'), (hair, head, 'hair'), (pants, 0, 'pants')):
            if part is None:
                continue
            names, W = self._attach(plan, part, body, body_names, W_body, max(bone, 0), share=True)
            tex = F.resolve_textures(part.textures, [part.textures, face and face.textures, hair and hair.textures])
            if kind == 'pants' and tights:
                tex = [dict(tights, hash=t['hash'], name=t['name']) if t['hash'] == F.tamsoft_hash('swim_blend00_EE') else t for t in tex]
            hid = layer_hidden(part, ls)
            if kind == 'face' and not face_overlays:
                for k in range(2, min(7, len(part.meshes))):
                    hid[k] = hid[k] or 'face expression overlay'
            self._mesh_objects(plan, part, tex, names, W, hid, hidden_meshes, kind)

        if accessories:
            self._accessories(plan, body, body_names, W_body, side)
        if weapons:
            self._weapons(plan, body, body_names, W_body, face, hair)
        plan.side = side
        plan.layer = state
        plan.chara = ix
        plan.costume_id = costume_id
        plan.hair_id = hair_id
        return plan

    def _attach(self, plan, part, body, body_names, W_body, parent_bone, share):
        """bones of an attached model: shared names follow the body bone; others hang off parent_bone"""
        W, names = [], []
        for i, b in enumerate(part.bones):
            j = body.find_bone(b['hash']) if share else -1
            if j >= 0:
                W.append(W_body[j]); names.append(body_names[j]); continue
            L = np.eye(4); L[3, :3] = b['offset']
            W.append(L @ (W[b['parent']] if b['parent'] >= 0 else W_body[parent_bone]))
            parent = names[b['parent']] if b['parent'] >= 0 else body_names[parent_bone]
            names.append(plan.add_bone(F.bone_name(b['hash']), b['hash'], parent, W[-1], part.name, offset=b['offset']))
        return names, W

    def _accessories(self, plan, body, body_names, W_body, side):
        gd, rec = self.gd, self.rec
        offs = gd.read('Binary/Param/Player/accoffset.bin')
        for k, aid in enumerate(rec['acc']):
            aux = rec['acc_aux'][k]
            if aid is None or (aux == 0 and side != 0) or (aux == 1 and side != 1):
                continue
            ref = F.accessory_ref(aid)
            if not ref:
                continue
            record, asset, stem = ref
            m = self.model(gd.asset_path(asset, 'Model/Acc/%s.cat' % stem))
            if m is None:
                continue
            off, rot, bone = F.acc_offset(offs, record)
            bi = body.find_bone(F.tamsoft_hash(bone))
            if bi < 0:
                self.notes.append('accessory %s: the body has no bone %s' % (stem, bone)); continue
            d2r = np.pi / 180
            local = F.euler_xyz(rot[0] * d2r, rot[1] * d2r, rot[2] * d2r)
            local[3, :3] = off
            attach = local @ W_body[bi]
            if m.bones:
                W, names = [], []
                for i, b in enumerate(m.bones):
                    L = np.eye(4); L[3, :3] = b['offset']
                    W.append(L @ (W[b['parent']] if b['parent'] >= 0 else attach))
                    parent = names[b['parent']] if b['parent'] >= 0 else body_names[bi]
                    names.append(plan.add_bone(F.bone_name(b['hash']), b['hash'], parent, W[-1], m.name, offset=b['offset']))
                self._mesh_objects(plan, m, m.textures, names, W, [None] * len(m.meshes), False, 'accessory')
            else:
                self._mesh_objects(plan, m, m.textures, [], [], [None] * len(m.meshes), False, 'accessory',
                                   rigid=(attach, body_names[bi]))

    def _weapons(self, plan, body, body_names, W_body, face, hair):
        gd, ix = self.gd, self.chara
        cat = 'Model/Playable/wpn%02d/wpn%02d_H.cat' % (ix, ix)
        for k, bone, hidden in F.weapon_slots(gd.read('Binary/Param/Player/weaponinfo.bin'), ix):
            m = self.model(cat, 'pl%02d_weapon%02d' % (ix, k))
            bi = body.find_bone(F.tamsoft_hash(bone))
            if m is None or bi < 0:
                continue
            names, W = self._attach(plan, m, body, body_names, W_body, bi, share=False)
            tex = F.resolve_textures(m.textures, [m.textures, face and face.textures, hair and hair.textures])
            hid = [('weaponinfo hidden mesh' if i in hidden else None) for i in range(len(m.meshes))]
            self._mesh_objects(plan, m, tex, names, W, hid, False, 'weapon')

    def _mesh_objects(self, plan, m, textures, bone_names, W, hidden, keep_hidden, kind, rigid=None):
        """one object per tmd2 mesh node: vertices skinned to the rest pose (invBind * W), weights to plan bones"""
        S = None
        if m.bones and len(m.inv_bind):
            S = np.stack([m.inv_bind[i] @ W[i] if i < len(m.inv_bind) else W[i] for i in range(len(m.bones))])
        for mi, mesh in enumerate(m.meshes):
            why = hidden[mi] if mi < len(hidden) else None
            if why and not keep_hidden:
                continue
            draws, mat, pal = [], -1, None
            for t, a in mesh['commands']:
                if t in (2, 5, 6):
                    mat = a
                elif t == 4 and m.skinned and a < len(m.skin_batches):
                    pal = a
                elif t == 3 and a < len(m.groups) and len(m.groups[a]):
                    draws.append((mat, a, pal))
            if not draws:
                continue
            mats = []
            for mat, g, pal in draws:
                if mat not in mats:
                    mats.append(mat)
            if all(pal is None for _, _, pal in draws):
                tris = [np.asarray(m.groups[g]) for _, g, _ in draws]
                allv = np.concatenate([t.ravel() for t in tris])
                verts, inv = np.unique(allv, return_inverse=True)
                faces = inv.reshape(-1, 3)
                fmat = np.concatenate([np.full(len(t), mats.index(mat)) for (mat, _, _), t in zip(draws, tris)])
                pals = [None] * len(verts)
            else:
                keymap, verts, pals, faces, fmat = {}, [], [], [], []
                for mat, g, pal in draws:
                    tri = m.groups[g]
                    out = np.empty_like(tri)
                    for idx in range(tri.size):
                        v = int(tri.flat[idx])
                        key = (v, pal)
                        j = keymap.get(key)
                        if j is None:
                            j = keymap[key] = len(verts)
                            verts.append(v); pals.append(pal)
                        out.flat[idx] = j
                    faces.append(out)
                    fmat.append(np.full(len(tri), mats.index(mat)))
                verts = np.array(verts)
                faces = np.concatenate(faces)
                fmat = np.concatenate(fmat)
            n = len(verts)
            pos = m.pos[verts]
            nrm = m.normal[verts] if m.normal is not None else None
            groups = None
            if S is not None and m.weights is not None:
                w = m.weights[verts].copy()
                local = m.bone_idx[verts]
                gidx = np.zeros_like(local)
                if all(p is None for p in pals):
                    gidx[:] = local
                for k in ([] if all(p is None for p in pals) else range(n)):
                    p = pals[k]
                    if p is None:
                        gidx[k] = local[k]
                    else:
                        table = m.skin_batches[p]
                        gidx[k] = [table[x] if x < len(table) else 0 for x in local[k]]
                gidx[gidx >= len(m.bones)] = 0
                ws = w.sum(1, keepdims=True)
                w = np.where(ws > 0, w / np.where(ws > 0, ws, 1), np.array([1.0, 0, 0, 0]))
                M = np.einsum('nk,nkij->nij', w, S[gidx])
                pos = np.einsum('ni,nij->nj', np.c_[pos, np.ones(n)], M)[:, :3]
                if nrm is not None:
                    nrm = np.einsum('ni,nij->nj', nrm, M[:, :3, :3])
                groups = (bone_names, gidx, w)
            elif rigid is not None:
                attach, bone = rigid
                Mm = m.mesh_matrix(mi)
                T = (Mm @ attach) if Mm is not None else attach
                pos = (np.c_[pos, np.ones(n)] @ T)[:, :3]
                if nrm is not None:
                    nrm = nrm @ T[:3, :3]
                groups = ([bone], np.zeros((n, 4), int), np.tile([1.0, 0, 0, 0], (n, 1)))
            else:
                Mm = m.mesh_matrix(mi)
                if Mm is not None:
                    pos = (np.c_[pos, np.ones(n)] @ Mm)[:, :3]
                    if nrm is not None:
                        nrm = nrm @ Mm[:3, :3]
            if nrm is not None:
                l = np.linalg.norm(nrm, axis=1, keepdims=True)
                nrm = nrm / np.where(l > 1e-12, l, 1)
            keys = []
            for mat in mats:
                key = '%s:%d' % (m.name, mat)
                if mat >= 0 and key not in plan.materials:
                    plan.materials[key] = self._material(m, mat, textures)
                keys.append(key if mat >= 0 else None)
            plan.objects.append(dict(
                name='%s_%02d' % (m.name, mi), model=m.name, kind=kind, mesh_index=mi, tags=m.tags[mi] if m.tags else '',
                hidden=why, xform_type=mesh['xform_type'], flags=mesh['flags'], bounds=mesh['bounds'],
                positions=pos, normals=nrm,
                frame=None if m.tangent is None or m.binormal is None or m.normal is None else
                (m.tangent[verts], m.binormal[verts], m.normal[verts]),
                uvs=[uv[verts] if uv is not None else None for uv in m.uv],
                color=m.color[verts] if m.color is not None else None, faces=faces, face_material=fmat,
                materials=keys, groups=groups, vertex_flags=m.flags, source=getattr(m, 'source_path', ''),
                zero_normals=(np.linalg.norm(m.normal[verts], axis=1) < 0.1) if m.normal is not None else None))

    def _material(self, m, mi, textures):
        mat = m.materials[mi]
        slots = []
        for s, sl in enumerate(mat['slots']):
            t = next((x for x in textures if x['hash'] == sl['hash']), None)
            if t is None:
                t = next((x for x in m.textures if x['hash'] == sl['hash']), None)
            slots.append(dict(slot=s, kind=sl['kind'], hash=sl['hash'], name=t['name'] if t else 'tex_%08x' % sl['hash'],
                              dds=t['dds'] if t else None))
        shader = toon_kind(mat['fourcc'], m.name)
        stage = stage_info(mat['fourcc'], m.name, len(m.uv) > 2 and m.uv[2] is not None)
        if stage is not None:
            shader = 'STAGE' if m.uv[0] is not None and m.normal is not None else 'PLAIN'
        elif m.uv[0] is None or m.normal is None or m.tangent is None:
            shader = 'PLAIN'
        outline = None
        if shader not in ('PLAIN', 'STAGE') and slots and slots[0]['dds'] is not None and _remap(mat['fourcc']) not in NO_OUTLINE:
            n = slots[0]['name']
            n = n[:-3] + '_outline_EE' if n.endswith('_EE') else n + '_outline'
            t = next((x for x in list(textures) + list(m.textures) if x['name'] == n), None)
            if t is not None:
                outline = dict(name=t['name'], dds=t['dds'])
        return dict(name='%s_%02d_%s' % (m.name, mi, mat['fourcc']), model=m.name, index=mi, fourcc=mat['fourcc'],
                    params=m.material_params(mi), slots=slots, shader=shader, outline=outline,
                    blend=stage['blend'] if stage else mat['fourcc'][3] in '23', stage=stage,
                    cube_slots=[x['slot'] for x in slots if F.dds_is_cube(x['dds'])],
                    has_uv=[u is not None for u in m.uv], param_keys=m.material_keys(mi))


class CosInfoCache:
    _last = (None, None)

    @classmethod
    def get(cls, data):
        if cls._last[0] is not data:
            cls._last = (data, F.CosInfo(data))
        return cls._last[1]


def scene_plan(data, model_name, entries=None):
    """Plan for a standalone model of a catalogue (a scene prop: the tree of plbg_die01, the tower + drum + grunt of
    plbg_die46): its own skeleton in bind pose, its textures from the same catalogue"""
    entries = entries or F.parse_cat(data)
    m = F.model_from_cat(data, model_name)
    c = Character.__new__(Character)
    c.notes, c.models = [], {}
    plan = Plan()
    plan.name = model_name
    plan.notes = c.notes
    plan.layer = 'NTRL'
    W = m.bind_world() if m.bones else []
    names = []
    for i, b in enumerate(m.bones):
        parent = names[b['parent']] if b['parent'] >= 0 else None
        names.append(plan.add_bone(F.bone_name(b['hash']), b['hash'], parent, W[i], m.name, offset=b['offset']))
    c._mesh_objects(plan, m, m.textures, names, W, [None] * len(m.meshes), False, 'prop')
    return plan
