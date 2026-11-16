import math
import struct
import numpy as np

from . import formats as F

G = 9.80665
GAME_G = 3.92266
STEP = 1.0 / 60.0
SKIRT = ('mini', 'middle', 'long', 'slit', 'none')
ROLE = ('root', 'middle', 'tip', 'special')
ANCHOR_COLL = 19
BUST_COLL = 18


def parse_yure(data):
    """yure_*.bin -> dict(chains=[[joint]], links=[(chainA, jointA, chainB, jointB, type)]) or None when empty"""
    if not data or len(data) < 0x9c + 4 * 0x1b0:
        return None
    _, _, nj, nl, nc, ob, ol = struct.unpack_from('<IfIIIII', data, 0)
    if nj == 0 or nc > 32:
        return None
    levels = []
    for r in range(4):
        o = 0x9c + r * 0x1b0
        subs = []
        for k in range(4):
            so = o + 0xc + k * 0x68
            p0, p1, p2, p3 = struct.unpack_from('<4f', data, so)
            flags = tuple(data[so + 0x10:so + 0x14])
            stretch, swing = (0.0, 0.0), 180.0
            for l in range(3):
                kind, _, w, lo, hi = struct.unpack_from('<hhfff', data, so + 0x14 + l * 0x1c)
                if kind == 1:
                    stretch = (lo, hi)
                elif kind == 2:
                    swing = max(abs(lo), abs(hi))
            subs.append(dict(pull=p0, gravity=p1, keep=p2, spring=p3, flags=flags, stretch=stretch, swing=swing))
        levels.append(dict(type=struct.unpack_from('<h', data, o)[0], sub=subs, axis_free=tuple(data[o + 0x1ac:o + 0x1b0])))
    lengths = struct.unpack_from('<32i', data, 0x1c)[:nc]
    chains, j = [], 0
    for c in range(nc):
        n = lengths[c]
        if n <= 0 or j + n > nj:
            return None
        chain = []
        for k in range(n):
            h, root, tip, special, _, level = struct.unpack_from('<IBBBBh', data, ob + 12 * j)
            j += 1
            lv = levels[level if 0 <= level < 4 else 0]
            role = 3 if special else 0 if root else 2 if tip else 1
            chain.append(dict(hash=h, anchored=bool(root) or k == 0, role=ROLE[role], level=level,
                              type=lv['type'], group=lv['axis_free'][3], prm=lv['sub'][role]))
        chains.append(chain)
    links = []
    for i in range(nl):
        a, ja, b, jb, _, _, t = struct.unpack_from('<HHHHIHH', data, ol + 16 * i)
        if a != b and a < nc and b < nc and ja < len(chains[a]) and jb < len(chains[b]):
            links.append((a, ja, b, jb, t))
    return dict(chains=chains, links=links)


def parse_collision(data):
    if not data or len(data) < 4:
        return []
    n = struct.unpack_from('<I', data, 0)[0]
    out = []
    for i in range(n):
        if 4 + 0x2c * (i + 1) > len(data):
            break
        f = struct.unpack_from('<HHIIf3f3fI', data, 4 + 0x2c * i)
        out.append(dict(type=f[1], boneA=f[2], boneB=f[3], radius=f[4] * 0.01,
                        offA=np.array(f[5:8]) * 0.01, offB=np.array(f[8:11]) * 0.01, group=f[11]))
    return out


def skirt_group(data, costume_id):
    """category_list.bin (asset 0x1324, sub_14012a570 / sub_14012a9c0) -> group 0..4 = mini, middle, long, slit, none"""
    if not data:
        return None
    dflt, n = struct.unpack_from('<ii', data, 0)
    c = costume_id % 174
    want = (1 if c >= 29 else 0, c - 29 if c >= 29 else c)
    off = 0
    for _ in range(n):
        o = 8 + off
        idx, cnt, nxt = struct.unpack_from('<iii', data, o)
        for k in range(cnt):
            if struct.unpack_from('<ii', data, o + 12 + 8 * k) == want:
                return idx
        off = nxt
    return dflt


def bust_params(data, costume_id, flag=0):
    """BustInfo.bin (asset 0x1326) -> [R, L]: box (x0, x1, y0, y1, z0, z1) in metres (thorax frame), keep, spring"""
    if not data or len(data) < 4 + 0x78:
        return None
    count_a = struct.unpack_from('<h', data, 2)[0]
    idx = costume_id % 174
    ofs = 4
    if idx > 28:
        ofs += count_a * 0x78
        idx -= 29
    ofs += idx * 0x78
    if ofs + 0x78 > len(data):
        return None
    b = flag == 0
    out = []
    for side in range(2):
        f = struct.unpack_from('<15f', data, ofs + side * 0x3c)
        l = f[4:8] if b else f[0:4]
        z = 1.0 if b else 0.5
        out.append(dict(box=[l[0] * .01, l[1] * .01, l[2] * .01, l[3] * .01, f[8] * .01 * z, f[9] * .01 * z],
                        keep=f[12] if b else f[10], spring=f[13] if b else f[11]))
    return out


def physics_files(gd, costume_id, hair_id):
    """[(label, yure path, collision path)] the game loads for the character (sub_140118c30)"""
    c = costume_id
    out = []
    if c < 29:
        y = 0x1390 + c
        out.append(('costume', gd.asset_path(y, 'Binary/physics/Uniform/yure_unicos_%02d.bin' % c),
                    gd.asset_path(y + 0x1F5, 'Binary/physics/Uniform/collision_unicos_%02d.bin' % c)))
    else:
        y = 0x13c3 + c - 29
        sub = 'Ninja/yure_ninjacos_%02d.bin' % (c - 29) if c < 74 else 'Freecos/yure_freecos_%02d.bin' % (c - 74)
        out.append(('costume', gd.asset_path(y, 'Binary/physics/' + sub),
                    gd.asset_path(y + 0x1F5, 'Binary/physics/' + sub.replace('yure_', 'collision_'))))
    g = skirt_group(gd.read(gd.asset_path(0x1324, 'Binary/physics/category_list.bin')), c)
    if g is not None and 0 <= g < 5:
        out.append(('skirt (%s)' % SKIRT[g], gd.asset_path(0x1327 + g, 'Binary/physics/cate/yure_cate_%s.bin' % SKIRT[g]),
                    gd.asset_path(0x1327 + g + 0x1F5, 'Binary/physics/cate/collision_cate_%s.bin' % SKIRT[g])))
    if hair_id is not None:
        out.append(('hair', gd.asset_path(0x132c + hair_id, 'Binary/physics/hr/yure_hr_%02d.bin' % hair_id),
                    gd.asset_path(0x132c + hair_id + 0x1F5, 'Binary/physics/hr/collision_hr_%02d.bin' % hair_id)))
    return out


def seg_dist(p1, q1, p2, q2):
    """closest distance between segments p1-q1 and p2-q2 (Ericson 5.1.9)"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def frame_y(axis):
    """3x3 (columns = local x, y, z) with local Y along axis"""
    y = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, ref)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


class RigidRig:
    """bodies, constraints and bone drives in game space; see the header for the model"""

    def __init__(self):
        self.bodies = []
        self.constraints = []
        self.drives = []
        self.files = []
        self.notes = []
        self.gravity = G

    def body(self, **kw):
        kw.setdefault('mass', 1.0)
        kw.setdefault('damping', (0.5, 0.5))
        kw.setdefault('collections', ())
        kw.setdefault('parent_bone', None)
        self.bodies.append(kw)
        return kw['name']

    def constraint(self, **kw):
        kw.setdefault('lin', None)
        kw.setdefault('ang', None)
        kw.setdefault('lin_k', (0.0, 0.0, 0.0))
        kw.setdefault('ang_k', (0.0, 0.0, 0.0))
        kw.setdefault('damping', 0.5)
        kw.setdefault('frame', np.eye(3))
        self.constraints.append(kw)
        return kw


def _damping(keep):
    """the game keeps keep x the velocity per 1/60 s step; Bullet: v *= (1 - d)^dt -> d = 1 - keep^60"""
    return float(min(max(1.0 - max(keep, 0.0) ** 60, 0.0), 0.9999999))


def _sag_angle(j, length):
    """the angle a horizontal segment of this joint sags in the game's spring / gravity balance (radians)"""
    p = j['prm']
    g = GAME_G * p['gravity'] * p['pull']
    if j['type'] == 0:
        k = min(p['spring'], 1.0) ** 2
        if p['flags'][0] == 1:
            g *= 0.5
        e = g * STEP * STEP / k if k > 1e-9 else 1e9
    else:
        e = g / p['spring'] if p['spring'] > 1e-9 else 1e9
    return min(max(e / max(length, 1e-4), math.radians(0.5)), math.radians(60.0))


def build_rig(gd, plan, radius=0.02, mass=1.0, stiffness=1.0, bust=True, bust_flag=0):
    """the rigid body rig of an import plan (character.Character.build), game space"""
    rig = RigidRig()
    by_hash, bone = {}, {}
    for b in plan.bones:
        by_hash.setdefault(b['hash'], b['name'])
        bone[b['name']] = b
    chains, links, shapes = [], [], []
    for label, ypath, cpath in physics_files(gd, plan.costume_id, plan.hair_id):
        y = parse_yure(gd.read(ypath))
        cs = parse_collision(gd.read(cpath))
        used = 0
        if y:
            base = len(chains)
            for ch in y['chains']:
                for j in ch:
                    j['bone'] = by_hash.get(j['hash'])
                    used += j['bone'] is not None
                chains.append(ch)
            links += [(base + a, ja, base + b, jb, t) for a, ja, b, jb, t in y['links']]
        for s in cs:
            s['file'] = label
        shapes += cs
        rig.files.append('%s: %s (%s), %s (%d shapes)' % (label, ypath, '%d joints matched' % used if y else 'no chains',
                                                           cpath, len(cs)))
    groups = sorted({j['group'] for ch in chains for j in ch} | {s['group'] for s in shapes} | {0})
    allc = tuple(g for g in groups)

    def colls(g):
        return allc if g == 0 else (g,)

    shape_geo = []
    for i, s in enumerate(shapes):
        a_name, b_name = by_hash.get(s['boneA']), by_hash.get(s['boneB'])
        if a_name is None or s['type'] not in (0, 1):
            continue
        A = bone[a_name]
        a = A['head'] + s['offA'] @ A['rot']
        if s['type'] == 0 and b_name is not None:
            b = bone[b_name]['head'] + s['offB'] @ A['rot']
        else:
            b = a.copy()
        name = 'col_%s_%02d_%s' % (s['file'].split(' ')[0], i, a_name)
        rig.body(name=name, kind='shape', shape='CAPSULE' if s['type'] == 0 and np.linalg.norm(b - a) > 1e-4 else 'SPHERE',
                 a=a, b=b, radius=s['radius'], parent_bone=a_name, collections=colls(s['group']), group=s['group'],
                 label='%s shape %d, group %d' % (s['file'], i, s['group']))
        shape_geo.append((name, a, b, s['radius'], s['group']))

    dyn = []
    seg_body = {}
    for ci, ch in enumerate(chains):
        parent_body, parent_pivot_bone = None, None
        for k, j in enumerate(ch):
            if j['bone'] is None:
                parent_body = None
                continue
            p = j['bone']
            if k == 0 or j['anchored']:
                parent_body = None
                continue
            prev = ch[k - 1]
            if prev['bone'] is None:
                continue
            a, b = bone[prev['bone']]['head'], bone[p]['head']
            length = float(np.linalg.norm(b - a))
            if length < 1e-4:
                continue
            r = min(radius, 0.45 * length)
            prm = j['prm']
            d = _damping(prm['keep'])
            name = 'rb_%02d_%02d_%s' % (ci, k, prev['bone'])
            if parent_body is None:
                ref = bone[prev['bone']]['parent'] or prev['bone']
            rig.body(name=name, kind='dynamic', shape='SPHERE', a=b, b=b, radius=r, mass=mass, damping=(d, d),
                     parent_bone=ref, collections=colls(j['group']), group=j['group'],
                     label='chain %d joint %d (%s, %s, type %d)' % (ci, k, j['role'], 'level %d' % j['level'], j['type']))
            dyn.append((name, b, b, r, j['group'], ci, k))
            seg_body[(ci, k)] = name
            if parent_body is None:
                parent_body = rig.body(name='anchor_%02d_%02d_%s' % (ci, k, prev['bone']), kind='anchor', shape='SPHERE',
                                       a=a, b=a, radius=min(0.005, r), parent_bone=ref, collections=(ANCHOR_COLL,),
                                       group=-1, label='chain %d root anchor' % ci)
            theta = _sag_angle(j, length)
            ang_k = stiffness * mass * G * length * 0.5 / theta
            sw = math.radians(min(prm['swing'], 170.0))
            rig.constraint(name='joint_%02d_%02d_%s' % (ci, k, prev['bone']), type='GENERIC_SPRING', a=parent_body, b=name,
                           pivot=a, frame=frame_y(b - a),
                           lin=[(0.0, 0.0)] * 3, ang=[(-sw, sw), (0.0, 0.0), (-sw, sw)],
                           ang_k=(ang_k, ang_k * 0.5, ang_k), damping=0.5,
                           label='swing %.1f deg, sag %.1f deg' % (prm['swing'], math.degrees(theta)))
            if all(dr['bone'] != prev['bone'] for dr in rig.drives):
                rig.drives.append(dict(bone=prev['bone'], kind='DAMPED_TRACK', body=name))
            parent_body = name

    done = set()
    for a, ja, b, jb, t in links:
        A, B = seg_body.get((a, ja)), seg_body.get((b, jb))
        if not A or not B or A == B or (A, B) in done or (B, A) in done:
            continue
        done.add((A, B))
        pa, pb = bone[chains[a][ja]['bone']]['head'], bone[chains[b][jb]['bone']]['head']
        rest = float(np.linalg.norm(pb - pa))
        tr = 0.1 * rest
        rig.constraint(name='link_%s_%s' % (A[3:], B[3:]), type='GENERIC_SPRING', a=A, b=B, pivot=(pa + pb) * 0.5,
                       lin=[(-tr, tr)] * 3, lin_k=(stiffness * mass * 50.0,) * 3, damping=0.5,
                       label='cross-chain link, rest %.1f cm' % (rest * 100))

    constrained = {(c['a'], c['b']) for c in rig.constraints} | {(c['b'], c['a']) for c in rig.constraints}
    nc = 0
    for i in range(len(dyn)):
        n1, a1, b1, r1, g1 = dyn[i][:5]
        for j2 in range(i + 1, len(dyn)):
            n2, a2, b2, r2, g2 = dyn[j2][:5]
            if (n1, n2) in constrained or not (set(colls(g1)) & set(colls(g2))):
                continue
            if seg_dist(a1, b1, a2, b2) < (r1 + r2) * 1.5 + 0.002:
                rig.constraint(name='nocol_%s_%s' % (n1[3:], n2[3:]), type='GENERIC', a=n1, b=n2, pivot=(a1 + b2) * 0.5,
                               label='no collision (touching at rest)')
                nc += 1

    if bust:
        prm = bust_params(gd.read(gd.asset_path(0x1326, 'Binary/physics/BustInfo.bin')), plan.costume_id, bust_flag)
        thorax = by_hash.get(F.tamsoft_hash('thorax'))
        sides = [by_hash.get(F.tamsoft_hash('bust_R00')), by_hash.get(F.tamsoft_hash('bust_L00'))]
        if prm and thorax and all(sides):
            pR, pL = bone[sides[0]]['head'], bone[sides[1]]['head']
            r = min(0.04, 0.48 * float(np.linalg.norm(pR - pL)))
            for s, bn in enumerate(sides):
                p = prm[s]
                pos = bone[bn]['head']
                k = mass * p['spring'] / (STEP * STEP)
                d = _damping(p['keep'])
                name = 'rb_bust_%s' % ('R', 'L')[s]
                rig.body(name=name, kind='dynamic', shape='SPHERE', a=pos, b=pos, radius=r, mass=mass, damping=(d, d),
                         parent_bone=thorax, collections=(BUST_COLL,), group=-2, label='bust %s' % ('R', 'L')[s])
                anc = rig.body(name='anchor_bust_%s' % ('R', 'L')[s], kind='anchor', shape='SPHERE', a=pos, b=pos,
                               radius=0.005, parent_bone=thorax, collections=(ANCHOR_COLL,), group=-1, label='bust anchor')
                bx = p['box']
                ky = max(k, mass * G / 0.01)
                rig.constraint(name='joint_bust_%s' % ('R', 'L')[s], type='GENERIC_SPRING', a=anc, b=name, pivot=pos,
                               frame=(bone[thorax]['rot'] / np.linalg.norm(bone[thorax]['rot'], axis=1)[:, None]).T,
                               lin=[(bx[0], bx[1]), (bx[2], bx[3]), (bx[4], bx[5])], ang=[(0.0, 0.0)] * 3,
                               lin_k=(k, ky, k), damping=0.5,
                               label='bust: keep %.2f, spring %.3f' % (p['keep'], p['spring']))
                rig.drives.append(dict(bone=bn, kind='COPY_LOCATION', body=name))
        elif not prm:
            rig.notes.append('BustInfo.bin: no entry')
    rig.notes.append('%d bodies (%d dynamic), %d constraints (%d no-collision), %d driven bones' % (
        len(rig.bodies), sum(b['kind'] == 'dynamic' for b in rig.bodies), len(rig.constraints), nc, len(rig.drives)))
    return rig
