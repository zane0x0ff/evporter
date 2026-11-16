import math

from .toon import Graph, _render_settings, _p, globals_group

GAMMA = 2.2
HALF_PI_INV = 1.0 / math.pi


class _B:
    """a stage material being built: texture / UV / geometry inputs as sockets"""

    def __init__(self, g, spec, images, cubes):
        self.g, self.spec, self.images, self.cubes = g, spec, images, cubes
        self.P = spec['params']
        has = spec.get('has_uv') or [True, True, True]
        self.uvs = {}
        for k, name in enumerate(('UVMap', 'UV1', 'UV2')):
            if k < len(has) and has[k]:
                n = g.node('ShaderNodeUVMap', label='UV%d' % k)
                n.uv_map = name
                self.uvs[k] = n.outputs['UV']
            else:
                self.uvs[k] = g.xyz(0.0, 1.0, 0.0, 'UV%d (missing: D3D 0, 0)' % k)
        geo = g.node('ShaderNodeNewGeometry')
        self.N, self.V, self.pos = geo.outputs['Normal'], geo.outputs['Incoming'], geo.outputs['Position']
        vc = g.node('ShaderNodeVertexColor', label='vertex colour = baked light')
        vc.layer_name = 'Col'
        self.vc, self.va = vc.outputs['Color'], vc.outputs['Alpha']
        self._d = None
        self._R = None

    def ch(self, k):
        return _p(self.P, k)

    def uv(self, k, du=0.0, dv=0.0, scale=1.0):
        """UV set k offset by (du, dv) in D3D terms (v down): Blender v = 1 - v_d3d, so the v offset flips"""
        s = self.uvs[k]
        if du or dv:
            s = self.g.vec('ADD', s, self.g.xyz(du, -dv, 0.0))
        if scale != 1.0:
            s = self.g.vec('SCALE', s, scale=scale)
        return s

    def tex(self, slot, uv, default=(0.0, 0.0, 0.0, 1.0), label=''):
        """(colour, alpha) of texture slot at uv; the default when the slot is empty"""
        img = self.images.get(slot)
        if img is None or slot in self.spec.get('cube_slots', ()):
            return self.g.xyz(*default[:3], 't%d missing' % slot), default[3]
        n = self.g.node('ShaderNodeTexImage', label='t%d %s: %s' % (slot, label, img.name))
        n.image = img
        n.interpolation = 'Linear'
        n.extension = 'REPEAT'
        self.g.set(n.inputs['Vector'], uv)
        return n.outputs['Color'], n.outputs['Alpha']

    def d(self):
        """V.N (V toward the eye)"""
        if self._d is None:
            self._d = self.g.vec('DOT_PRODUCT', self.V, self.N)
        return self._d

    def facing(self, e_ch=0, s_ch=1):
        """|V.N|^ch0 * ch1"""
        g = self.g
        a = g.math('MAXIMUM', g.math('ABSOLUTE', self.d()), 1e-8)
        return g.math('MULTIPLY', g.math('POWER', a, self.ch(e_ch)), self.ch(s_ch))

    def R(self):
        if self._R is None:
            g = self.g
            self._R = g.vec('SUBTRACT', g.vec('SCALE', self.N, scale=g.math('MULTIPLY', self.d(), 2.0)), self.V)
        return self._R

    def cube(self, k, label='cube'):
        """the k-th cube map of the material along R (latitude / longitude image, game axes) or black"""
        g = self.g
        if k >= len(self.cubes) or self.cubes[k] is None:
            return g.xyz(0.0, 0.0, 0.0, 'no cube map %d' % k)
        rx, ry, rz = g.split(g.vec('NORMALIZE', self.R()))
        gx, gy, gz = rx, rz, g.math('MULTIPLY', ry, -1.0)
        u = g.math('ADD', g.math('MULTIPLY', g.math('ARCTAN2', gx, gz), 0.5 / math.pi), 0.5)
        v = g.math('ADD', g.math('MULTIPLY', g.math('ARCSINE', g.math('MINIMUM', g.math('MAXIMUM', gy, -1.0), 1.0)),
                                 HALF_PI_INV), 0.5)
        n = g.node('ShaderNodeTexImage', label='%s: %s' % (label, self.cubes[k].name))
        n.image = self.cubes[k]
        n.interpolation = 'Linear'
        n.extension = 'REPEAT'
        g.set(n.inputs['Vector'], g.xyz(u, v, 0.0))
        return n.outputs['Color']

    def rgb(self, *v):
        return self.g.xyz(*v)

    def _fog_f(self, near, far, dist):
        """clamp((far - dist) / (far - near), 0, 1) with near / far in game units x 0.01"""
        g = self.g
        den = far * 0.01 - near * 0.01
        k = 1.0 / den if abs(den) > 1e-12 else 1e12
        return g.math('MULTIPLY', g.math('SUBTRACT', far * 0.01, dist), k, clamp=True)

    def distance(self):
        cam = self.g.node('ShaderNodeCameraData')
        return cam.outputs['View Distance']

    def material_fog(self, col, o, above):
        """VFG0 / UFG0 / MUF0 (viewer MaterialFog): height fog below (VFG0) / above (UFG0, MUF0) the plane ch[o],
        distance fog"""
        g, ch = self.g, self.ch
        if ch(o + 11) != 0.0:
            h = g.math('SUBTRACT', ch(o), g.split(self.pos)[2])
            cond = g.math('SUBTRACT', 1.0, g.math('GREATER_THAN', h, 0.0) if above else g.math('LESS_THAN', h, 0.0))
            f = self._fog_f(ch(o + 10), ch(o + 11), g.math('ABSOLUTE', h))
            fogged = g.lerp(self.rgb(ch(o + 5), ch(o + 6), ch(o + 7)), col, f)
            col = g.lerp(col, fogged, cond)
        if ch(o + 9) > 0.0:
            f = self._fog_f(ch(o + 8), ch(o + 9), self.distance())
            col = g.lerp(self.rgb(ch(o + 2), ch(o + 3), ch(o + 4)), col, f)
        return col

    def fg_fog(self, col):
        """FG0x: lerp(ch4..6, c, clamp((ch8 - dist) / (ch8 - ch7)))"""
        ch = self.ch
        if ch(8) == 0.0 and ch(7) == 0.0:
            return col
        f = self._fog_f(ch(7), ch(8), self.distance())
        return self.g.lerp(self.rgb(ch(4), ch(5), ch(6)), col, f)


def _lightmap(b, kind):
    g, ch = b.g, b.ch
    uv0 = b.uv(0, -ch(4), -ch(5)) if kind == 'NRS0' else b.uv(0)
    c, ca = b.tex(0, uv0, label='colour')
    keep = g.math('GREATER_THAN', ca, ch(3))
    base = g.vec('ADD', g.vec('MULTIPLY', c, b.tex(2, b.uv(1), (1, 1, 1, 1), 'multiMap')[0]),
                 b.tex(3, b.uv(1), (0, 0, 0, 1), 'addMap')[0])
    lit = g.vec('ADD', base, g.vec('SCALE', b.tex(1, uv0, label='speculer')[0], scale=b.facing()))
    col = g.vec('MULTIPLY', lit, b.vc)
    if kind in ('VFG0', 'UFG0'):
        col = b.material_fog(col, 5, kind == 'UFG0')
    elif b.spec['stage'].get('fog') == 'FG':
        col = b.fg_fog(col)
    return col, g.math('MULTIPLY', ca, b.va), keep


def _mt00(b, kind):
    g, ch = b.g, b.ch
    uva, uvb = b.uv(0, ch(4), -ch(5)), b.uv(2, ch(6), -ch(7))
    c, ca = b.tex(0, uva, label='colour A')
    keep = g.math('GREATER_THAN', ca, ch(3))
    mask = g.split(b.tex(5, b.uv(1), (1, 1, 1, 1), 'BlendMask')[0])[0]
    w = g.math('MULTIPLY', g.math('MAXIMUM', g.math('SUBTRACT', ca, g.math('SUBTRACT', 1.0, mask)), 0.0), b.va)
    diffuse = g.lerp(b.tex(4, uvb, label='colour B')[0], c, w)
    spec = g.lerp(b.tex(6, uvb, label='speculer B')[0], b.tex(1, uva, label='speculer A')[0], w)
    base = g.vec('ADD', g.vec('MULTIPLY', diffuse, b.tex(2, b.uv(1), (1, 1, 1, 1), 'multiMap')[0]),
                 b.tex(3, b.uv(1), (0, 0, 0, 1), 'addMap')[0])
    col = g.vec('MULTIPLY', g.vec('ADD', base, g.vec('SCALE', spec, scale=b.facing())), b.vc)
    if kind == 'MUF0':
        col = b.material_fog(col, 9, True)
    return col, g.math('MULTIPLY', ca, b.va), keep


def _mta1(b, kind):
    g, ch = b.g, b.ch
    uv0, uv2 = b.uv(0, ch(4), -ch(5)), b.uv(2, ch(6), -ch(7))
    c, ca = b.tex(0, uv0, label='colour')
    lit = g.vec('ADD', g.vec('ADD', g.vec('MULTIPLY', c, b.tex(2, b.uv(1), (1, 1, 1, 1), 'multiMap')[0]),
                                 b.tex(3, b.uv(1), (0, 0, 0, 1), 'addMap')[0]),
                g.vec('SCALE', b.tex(1, uv0, label='speculer')[0], scale=b.facing()))
    t4, t4a = b.tex(4, uv2, label='layer 2')
    col = g.vec('SCALE', g.vec('MULTIPLY', g.vec('ADD', t4, lit), b.vc), scale=0.5)
    a = g.math('MULTIPLY', g.math('MULTIPLY', g.math('ADD', t4a, ca), b.va), 0.5)
    return col, a, 1.0


def _bar0(b, kind):
    g, ch = b.g, b.ch
    c, ca = b.tex(0, b.uv(0, ch(1), -ch(2)), label='barrier')
    a = g.math('MULTIPLY', g.math('MULTIPLY', g.math('MULTIPLY', ca, ch(0)), b.va), 0.5)
    return g.vec('MULTIPLY', c, b.vc), a, 1.0


def _cube_family(b, kind):
    g, ch = b.g, b.ch
    lm = kind in ('LMC0', 'LMP0', 'NEC0', 'NEO0', 'LMB0')
    uv0 = b.uv(0, ch(7), -ch(8)) if lm else b.uv(0)
    uv2 = b.uv(2, ch(9), -ch(10))
    c, ca = b.tex(0, uv0, label='colour')
    keep = g.math('GREATER_THAN', ca, ch(3))
    base = g.vec('ADD', g.vec('MULTIPLY', c, b.tex(2, b.uv(1), (1, 1, 1, 1), 'multiMap')[0]),
                 b.tex(3, b.uv(1), (0, 0, 0, 1), 'addMap')[0])
    spec = b.tex(1, uv0, label='speculer')[0]
    if kind in ('FRR0', 'FRR0_0649'):
        cub = g.vec('SCALE', b.cube(0), scale=ch(2))
        env = g.vec('MULTIPLY', g.vec('SCALE', cub, scale=ch(5)),
                    g.vec('MAXIMUM', g.vec('SUBTRACT', spec, g.vec('SCALE', g.xyz(1.0, 1.0, 1.0), scale=g.math('MULTIPLY', b.d(), ch(4)))),
                          g.xyz(0.0, 0.0, 0.0)))
        base = g.vec('MULTIPLY', base, env) if kind == 'FRR0' else g.vec('ADD', base, g.vec('MULTIPLY', cub, env))
    elif kind not in ('LMP0', 'NEO0', 'LMB0'):
        base = g.vec('ADD', base, g.vec('MULTIPLY', g.vec('SCALE', b.cube(0), scale=ch(2)), spec))
    base = g.vec('ADD', base, g.vec('SCALE', spec, scale=b.facing()))
    if kind == 'LMB0':
        base = g.vec('ADD', base, g.vec('MULTIPLY', b.rgb(ch(4), ch(5), ch(6)), b.tex(4, uv2, label='emissive')[0]))
    col = g.vec('MULTIPLY', base, b.vc)
    a = g.math('MULTIPLY', ca, b.va)
    if kind == 'WM20':
        e = g.vec('SCALE', b.V, scale=b.distance())
        ex, ey, ez = g.split(e)
        l1 = g.math('ADD', g.math('ADD', g.math('ABSOLUTE', ex), g.math('ABSOLUTE', ey)), g.math('ABSOLUTE', ez))
        w3 = g.math('ADD', 0.2, g.math('MULTIPLY', l1, 0.008))
        dd = g.math('DIVIDE', g.math('MULTIPLY', b.d(), 1.0 / math.sqrt(2.0)),
                    g.math('SQRT', g.math('ADD', 1.0, g.math('MULTIPLY', w3, w3))))
        wet = g.vec('MAXIMUM', g.vec('SUBTRACT', b.tex(4, b.uv(1), (0, 0, 0, 1), 'puddle mask')[0],
                                     g.vec('SCALE', g.xyz(1.0, 1.0, 1.0), scale=g.math('MULTIPLY', dd, ch(5)))),
                    g.xyz(0.0, 0.0, 0.0))
        wr = g.split(wet)[0]
        col = g.vec('ADD', g.vec('SCALE', col, scale=g.math('SUBTRACT', 1.0, wr)),
                    g.vec('MULTIPLY', wet, g.vec('SCALE', b.cube(1, 'cube 2'), scale=ch(2))))
    if kind in ('LMC0', 'LMP0'):
        col = g.vec('ADD', col, g.vec('MULTIPLY', b.rgb(ch(4), ch(5), ch(6)), b.tex(4, uv2, label='emissive')[0]))
    if kind in ('NEC0', 'NEO0'):
        col = g.vec('ADD', col, g.vec('MULTIPLY', g.vec('MULTIPLY', b.tex(4, uv0, label='emissive')[0], b.rgb(ch(4), ch(5), ch(6))),
                                      b.tex(5, uv2, (1, 1, 1, 1), 'emissive mask')[0]))
    return col, a, keep


def _wtr0(b, kind):
    g, ch = b.g, b.ch
    uva, uvb = b.uv(0, -ch(4), -ch(5)), b.uv(0, -ch(6), -ch(7))

    def nmap(slot, uv, scale, label):
        t = b.tex(slot, g.vec('SCALE', uv, scale=scale) if scale != 1 else uv, (0.5, 0.5, 1.0, 1.0), label)[0]
        return g.vec('SUBTRACT', g.vec('SCALE', t, scale=2.0), g.xyz(1.0, 1.0, 1.0))

    n9 = g.vec('NORMALIZE', g.vec('ADD', nmap(2, uva, 9.0, 'normal A x9'), nmap(3, uvb, 9.0, 'normal B x9')))
    cam = g.node('ShaderNodeCameraData')
    vx, vy, vz = g.split(cam.outputs['View Vector'])
    V = g.vec('NORMALIZE', g.xyz(vx, vy, g.math('MULTIPLY', vz, -1.0)))
    refl = g.vec('SUBTRACT', V, g.vec('SCALE', n9, scale=g.math('MULTIPLY', g.vec('DOT_PRODUCT', V, n9), 2.0)))
    s = g.math('MINIMUM', g.math('MAXIMUM', g.vec('DOT_PRODUCT', refl, g.xyz(0.0, 0.447214, 0.894427)), 0.0), 1.0)
    glint = g.math('MULTIPLY', g.math('POWER', s, 50.0), g.math('SUBTRACT', 1.0, g.math('LESS_THAN', s, 0.85)))
    n1 = g.vec('ADD', nmap(3, uvb, 1.0, 'normal B'), nmap(2, uva, 1.0, 'normal A'))
    e = g.math('SUBTRACT', 1.0, g.math('DIVIDE', g.math('ABSOLUTE', g.split(n1)[1]),
                                       g.math('MAXIMUM', g.vec('LENGTH', n1), 1e-4)))
    f = g.math('POWER', g.math('MAXIMUM', e, 0.0), 5.0)
    vw = g.math('POWER', g.math('SUBTRACT', 1.0, g.math('MAXIMUM', g.split(b.V)[2], 0.0)), 5.0)
    tint = g.vec('MULTIPLY', g.vec('ADD', b.vc, g.xyz(1.0, 1.0, 1.0)), g.xyz(0.494, 0.541, 0.59))
    col = g.vec('ADD', g.vec('ADD', b.vc, g.vec('SCALE', tint, scale=g.math('MULTIPLY', f, g.math('MULTIPLY', vw, 0.36)))),
                g.vec('SCALE', g.xyz(1.0, 1.0, 1.0), scale=glint))
    return col, b.va, 1.0


PROGRAMS = {'BGS0': _lightmap, 'BGN0': _lightmap, 'BGG0': _lightmap, 'NRS0': _lightmap, 'VFG0': _lightmap,
            'UFG0': _lightmap, 'MT00': _mt00, 'MUF0': _mt00, 'MTA1': _mta1, 'BAR0': _bar0, 'WTR0': _wtr0}
for _k in ('BGC0', 'LMC0', 'LMP0', 'NEC0', 'NEO0', 'LMB0', 'WM20', 'FRR0', 'FRR0_0649'):
    PROGRAMS[_k] = _cube_family


def _output(g, col, alpha, state, gamma):
    """clamp, colour^gamma as Emission; state 0: mix with Transparent by alpha; 1 additive (c * a);
    6 (SRC_COLOR, ONE): + c * c; 2 subtractive, approximated as a multiply by 1 - c * a"""
    c = g.vec('MINIMUM', g.vec('MAXIMUM', col, g.xyz(0.0, 0.0, 0.0)), g.xyz(1.0, 1.0, 1.0))
    if state in (1, 2):
        c = g.vec('SCALE', c, scale=alpha)
    elif state == 6:
        c = g.vec('MULTIPLY', c, c)
    tr = g.node('ShaderNodeBsdfTransparent')
    if state == 2:
        g.set(tr.inputs['Color'], g.vec('SUBTRACT', g.xyz(1.0, 1.0, 1.0), c))
        return tr.outputs[0]
    gm = g.node('ShaderNodeGamma')
    g.set(gm.inputs['Color'], c)
    g.set(gm.inputs['Gamma'], gamma)
    em = g.node('ShaderNodeEmission')
    g.set(em.inputs['Color'], gm.outputs[0])
    em.inputs['Strength'].default_value = 1.0
    if state in (1, 6):
        add = g.node('ShaderNodeAddShader')
        g.set(add.inputs[0], tr.outputs[0])
        g.set(add.inputs[1], em.outputs[0])
        return add.outputs[0]
    mix = g.node('ShaderNodeMixShader')
    g.set(mix.inputs[0], alpha)
    g.set(mix.inputs[1], tr.outputs[0])
    g.set(mix.inputs[2], em.outputs[0])
    return mix.outputs[0]


def build_stage(mat, spec, images, cubes):
    """fill mat (use_nodes) with the stage tree for spec (character._material with spec['stage']);
    images: slot -> bpy image; cubes: the material's cube maps in slot order (latitude / longitude images)"""
    st = spec['stage']
    nt = mat.node_tree
    nt.nodes.clear()
    g = Graph(nt)
    glob = g.node('ShaderNodeGroup', label='TMD Globals')
    glob.node_tree = globals_group()
    b = _B(g, spec, images, cubes)
    col, alpha, keep = PROGRAMS[st['kind']](b, st['kind'])
    state = st['blend_state'] if st['blend'] else 0
    if not st['blend']:
        a = keep
    elif keep == 1.0:
        a = alpha
    else:
        a = g.math('MULTIPLY', keep, alpha)
    shader = _output(g, col, a, state, glob.outputs['Gamma'])
    out = g.node('ShaderNodeOutputMaterial')
    g.set(out.inputs['Surface'], shader)
    g.arrange(-1800)
    _render_settings(mat, st['blend'], cull=st['cull'], shadow=not st['blend'])
    mat['tmd_stage_kind'] = st['kind']
    mat['tmd_blend_state'] = state
    return mat
