import math
import bpy

VERSION = 1
E = math.e

LIGHT = (0.35, -0.76, 0.55)
FACE_LIGHT = (0.25, -0.88, 0.4)


class Graph:
    """small builder: nodes with values or sockets as inputs, a column layout by link depth at the end"""

    def __init__(self, nt):
        self.nt = nt

    def node(self, kind, label=None, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        if label:
            n.label = label
        return n

    def set(self, sock, v):
        if v is None:
            return
        if hasattr(v, 'is_output') or hasattr(v, 'links'):
            self.nt.links.new(v, sock)
            return
        dv = sock.default_value
        try:
            n = len(dv)
        except TypeError:
            n = 0
        if n:
            if isinstance(v, (int, float)):
                v = (float(v),) * min(n, 3) + ((1.0,) if n == 4 else ())
            elif len(v) < n:
                v = tuple(v) + (1.0,) * (n - len(v))
        sock.default_value = v

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.set(n.inputs[0], a)
        self.set(n.inputs[1], b)
        self.set(n.inputs[2], c)
        return n.outputs[0]

    def vec(self, op, a, b=None, scale=None):
        n = self.node('ShaderNodeVectorMath', operation=op)
        self.set(n.inputs[0], a)
        self.set(n.inputs[1], b)
        if scale is not None:
            self.set(n.inputs[3], scale)
        return n.outputs[1] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else n.outputs[0]

    def xyz(self, x, y, z, label=None):
        n = self.node('ShaderNodeCombineXYZ', label=label)
        for s, v in zip(n.inputs, (x, y, z)):
            self.set(s, v)
        return n.outputs[0]

    def split(self, v):
        n = self.node('ShaderNodeSeparateXYZ')
        self.set(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def lerp(self, a, b, f):
        return self.vec('ADD', a, self.vec('SCALE', self.vec('SUBTRACT', b, a), scale=f))

    def exp_log(self, lg, k):
        return self.math('EXPONENT', self.math('MULTIPLY', lg, k))

    def arrange(self, x0=0, dx=220, dy=170):
        nodes, links = list(self.nt.nodes), list(self.nt.links)
        depth = {n.name: 0 for n in nodes}
        for _ in range(len(nodes)):
            changed = False
            for l in links:
                d = depth[l.from_node.name] + 1
                if d > depth[l.to_node.name]:
                    depth[l.to_node.name] = d
                    changed = True
            if not changed:
                break
        rows = {}
        for n in nodes:
            d = depth[n.name]
            rows[d] = rows.get(d, 0) + 1
            n.location = (x0 + d * dx, -(rows[d] - 1) * dy)


def _group(name, build, keep=False):
    """the node group `name`, (re)built when missing or from another add-on version; keep: never rebuild an
    existing one (user settings)"""
    ng = bpy.data.node_groups.get(name)
    if ng is not None and (keep or ng.get('tmd_version') == VERSION):
        return ng
    if ng is None:
        ng = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    else:
        ng.nodes.clear()
        ng.interface.clear()
    build(ng)
    ng['tmd_version'] = VERSION
    return ng


def _io(ng, inputs, outputs):
    for name, kind, default in inputs:
        s = ng.interface.new_socket(name=name, in_out='INPUT', socket_type='NodeSocket' + kind)
        if default is not None:
            s.default_value = default
    for name, kind in outputs:
        ng.interface.new_socket(name=name, in_out='OUTPUT', socket_type='NodeSocket' + kind)
    g = Graph(ng)
    gi = g.node('NodeGroupInput')
    go = g.node('NodeGroupOutput')
    return g, gi.outputs, go.inputs


def globals_group():
    """shared settings: edit the values in this group (Tab into it); the EVPorter panel can aim the light at a Sun"""
    def build(ng):
        g, _, out = _io(ng, [], [('Light Dir', 'Vector'), ('Face Light Dir', 'Vector'), ('Rim Color', 'Vector'),
                                 ('Rim Power', 'Float'), ('Wetness', 'Float'), ('Hair Screen Scale', 'Float'),
                                 ('Gamma', 'Float')])
        light = g.node('ShaderNodeCombineXYZ', label='Light: direction toward the light (world)')
        light.name = 'tmd_light'
        for s, v in zip(light.inputs, LIGHT):
            s.default_value = v
        face = g.node('ShaderNodeCombineXYZ', label='Face light (SKF0 m_ToonLightDir, object space)')
        face.name = 'tmd_face_light'
        for s, v in zip(face.inputs, FACE_LIGHT):
            s.default_value = v
        g.set(out['Light Dir'], g.vec('NORMALIZE', light.outputs[0]))
        g.set(out['Face Light Dir'], g.vec('NORMALIZE', face.outputs[0]))
        g.set(out['Rim Color'], g.xyz(0.25, 0.25, 0.25, 'Rim colour (m_RimParam.rgb)'))
        for name, v, label in (('Rim Power', 4.0, 'Rim power (m_RimParam.w)'),
                               ('Wetness', 0.0, 'Wetness (m_CharaParameter.x)'),
                               ('Hair Screen Scale', 2.0, 'Hair ring: projection scale of x / y'),
                               ('Gamma', 2.2, 'Output gamma (Standard view transform)')):
            n = g.node('ShaderNodeValue', label=label)
            n.name = 'tmd_' + name.lower().replace(' ', '_')
            n.outputs[0].default_value = v
            g.set(out[name], n.outputs[0])
        g.arrange()
    return _group('TMD Globals', build, keep=True)


def ramp_group():
    """t = max((L.n + 1) / 2, 0) ^ exponent -> the ramp texel (t, top row) of the colour texture"""
    def build(ng):
        g, i, o = _io(ng, [('Normal', 'Vector', None), ('Light', 'Vector', None), ('Exponent', 'Float', 1.0),
                           ('Half Texel', 'Float', 0.0), ('Row V', 'Float', 1.0)],
                      [('Vector', 'Vector'), ('U', 'Float')])
        hl = g.math('MAXIMUM', g.math('MULTIPLY', g.math('ADD', g.vec('DOT_PRODUCT', g.vec('NORMALIZE', i['Light']),
                                                                       i['Normal']), 1.0), 0.5), 0.0)
        t = g.math('POWER', g.math('MAXIMUM', hl, 1e-8), i['Exponent'])
        u = g.math('MINIMUM', g.math('MAXIMUM', t, i['Half Texel']), g.math('SUBTRACT', 1.0, i['Half Texel']))
        g.set(o['Vector'], g.xyz(u, i['Row V'], 0.0))
        g.set(o['U'], t)
        g.arrange()
    return _group('TMD Ramp', build)


def layer_group():
    """CSW0 / CSWB tights: c.rgb * (1 - b.a) + b.rgb * c.a"""
    def build(ng):
        g, i, o = _io(ng, [('Color', 'Vector', None), ('Alpha', 'Float', 1.0), ('Layer Color', 'Vector', None),
                           ('Layer Alpha', 'Float', 0.0)], [('Color', 'Vector')])
        a = g.vec('SCALE', i['Color'], scale=g.math('SUBTRACT', 1.0, i['Layer Alpha']))
        g.set(o['Color'], g.vec('ADD', a, g.vec('SCALE', i['Layer Color'], scale=i['Alpha'])))
        g.arrange()
    return _group('TMD Layer Blend', build)


def _rim(g, i):
    geo = g.node('ShaderNodeNewGeometry')
    nv = g.math('ABSOLUTE', g.vec('DOT_PRODUCT', geo.outputs['Normal'], geo.outputs['Incoming']))
    rim = g.math('POWER', g.math('MAXIMUM', g.math('SUBTRACT', 1.0, nv), 1e-8), i['Rim Power'])
    return g.vec('SCALE', i['Rim Color'], scale=rim), geo


def _highlights(g, i, col, lg):
    """sheen: col * (1 - s) + s * sheen colour, s = exp(lg * ch13) * ch14 * mask; specular += exp(lg * ch9) * spec * mask"""
    sheen = g.math('MULTIPLY', g.math('MULTIPLY', g.exp_log(lg, i['Sheen Exponent']), i['Sheen Strength']), i['Mask'])
    col = g.vec('ADD', g.vec('SCALE', col, scale=g.math('SUBTRACT', 1.0, sheen)),
                g.vec('SCALE', i['Sheen Color'], scale=sheen))
    spec = g.math('MULTIPLY', g.exp_log(lg, i['Specular Exponent']), i['Mask'])
    return g.vec('ADD', col, g.vec('SCALE', i['Specular Color'], scale=spec))


HIGHLIGHT_INPUTS = [('Mask', 'Float', 0.0),
                    ('Specular Color', 'Vector', (0.0, 0.0, 0.0)), ('Specular Exponent', 'Float', 1.0),
                    ('Sheen Color', 'Vector', (0.0, 0.0, 0.0)), ('Sheen Exponent', 'Float', 1.0),
                    ('Sheen Strength', 'Float', 0.0)]
SHADE_INPUTS = [('Shade Strength', 'Float', 0.0), ('Shade Tint', 'Vector', (1.0, 1.0, 1.0))]
RIM_INPUTS = [('Rim Color', 'Vector', (0.25, 0.25, 0.25)), ('Rim Power', 'Float', 4.0)]


def toon_group():
    """CSB0 / SKC0 / SKF0 / CSBB body: rim, eye-angle sheen and specular, wet highlights, ramp shade"""
    def build(ng):
        g, i, o = _io(ng, [('Color', 'Vector', None), ('Ramp', 'Vector', None), ('Normal', 'Vector', None)]
                      + SHADE_INPUTS + [('Saturation', 'Float', 1.0), ('Light Exponent', 'Float', 1.0)]
                      + HIGHLIGHT_INPUTS
                      + [('Wet Strength', 'Float', 0.0), ('Wet Exponent', 'Float', 1.0), ('Costume Wet', 'Float', 0.0),
                         ('Wetness', 'Float', 0.0)] + RIM_INPUTS,
                      [('Color', 'Vector')])
        rim, geo = _rim(g, i)
        col = g.vec('ADD', i['Color'], rim)
        he = g.math('MAXIMUM', g.math('MULTIPLY', g.math('ADD', g.vec('DOT_PRODUCT', geo.outputs['Incoming'], i['Normal']),
                                                         1.0), 0.5), 1e-8)
        lg = g.math('MULTIPLY', g.math('LOGARITHM', he, E), i['Light Exponent'])
        col = _highlights(g, i, col, lg)
        w = g.math('MULTIPLY', i['Costume Wet'], i['Wetness'])
        col = g.vec('MULTIPLY', col, g.vec('ADD', g.xyz(1.0, 1.0, 1.0),
                                           g.vec('SCALE', g.xyz(-0.196078, -0.0980392, -0.0745098), scale=w)))
        col = g.vec('ADD', col, g.math('MULTIPLY', g.exp_log(lg, i['Wet Exponent']),
                                       g.math('MULTIPLY', i['Wetness'], i['Wet Strength'])))
        r = g.vec('MULTIPLY', i['Ramp'], i['Shade Tint'])
        lum = g.vec('DOT_PRODUCT', r, g.xyz(0.299, 0.587, 0.114))
        shade = g.lerp(g.xyz(lum, lum, lum), r, i['Saturation'])
        g.set(o['Color'], g.vec('SUBTRACT', col, g.vec('SCALE', shade, scale=i['Shade Strength'])))
        g.arrange()
    return _group('TMD Toon', build)


def hair_group():
    """HRH0 "angel ring": highlights from the screen-space direction head point -> pixel, then the ramp shade"""
    def build(ng):
        g, i, o = _io(ng, [('Color', 'Vector', None), ('Ramp', 'Vector', None),
                           ('Head Offset', 'Vector', (0.0, 0.0, 0.0)), ('Ring Offset', 'Float', 0.0),
                           ('Screen Scale', 'Float', 2.0)] + SHADE_INPUTS + HIGHLIGHT_INPUTS + RIM_INPUTS,
                      [('Color', 'Vector')])
        rim, geo = _rim(g, i)
        col = g.vec('ADD', i['Color'], rim)
        vt = g.node('ShaderNodeVectorTransform', vector_type='VECTOR', convert_from='WORLD', convert_to='CAMERA')
        g.set(vt.inputs[0], i['Head Offset'])
        x, y, z = g.split(vt.outputs[0])
        off = g.math('MULTIPLY', i['Ring Offset'], 0.01)
        dv = g.vec('NORMALIZE', g.xyz(g.math('MULTIPLY_ADD', x, i['Screen Scale'], off),
                                      g.math('MULTIPLY_ADD', y, i['Screen Scale'], off), z))
        _, dy, dz = g.split(dv)
        up = g.math('SUBTRACT', 1.0, dy)
        q = g.vec('NORMALIZE', g.xyz(g.math('POWER', up, 3.0), dz, 0.0))
        d = g.vec('DOT_PRODUCT', q, g.xyz(0.707107, -0.707107, 0.0))
        lg = g.math('LOGARITHM', g.math('MAXIMUM', d, 1e-30), E)
        col = _highlights(g, i, col, lg)
        r = g.vec('MULTIPLY', i['Ramp'], i['Shade Tint'])
        g.set(o['Color'], g.vec('SUBTRACT', col, g.vec('SCALE', r, scale=i['Shade Strength'])))
        g.arrange()
    return _group('TMD Toon Hair', build)


def output_group():
    """clamp to [0, 1] (8-bit target), colour^gamma as emission, alpha by a transparent mix"""
    def build(ng):
        g, i, o = _io(ng, [('Color', 'Vector', None), ('Alpha', 'Float', 1.0), ('Gamma', 'Float', 2.2)],
                      [('Shader', 'Shader')])
        c = g.vec('MINIMUM', g.vec('MAXIMUM', i['Color'], g.xyz(0.0, 0.0, 0.0)), g.xyz(1.0, 1.0, 1.0))
        gm = g.node('ShaderNodeGamma')
        g.set(gm.inputs['Color'], c)
        g.set(gm.inputs['Gamma'], i['Gamma'])
        em = g.node('ShaderNodeEmission')
        g.set(em.inputs['Color'], gm.outputs[0])
        em.inputs['Strength'].default_value = 1.0
        tr = g.node('ShaderNodeBsdfTransparent')
        mix = g.node('ShaderNodeMixShader')
        g.set(mix.inputs[0], i['Alpha'])
        g.set(mix.inputs[1], tr.outputs[0])
        g.set(mix.inputs[2], em.outputs[0])
        g.set(o['Shader'], mix.outputs[0])
        g.arrange()
    return _group('TMD Toon Output', build)


def _p(params, k):
    return float(params[k]) if k < len(params) else 0.0


def _render_settings(mat, blended, cull=False, shadow=True):
    mat.use_backface_culling = cull
    if hasattr(mat, 'surface_render_method'):
        mat.surface_render_method = 'BLENDED' if blended else 'DITHERED'
    if hasattr(mat, 'blend_method'):
        try:
            mat.blend_method = 'BLEND' if blended else 'CLIP'
            mat.alpha_threshold = 0.5
        except (TypeError, AttributeError):
            pass
    if not shadow and hasattr(mat, 'shadow_method'):
        try:
            mat.shadow_method = 'NONE'
        except (TypeError, AttributeError):
            pass


def _texture(g, img, label, vector=None, extension='REPEAT'):
    n = g.node('ShaderNodeTexImage', label=label)
    n.image = img
    n.interpolation = 'Linear'
    n.extension = extension
    if vector is not None:
        g.set(n.inputs['Vector'], vector)
    return n


BREAK_PROGRESS = 'tmd_ch8'


def _break_test(g, spec, images, alpha, color):
    """CSBB / CSWB transformation (docs 3.5.2): the outfit appears piece by piece as channel 8 grows.
    D3D: height = HeightTexture5((uv1.x, clamp(2 - uv1.y + ch8, 1/64, 63/64))).r (the vertex shader, mip 1),
         discard where BreakTexture6(uv1 - (0, 1)).r < height,
         colour += max((break - HeightTexture5((uv1.x, clamp(uv1.y - ch8, 1/128, 127/128))).r) * 4, 0) * ch26..28.
    Blender UV1 v = 1 - D3D v. CSWB keeps the height / break textures in slots 3 / 4."""
    params, cc = spec['params'], spec['fourcc']
    hs, bs = (3, 4) if cc.startswith('CSW') else (5, 6)
    himg, bimg = images.get(hs), images.get(bs)
    if himg is None or bimg is None:
        return alpha, color
    prog = g.node('ShaderNodeValue', label='ch8 break progress (keyed: material clock)')
    prog.name = BREAK_PROGRESS
    prog.outputs[0].default_value = _p(params, 8)
    ch8 = prog.outputs[0]
    uv = g.node('ShaderNodeUVMap', label='UV1: x, per-piece delay')
    uv.uv_map = 'UV1'
    u, v, _ = g.split(uv.outputs['UV'])
    y = g.math('ADD', g.math('ADD', v, 1.0), ch8)
    y = g.math('MINIMUM', g.math('MAXIMUM', y, 1.0 / 64.0), 1.0 - 1.0 / 64.0)
    ht = _texture(g, himg, 'height (threshold): ' + himg.name, g.xyz(u, g.math('SUBTRACT', 1.0, y), 0.0), 'EXTEND')
    height = g.split(ht.outputs['Color'])[0]
    bt = _texture(g, bimg, 'break noise: ' + bimg.name, uv.outputs['UV'], 'REPEAT')
    brk = g.split(bt.outputs['Color'])[0]
    keep = g.math('SUBTRACT', 1.0, g.math('LESS_THAN', brk, height))
    alpha = g.math('MULTIPLY', alpha, keep)
    if len(params) > 28:
        y2 = g.math('MINIMUM', g.math('MAXIMUM', g.math('SUBTRACT', g.math('SUBTRACT', 1.0, v), ch8), 1.0 / 128.0),
                    1.0 - 1.0 / 128.0)
        h2t = _texture(g, himg, 'height (edge)', g.xyz(u, g.math('SUBTRACT', 1.0, y2), 0.0), 'EXTEND')
        h2 = g.split(h2t.outputs['Color'])[0]
        glow = g.math('MAXIMUM', g.math('MULTIPLY', g.math('SUBTRACT', brk, h2), 4.0), 0.0)
        color = g.vec('ADD', color, g.vec('SCALE', g.xyz(_p(params, 26), _p(params, 27), _p(params, 28)), scale=glow))
    return alpha, color


def build_toon(mat, spec, images, ring=None):
    """fill mat (use_nodes) with the toon tree for spec (character.Character._material);
    images: slot -> bpy image (or None); ring: the head ring empty for HAIR"""
    shader, params, cc = spec['shader'], spec['params'], spec['fourcc']
    nt = mat.node_tree
    nt.nodes.clear()
    g = Graph(nt)
    glob = g.node('ShaderNodeGroup', label='TMD Globals')
    glob.node_tree = globals_group()
    G = glob.outputs

    c0 = images.get(0)
    tex = _texture(g, c0, 'colour: ' + (c0.name if c0 else '-'))
    color, alpha = tex.outputs['Color'], tex.outputs['Alpha']
    nrm = g.node('ShaderNodeNormalMap', space='TANGENT', uv_map='UVMap')
    n1 = images.get(1)
    if n1 is not None and '_nml' in n1.name.lower():
        t1 = _texture(g, n1, 'normal map: ' + n1.name)
        g.set(nrm.inputs['Color'], t1.outputs['Color'])
    normal = nrm.outputs['Normal']

    light, ramp_n = G['Light Dir'], normal
    if shader == 'FACE':
        lts = []
        for a in ('tmd_tangent', 'tmd_binormal', 'tmd_normal'):
            at = g.node('ShaderNodeAttribute', label='unskinned ' + a[4:])
            at.attribute_type = 'GEOMETRY'
            at.attribute_name = a
            lts.append(g.vec('DOT_PRODUCT', G['Face Light Dir'], at.outputs['Vector']))
        light = g.xyz(*lts)
        if n1 is not None and '_nml' in n1.name.lower():
            ramp_n = g.vec('NORMALIZE', g.vec('SUBTRACT', g.vec('SCALE', t1.outputs['Color'], scale=2.0), g.xyz(1.0, 1.0, 1.0)))
        else:
            ramp_n = g.xyz(0.0, 0.0, 1.0)
    rg = g.node('ShaderNodeGroup', label='ramp coordinate')
    rg.node_tree = ramp_group()
    g.set(rg.inputs['Normal'], ramp_n)
    g.set(rg.inputs['Light'], light)
    g.set(rg.inputs['Exponent'], _p(params, 6 if shader == 'BREAK' else 16))
    if c0 is not None:
        w, h = c0.size[0] or 1, c0.size[1] or 1
        g.set(rg.inputs['Half Texel'], 0.5 / w)
        g.set(rg.inputs['Row V'], 1.0 - 0.5 / h)
    rt = _texture(g, c0, 'ramp: top row of the colour texture', rg.outputs['Vector'], 'EXTEND')
    ramp, ramp_a = rt.outputs['Color'], rt.outputs['Alpha']

    if cc.startswith('CSW') and images.get(2) is not None:
        b = images[2]
        bt = _texture(g, b, 'blend layer: ' + b.name)
        lb = g.node('ShaderNodeGroup', label='blend layer')
        lb.node_tree = layer_group()
        for k, v in (('Color', color), ('Alpha', alpha), ('Layer Color', bt.outputs['Color']), ('Layer Alpha', bt.outputs['Alpha'])):
            g.set(lb.inputs[k], v)
        br = _texture(g, b, 'blend layer ramp row', g.xyz(rg.outputs['U'], 1.0, 0.0), 'EXTEND')
        rb = g.node('ShaderNodeGroup', label='blend layer (ramp)')
        rb.node_tree = layer_group()
        for k, v in (('Color', ramp), ('Alpha', ramp_a), ('Layer Color', br.outputs['Color']), ('Layer Alpha', br.outputs['Alpha'])):
            g.set(rb.inputs[k], v)
        color, ramp = lb.outputs['Color'], rb.outputs['Color']

    vc = g.node('ShaderNodeVertexColor', label='vertex colour R = highlight mask')
    vc.layer_name = 'Col'
    mask = g.split(vc.outputs['Color'])[0]

    tg = g.node('ShaderNodeGroup', label='%s (%s)' % (cc, shader.lower()))
    tg.node_tree = hair_group() if shader == 'HAIR' else toon_group()
    I = tg.inputs
    g.set(I['Color'], color)
    g.set(I['Ramp'], ramp)
    g.set(I['Shade Strength'], _p(params, 0))
    g.set(I['Shade Tint'], (_p(params, 1), _p(params, 2), _p(params, 3)))
    g.set(I['Rim Color'], G['Rim Color'])
    g.set(I['Rim Power'], G['Rim Power'])
    if shader != 'BREAK':
        g.set(I['Mask'], mask)
        g.set(I['Specular Color'], (_p(params, 6), _p(params, 7), _p(params, 8)))
        g.set(I['Specular Exponent'], _p(params, 9))
        g.set(I['Sheen Color'], (_p(params, 10), _p(params, 11), _p(params, 12)))
        g.set(I['Sheen Exponent'], _p(params, 13))
        g.set(I['Sheen Strength'], _p(params, 14))
    if shader == 'HAIR':
        g.set(I['Ring Offset'], _p(params, 17))
        g.set(I['Screen Scale'], G['Hair Screen Scale'])
        if ring is not None:
            tc = g.node('ShaderNodeTexCoord', label='position relative to the head point')
            tc.object = ring
            g.set(I['Head Offset'], tc.outputs['Object'])
    else:
        g.set(I['Normal'], normal)
        g.set(I['Light Exponent'], _p(params, 16))
        g.set(I['Saturation'], _p(params, 25 if shader == 'BREAK' else 20))
        g.set(I['Wetness'], G['Wetness'])
        if shader != 'BREAK':
            g.set(I['Wet Strength'], _p(params, 18))
            g.set(I['Wet Exponent'], _p(params, 19))
            g.set(I['Costume Wet'], 1.0 if shader == 'COSTUME' else 0.0)

    blended = spec.get('blend', False)
    if shader == 'HAIR':
        a = alpha if blended else 1.0
    else:
        keep = g.math('GREATER_THAN', alpha, 0.4999)
        a = g.math('MULTIPLY', keep, alpha) if blended else keep
    color_out = tg.outputs['Color']
    if shader == 'BREAK':
        a, color_out = _break_test(g, spec, images, a, color_out)
    og = g.node('ShaderNodeGroup', label='output')
    og.node_tree = output_group()
    g.set(og.inputs['Color'], color_out)
    g.set(og.inputs['Alpha'], a)
    g.set(og.inputs['Gamma'], G['Gamma'])
    out = g.node('ShaderNodeOutputMaterial')
    g.set(out.inputs['Surface'], og.outputs['Shader'])
    g.arrange(-1400)
    _render_settings(mat, blended)
    return mat


def build_outline(mat, spec):
    """the CSB0 0x467f outline hull: colour and alpha * 0.5 of the outline texture per vertex (corner attribute
    tmd_outline, written by build.py), drawn where ch15 < alpha, back faces of the inverted hull only"""
    nt = mat.node_tree
    nt.nodes.clear()
    g = Graph(nt)
    glob = g.node('ShaderNodeGroup', label='TMD Globals')
    glob.node_tree = globals_group()
    at = g.node('ShaderNodeAttribute', label='outline colour / alpha * 0.5')
    at.attribute_type = 'GEOMETRY'
    at.attribute_name = 'tmd_outline'
    geo = g.node('ShaderNodeNewGeometry')
    vis = g.math('MULTIPLY', g.math('GREATER_THAN', at.outputs['Alpha'], _p(spec['params'], 15)),
                 g.math('SUBTRACT', 1.0, geo.outputs['Backfacing']))
    og = g.node('ShaderNodeGroup', label='output')
    og.node_tree = output_group()
    g.set(og.inputs['Color'], at.outputs['Color'])
    g.set(og.inputs['Alpha'], vis)
    g.set(og.inputs['Gamma'], glob.outputs['Gamma'])
    out = g.node('ShaderNodeOutputMaterial')
    g.set(out.inputs['Surface'], og.outputs['Shader'])
    g.arrange(-800)
    _render_settings(mat, False, cull=True, shadow=False)
    return mat


def set_light(direction):
    """aim the shared toon light: direction = toward the light, Blender world space"""
    ng = globals_group()
    n = ng.nodes.get('tmd_light')
    if n is not None:
        for s, v in zip(n.inputs, direction):
            s.default_value = v


CHANNEL_NAMES = {
    'BODY': {0: 'shade strength', 1: 'shade tint R', 2: 'shade tint G', 3: 'shade tint B', 6: 'highlight R',
             7: 'highlight G', 8: 'highlight B', 9: 'highlight sharpness', 10: 'sheen R', 11: 'sheen G', 12: 'sheen B',
             13: 'sheen sharpness', 14: 'sheen strength', 15: 'outline alpha cut', 16: 'light exponent',
             18: 'wet strength', 19: 'wet sharpness', 20: 'saturation'},
    'HAIR': {0: 'shade strength', 1: 'shade tint R', 2: 'shade tint G', 3: 'shade tint B', 6: 'highlight R',
             7: 'highlight G', 8: 'highlight B', 9: 'highlight sharpness', 10: 'sheen R', 11: 'sheen G', 12: 'sheen B',
             13: 'sheen sharpness', 14: 'sheen strength', 15: 'outline alpha cut', 17: 'ring offset'},
    'BREAK': {0: 'shade strength', 1: 'shade tint R', 2: 'shade tint G', 3: 'shade tint B', 6: 'light exponent',
              8: 'transformation progress', 25: 'saturation', 26: 'burn edge R', 27: 'burn edge G', 28: 'burn edge B'},
}


def channel_names(mat):
    """{channel: meaning} for the material's shader family (empty for the stage shaders)"""
    sh = mat.get('tmd_shader', '')
    if sh == 'HAIR':
        return CHANNEL_NAMES['HAIR']
    if sh == 'BREAK':
        return CHANNEL_NAMES['BREAK']
    if sh in ('COSTUME', 'SKIN', 'FACE'):
        return CHANNEL_NAMES['BODY']
    return {}


def apply_channels(mat):
    """show the material's edited channels (mat['tmd_params']) in its preview: the toon trees' inputs are set in
    place, the stage trees are rebuilt from the spec stored at import. Returns a message when that is not possible"""
    params = [float(x) for x in mat.get('tmd_params', [])]
    if not params or mat.node_tree is None:
        return 'no game channels'
    nodes = mat.node_tree.nodes
    if mat.get('tmd_shader') == 'STAGE':
        js = mat.get('tmd_stage_spec')
        if not js:
            return 'imported with an older add-on: re-import the stage to preview channel edits (export works)'
        import json
        from . import stage
        spec = json.loads(js)
        spec['params'] = params
        spec['stage'] = dict(spec['stage'])
        images = {int(k): bpy.data.images.get(v) for k, v in dict(mat.get('tmd_images', {})).items()}
        images = {k: v for k, v in images.items() if v is not None}
        cubes = [images.get(k) for k in spec.get('cube_slots', ())]
        stage.build_stage(mat, spec, images, cubes)
        return None
    shader = mat.get('tmd_shader', '')
    brk = shader == 'BREAK'

    def put(node, name, value):
        sock = node.inputs.get(name)
        if sock is None or sock.is_linked:
            return
        if hasattr(sock.default_value, '__len__'):
            sock.default_value = tuple(value)[:len(sock.default_value)]
        else:
            sock.default_value = float(value)
    p = lambda k: _p(params, k)
    found = False
    for n in nodes:
        if n.type != 'GROUP' or n.node_tree is None:
            continue
        name = n.node_tree.name
        if name.startswith('TMD Ramp'):
            put(n, 'Exponent', p(6 if brk else 16))
        elif name.startswith('TMD Toon') and not name.startswith('TMD Toon Output'):
            found = True
            put(n, 'Shade Strength', p(0))
            put(n, 'Shade Tint', (p(1), p(2), p(3)))
            if not brk:
                put(n, 'Specular Color', (p(6), p(7), p(8)))
                put(n, 'Specular Exponent', p(9))
                put(n, 'Sheen Color', (p(10), p(11), p(12)))
                put(n, 'Sheen Exponent', p(13))
                put(n, 'Sheen Strength', p(14))
            if shader == 'HAIR':
                put(n, 'Ring Offset', p(17))
            else:
                put(n, 'Light Exponent', p(16))
                put(n, 'Saturation', p(25 if brk else 20))
                if not brk:
                    put(n, 'Wet Strength', p(18))
                    put(n, 'Wet Exponent', p(19))
    return None if found else 'no toon shader in this material (Principled preview): export still uses the channels'
