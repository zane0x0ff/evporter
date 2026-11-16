import os
import tempfile
import numpy as np
import bpy

from . import formats as F
from . import toon
from . import stage
from . import rigid


def to_blender(a):
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[..., 0], -a[..., 2], a[..., 1]], -1)


def image_from_dds(name, dds, cache_dir, pack):
    """the game texture as a Blender image: decoded here (formats.decode_dds), rows flipped to Blender's bottom-up
    order, then packed into the .blend or saved as PNG in the cache folder"""
    img = bpy.data.images.get(name)
    if img is not None and img.get('tmd_size') == len(dds):
        return img
    w, h, rgba = F.decode_dds(dds)
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set((rgba[::-1].astype(np.float32) / 255.0).ravel())
    img['tmd_size'] = len(dds)
    if pack:
        img.pack()
    else:
        os.makedirs(cache_dir, exist_ok=True)
        img.filepath_raw = os.path.join(cache_dir, name + '.png')
        img.file_format = 'PNG'
        img.save()
    return img


def is_data_texture(name, slot):
    n = name.lower()
    return slot != 0 or any(k in n for k in ('_nml', 'height', 'break', '_spc', '_spec', 'mask', 'outline'))


def _set_colorspace(img, name):
    try:
        img.colorspace_settings.name = name
    except TypeError:
        pass


def image_from_cube(name, dds, cache_dir, pack):
    """a DDS cube map as a latitude / longitude image in game space (formats.cube_to_equirect, read by stage.py)"""
    iname = name + '_latlong'
    img = bpy.data.images.get(iname)
    if img is not None and img.get('tmd_size') == len(dds):
        return img
    rgba = F.cube_to_equirect(F.decode_dds_cube(dds))
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(iname, w, h, alpha=True)
    img.pixels.foreach_set((rgba[::-1].astype(np.float32) / 255.0).ravel())
    img['tmd_size'] = len(dds)
    img['tmd_cube'] = True
    if pack:
        img.pack()
    else:
        os.makedirs(cache_dir, exist_ok=True)
        img.filepath_raw = os.path.join(cache_dir, iname + '.png')
        img.file_format = 'PNG'
        img.save()
    return img


def animate_channels(mat, spec):
    """key the material's animated channels that the node tree exposes as Value nodes named tmd_chN (CSBB / CSWB:
    ch8, the transformation progress). The keys run on the model's own material clock (60 fps), placed from the
    scene start; the game loops that clock (840 frames in ninjacos00_00), here the last value holds"""
    keys = spec.get('param_keys') or {}
    nodes = mat.node_tree.nodes
    todo = [(k, nodes.get('tmd_ch%d' % k)) for k in sorted(keys)]
    todo = [(k, n) for k, n in todo if n is not None]
    if not todo:
        return None
    from .anim import _container, _keys
    scene = bpy.context.scene
    scale = (scene.render.fps / scene.render.fps_base) / 60.0
    start = float(scene.frame_start)
    nt = mat.node_tree
    action = bpy.data.actions.new(mat.name + '_channels')
    new, slot = _container(action, nt, 'NODETREE')
    last = start
    for k, node in todo:
        fr, va = keys[k]
        _keys(new('nodes["%s"].outputs[0].default_value' % node.name, 0, 'material channels'), start + fr * scale, va)
        last = max(last, start + fr[-1] * scale)
    if nt.animation_data is None:
        nt.animation_data_create()
    nt.animation_data.action = action
    if slot is not None and hasattr(nt.animation_data, 'action_slot'):
        try:
            nt.animation_data.action_slot = slot
        except (AttributeError, TypeError, RuntimeError):
            pass
    mat['tmd_clock_end'] = last
    if scene.frame_end < last:
        scene.frame_end = int(np.ceil(last))
    return action


def load_images(spec, cache_dir, pack, raw):
    """slot -> bpy image for the material's texture slots; raw: every image as Non-Color (toon trees);
    cube maps (spec cube_slots) as latitude / longitude images"""
    images = {}
    cubes = set(spec.get('cube_slots', ()))
    for s in spec['slots']:
        if s['dds'] is None:
            continue
        try:
            if s['slot'] in cubes:
                img = image_from_cube(s['name'], s['dds'], cache_dir, pack)
                _set_colorspace(img, 'Non-Color' if raw else 'sRGB')
                images[s['slot']] = img
                continue
            img = image_from_dds(s['name'], s['dds'], cache_dir, pack)
        except Exception as e:
            print('evporter: %s: %s' % (s['name'], e))
            continue
        _set_colorspace(img, 'Non-Color' if raw or is_data_texture(s['name'], s['slot']) else 'sRGB')
        images[s['slot']] = img
    return images


def _principled(mat, spec, images):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    out = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if bsdf is None:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        if out:
            links.new(bsdf.outputs[0], out.inputs[0])
    bsdf.location = (300, 300)
    if 'Roughness' in bsdf.inputs:
        bsdf.inputs['Roughness'].default_value = 0.8
    y = 400
    for s in spec['slots']:
        node = nodes.new('ShaderNodeTexImage')
        node.label = 'slot %d: %s (kind %d)' % (s['slot'], s['name'], s['kind'])
        node.location = (-400, y)
        y -= 280
        img = images.get(s['slot'])
        if img is None:
            continue
        node.image = img
        if s['slot'] == 0 and not s['name'].lower().endswith('_outline_ee'):
            links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
            if 'Alpha' in bsdf.inputs:
                links.new(node.outputs['Alpha'], bsdf.inputs['Alpha'])
        elif '_nml' in s['name'].lower() and s['slot'] <= 2:
            nm = nodes.new('ShaderNodeNormalMap')
            nm.location = (0, y + 280)
            links.new(node.outputs['Color'], nm.inputs['Color'])
            links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    if hasattr(mat, 'surface_render_method'):
        mat.surface_render_method = 'DITHERED'
    if hasattr(mat, 'blend_method'):
        try:
            mat.blend_method = 'CLIP'
            mat.alpha_threshold = 0.5
        except (TypeError, AttributeError):
            pass


def make_material(spec, cache_dir, pack, shading='TOON', ring=None):
    """shading TOON: the game shader emulation (toon.py) for the ported families, Principled BSDF otherwise"""
    toon_on = shading == 'TOON' and spec.get('shader', 'PLAIN') != 'PLAIN'
    images = load_images(spec, cache_dir, pack, toon_on)
    mat = bpy.data.materials.new(spec['name'])
    mat.use_nodes = True
    if toon_on and spec.get('shader') == 'STAGE':
        cubes = [images.get(k) for k in spec.get('cube_slots', ())]
        stage.build_stage(mat, spec, images, cubes)
    elif toon_on:
        toon.build_toon(mat, spec, images, ring)
        animate_channels(mat, spec)
        used = {n.image.name for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image}
        y = -900
        for s in spec['slots']:
            img = images.get(s['slot'])
            if img is not None and img.name in used:
                continue
            node = mat.node_tree.nodes.new('ShaderNodeTexImage')
            node.label = 'slot %d: %s (kind %d, unused)' % (s['slot'], s['name'], s['kind'])
            node.image = img
            node.location = (-1600, y)
            y -= 280
    else:
        _principled(mat, spec, images)
    mat['tmd_fourcc'] = spec['fourcc']
    mat['tmd_model'] = spec['model']
    mat['tmd_index'] = spec['index']
    mat['tmd_shader'] = spec.get('shader', '')
    mat['tmd_params'] = list(spec['params'])
    mat['tmd_params_orig'] = list(spec['params'])
    mat['tmd_param_keys'] = sorted(int(k) for k in (spec.get('param_keys') or {}))
    if spec.get('shader') == 'STAGE' and spec.get('stage'):
        import json
        mat['tmd_stage_spec'] = json.dumps(dict(stage=spec['stage'], cube_slots=list(spec.get('cube_slots', ())),
                                                has_uv=list(spec.get('has_uv') or []), fourcc=spec['fourcc'],
                                                shader='STAGE', blend=bool(spec.get('blend', False))))
    mat['tmd_blend'] = bool(spec.get('blend', False))
    mat['tmd_images'] = {str(k): v.name for k, v in images.items()}
    if spec.get('outline'):
        try:
            oi = image_from_dds(spec['outline']['name'], spec['outline']['dds'], cache_dir, pack)
            _set_colorspace(oi, 'Non-Color')
            mat['tmd_outline_image'] = oi.name
        except Exception as e:
            print('evporter: %s: %s' % (spec['outline']['name'], e))
    return mat


def make_outline_material(spec):
    mat = bpy.data.materials.new(spec['name'] + '_outline')
    mat.use_nodes = True
    toon.build_outline(mat, spec)
    mat['tmd_outline_of'] = spec['name']
    return mat


def sample_rgba(tex, uv):
    """bilinear sample of a decoded texture (h, w, 4 uint8, top row first) at D3D uvs, repeat wrap -> float 0..1"""
    h, w = tex.shape[:2]
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    x0, y0 = x0 % w, y0 % h
    t = tex.astype(np.float32) / 255.0
    return (t[y0, x0] * (1 - fx) * (1 - fy) + t[y0, x1] * fx * (1 - fy) +
            t[y1, x0] * (1 - fx) * fy + t[y1, x1] * fx * fy)


def make_armature(plan, name, collection):
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    collection.objects.link(obj)
    obj.show_in_front = True
    view_layer = bpy.context.view_layer
    for o in view_layer.objects:
        o.select_set(False)
    view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    heads = {b['name']: to_blender(b['head']) for b in plan.bones}
    kids = {}
    for b in plan.bones:
        if b['parent']:
            kids.setdefault(b['parent'], []).append(b['name'])
    for b in plan.bones:
        eb = arm.edit_bones.new(b['name'])
        h = heads[b['name']]
        tail = None
        for c in kids.get(b['name'], []):
            d = heads[c] - h
            if np.linalg.norm(d) > 1e-3:
                tail = heads[c]
                break
        if tail is None:
            if b['parent']:
                d = h - heads[b['parent']]
                l = np.linalg.norm(d)
                tail = h + (d / l * min(max(0.5 * l, 0.01), 0.05) if l > 1e-4 else np.array([0, 0, 0.02]))
            else:
                tail = h + np.array([0, 0, 0.05])
        if np.linalg.norm(tail - h) < 0.002:
            tail = h + np.array([0, 0, 0.005])
        eb.head = h.tolist()
        eb.tail = tail.tolist()
        eb.roll = 0.0
    for b in plan.bones:
        if b['parent']:
            arm.edit_bones[b['name']].parent = arm.edit_bones[b['parent']]
    bpy.ops.object.mode_set(mode='OBJECT')
    for b in plan.bones:
        bone = arm.bones[b['name']]
        bone['tmd_hash'] = '%08x' % b['hash']
        bone['tmd_model'] = b['model']
        W = np.eye(4)
        W[:3, :3] = b.get('rot', np.eye(3))
        W[3, :3] = b['head']
        bone['tmd_rest'] = [float(x) for x in W.ravel()]
        if b.get('offset') is not None:
            bone['tmd_offset'] = [float(x) for x in b['offset']]
        if b.get('adjust') is not None:
            bone['tmd_adjust'] = [float(x) for x in np.asarray(b['adjust']).ravel()]
    obj['tmd_body_model'] = plan.bones[0]['model'] if plan.bones else ''
    return obj


def clean_faces(spec):
    """drop degenerate faces and faces listed twice (Blender keeps one); returns faces, per-face material slot"""
    faces = np.asarray(spec['faces'])
    fmat = np.asarray(spec['face_material'])
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    s = np.sort(faces, 1)
    _, first = np.unique(s, axis=0, return_index=True)
    uniq = np.zeros(len(faces), bool)
    uniq[first] = True
    keep &= uniq
    return faces[keep], fmat[keep]


def _vertex_groups(obj, groups, verts=None):
    names, gidx, w = groups
    if verts is not None:
        gidx, w = gidx[verts], w[verts]
    vgs = {}
    for b in np.unique(gidx[w > 0]):
        vgs[int(b)] = obj.vertex_groups.new(name=names[int(b)])
    for v in range(len(gidx)):
        for k in range(4):
            if w[v, k] > 0:
                vgs[int(gidx[v, k])].add([v], float(w[v, k]), 'ADD')


def _link(obj, collection, arm_obj):
    collection.objects.link(obj)
    if arm_obj is not None:
        obj.parent = arm_obj
        mod = obj.modifiers.new('Armature', 'ARMATURE')
        mod.object = arm_obj


def make_mesh(spec, materials, collection, arm_obj):
    pos = to_blender(spec['positions'])
    faces, fmat = clean_faces(spec)

    me = bpy.data.meshes.new(spec['name'])
    me.from_pydata(pos.tolist(), [], faces.tolist())
    me.validate(clean_customdata=False)
    ok = len(me.polygons) == len(faces)
    for key in spec['materials']:
        me.materials.append(materials.get(key) if key else None)
    if ok and len(fmat):
        me.polygons.foreach_set('material_index', fmat.astype(np.int32))
    loop_v = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get('vertex_index', loop_v)
    for k, uv in enumerate(spec['uvs']):
        if uv is None:
            continue
        layer = me.uv_layers.new(name='UVMap' if k == 0 else 'UV%d' % k)
        a = uv[loop_v]
        a = np.stack([a[:, 0], 1.0 - a[:, 1]], 1)
        layer.data.foreach_set('uv', a.astype(np.float32).ravel())
    if spec['color'] is not None:
        attr = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
        attr.data.foreach_set('color', np.asarray(spec['color'], np.float32).ravel())
    if spec.get('frame') is not None and spec['kind'] == 'face':
        for name, v in zip(('tmd_tangent', 'tmd_binormal', 'tmd_normal'), spec['frame']):
            attr = me.attributes.new(name, 'FLOAT_VECTOR', 'POINT')
            attr.data.foreach_set('vector', to_blender(v).astype(np.float32).ravel())
    if hasattr(me, 'shade_smooth'):
        me.shade_smooth()
    else:
        me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    if spec['normals'] is not None:
        if hasattr(me, 'use_auto_smooth'):
            me.use_auto_smooth = True
        me.normals_split_custom_set_from_vertices(to_blender(spec['normals']).tolist())
        zero = spec.get('zero_normals')
        if zero is not None and zero.any():
            at = me.attributes.new('tmd_zero_normal', 'BOOLEAN', 'POINT')
            at.data.foreach_set('value', zero.astype(bool))

    obj = bpy.data.objects.new(spec['name'], me)
    _link(obj, collection, arm_obj)
    if spec['groups'] is not None:
        _vertex_groups(obj, spec['groups'])

    t = me.tmd_material
    t.model = spec['model']
    t.kind = spec['kind']
    t.mesh_index = spec['mesh_index']
    t.tags = spec['tags']
    t.xform_type = spec['xform_type']
    t.mesh_flags = spec['flags']
    t.vertex_flags = spec['vertex_flags']
    t.bounds = spec['bounds']
    t.hidden_reason = spec['hidden'] or ''
    if spec.get('source'):
        me['tmd_source'] = spec['source']
    for key in spec['materials']:
        it = t.materials.add()
        m = materials.get(key) if key else None
        it.material = m
        if m is not None:
            it.index = m.get('tmd_index', -1)
            it.fourcc = m.get('tmd_fourcc', '')
            params = list(m.get('tmd_params', []))
            it.num_params = len(params)
            it.params = (params + [0.0] * 32)[:32]
            it.textures = ' | '.join(n.label for n in m.node_tree.nodes if n.type == 'TEX_IMAGE')
    if spec['hidden']:
        obj.hide_set(True)
        obj.hide_render = True
    return obj


OUTLINE_WIDTH = 0.007 * 0.5


def make_outline(spec, outlines, collection, arm_obj):
    """the inverted hull of the outline pass as its own object: faces of materials with an outline texture,
    reversed, vertices pushed along the rest normal by 0.0035 * outline alpha (sampled per vertex, like the
    game's vertex shader), colour / alpha * 0.5 per corner in the tmd_outline attribute, same skin weights.
    outlines: material key -> (outline material, decoded texture (h, w, 4))"""
    if spec['normals'] is None or spec['uvs'][0] is None:
        return None
    faces, fmat = clean_faces(spec)
    keys = spec['materials']
    sel = np.array([keys[k] in outlines if keys[k] else False for k in fmat], bool) if len(fmat) else np.zeros(0, bool)
    if not sel.any():
        return None
    faces, fmat = faces[sel], fmat[sel]
    uv = spec['uvs'][0]
    corner = np.zeros((len(faces), 3, 4), np.float32)
    used = sorted({int(k) for k in fmat})
    for k in used:
        f = fmat == k
        c = sample_rgba(outlines[keys[k]][1], uv[faces[f].ravel()])
        corner[f] = c.reshape(-1, 3, 4)
    verts = np.unique(faces)
    remap = np.full(len(spec['positions']), -1, np.int64)
    remap[verts] = np.arange(len(verts))
    alpha = np.zeros(len(spec['positions']), np.float32)
    np.maximum.at(alpha, faces.ravel(), corner[:, :, 3].ravel())
    nrm = np.asarray(spec['normals'], np.float64)
    pos = np.asarray(spec['positions'], np.float64) + nrm * (OUTLINE_WIDTH * alpha)[:, None]
    pos = to_blender(pos[verts])
    tri = remap[faces][:, [0, 2, 1]]
    col = corner[:, [0, 2, 1], :].copy()
    col[:, :, 3] *= 0.5

    me = bpy.data.meshes.new(spec['name'] + '_outline')
    me.from_pydata(pos.tolist(), [], tri.tolist())
    me.validate(clean_customdata=False)
    slots = []
    for k in used:
        me.materials.append(outlines[keys[k]][0])
        slots.append(k)
    if len(me.polygons) == len(tri):
        me.polygons.foreach_set('material_index', np.array([slots.index(int(k)) for k in fmat], np.int32))
        attr = me.color_attributes.new('tmd_outline', 'FLOAT_COLOR', 'CORNER')
        attr.data.foreach_set('color', col.ravel())
    if hasattr(me, 'shade_smooth'):
        me.shade_smooth()
    obj = bpy.data.objects.new(spec['name'] + '_outline', me)
    _link(obj, collection, arm_obj)
    if spec['groups'] is not None:
        _vertex_groups(obj, spec['groups'], verts)
    obj['tmd_outline_of'] = spec['name']
    if spec['hidden']:
        obj.hide_set(True)
        obj.hide_render = True
    return obj


def make_head_ring(plan, arm, collection):
    """two empties for the HRH0 hair ring: <name>_head_point follows the head bone at the shader's head point,
    <name>_head_ring copies its location only (world axes), so the hair material's Texture Coordinate > Object
    gives the pixel's offset from the head point in world orientation"""
    import mathutils
    p = mathutils.Vector(to_blender(plan.head_point).tolist())
    bone = arm.data.bones.get(plan.head_bone)
    if bone is None:
        return None
    a = bpy.data.objects.new(plan.name + '_head_point', None)
    collection.objects.link(a)
    a.empty_display_type = 'SPHERE'
    a.empty_display_size = 0.02
    a.parent = arm
    a.parent_type = 'BONE'
    a.parent_bone = bone.name
    tail = bone.matrix_local @ mathutils.Matrix.Translation((0.0, bone.length, 0.0))
    a.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    a.location = tail.inverted() @ p
    b = bpy.data.objects.new(plan.name + '_head_ring', None)
    collection.objects.link(b)
    b.empty_display_type = 'PLAIN_AXES'
    b.empty_display_size = 0.05
    b.location = p
    c = b.constraints.new('COPY_LOCATION')
    c.target = a
    for o in (a, b):
        o.hide_set(True)
        o.hide_render = True
    return b


def build(plan, cache_dir='', pack=True, shading='TOON', outlines=True, view_standard=True, rig=None):
    """rig: a yure.RigidRig for the character's rigid body physics (None: no physics)"""
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'evporter_tmd')
    coll = bpy.data.collections.new(plan.name)
    bpy.context.scene.collection.children.link(coll)
    prev_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
    if prev_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    arm = make_armature(plan, plan.name, coll) if plan.bones else None
    toon_on = shading == 'TOON'
    ring = None
    if toon_on and arm is not None and plan.head_point is not None and \
            any(v.get('shader') == 'HAIR' for v in plan.materials.values()):
        ring = make_head_ring(plan, arm, coll)
    mats, outl = {}, {}
    for k, v in plan.materials.items():
        mats[k] = make_material(v, os.path.join(cache_dir, v['model']), pack, shading, ring)
        if toon_on and outlines and v.get('outline'):
            try:
                _, _, rgba = F.decode_dds(v['outline']['dds'])
                outl[k] = (make_outline_material(v), rgba)
            except Exception as e:
                print('evporter: outline %s: %s' % (v['outline']['name'], e))
    objs = [make_mesh(o, mats, coll, arm) for o in plan.objects]
    if outl:
        objs += [x for x in (make_outline(o, outl, coll, arm) for o in plan.objects) if x is not None]
    if toon_on and view_standard:
        try:
            bpy.context.scene.view_settings.view_transform = 'Standard'
            bpy.context.scene.view_settings.look = 'None'
        except (TypeError, AttributeError):
            pass
    if arm is not None:
        arm['tmd_character'] = plan.name
        arm['tmd_layer'] = plan.layer
        if rig is not None and rig.bodies:
            rigid.build(rig, arm, coll, plan.name)
    return arm, objs
