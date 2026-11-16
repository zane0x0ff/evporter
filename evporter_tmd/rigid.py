import math
import numpy as np
import bpy
import mathutils

from . import yure

PREFIX = 'TMD Physics'


def to_blender(a):
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[..., 0], -a[..., 2], a[..., 1]], -1)


def _frame(cols):
    """3x3 with game-space columns -> Blender rotation matrix (columns converted the same way as positions)"""
    m = np.stack([to_blender(cols[:, k]) for k in range(3)], 1)
    return mathutils.Matrix(m.tolist())


def capsule_points(r, length, rings=4, segs=12):
    """vertices / quads of a capsule along local Z, centred; bounding box (2r, 2r, length + 2r), which is how
    Blender sizes its CAPSULE (and SPHERE, length 0) collision shapes"""
    rows = []
    for half, lo, hi in ((length * 0.5, 0, rings), (-length * 0.5, -rings, 0)):
        for i in range(hi, lo - 1, -1):
            if half > 0 and i == 0 and length == 0:
                continue
            phi = 0.5 * math.pi * i / rings
            z, rr = half + r * math.sin(phi), r * math.cos(phi)
            rows.append([(rr * math.cos(2 * math.pi * k / segs), rr * math.sin(2 * math.pi * k / segs), z) for k in range(segs)])
    verts = [v for row in rows for v in row]
    faces = [(i * segs + k, i * segs + (k + 1) % segs, (i + 1) * segs + (k + 1) % segs, (i + 1) * segs + k)
             for i in range(len(rows) - 1) for k in range(segs)]
    return verts, faces


def _capsule_mesh(name, r, length):
    verts, faces = capsule_points(r, length)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    return me


def _world_collections(w):
    """the rigid body world's object and constraint collections (created when missing). Linking an object into
    them gives it rigid body / constraint settings (BKE_rigidbody_main_collection_object_add), without operators"""
    if w.collection is None:
        w.collection = bpy.data.collections.new('RigidBodyWorld')
    if w.constraints is None:
        w.constraints = bpy.data.collections.new('RigidBodyConstraints')
    return w.collection, w.constraints


def _with_active(o, op, **kw):
    """fallback: run a rigidbody operator with o as the real active, selected object"""
    vl = bpy.context.view_layer
    prev = vl.objects.active
    for x in bpy.context.selected_objects:
        x.select_set(False)
    hidden = o.hide_get()
    o.hide_set(False)
    o.select_set(True)
    vl.objects.active = o
    try:
        with bpy.context.temp_override(active_object=o, object=o, selected_objects=[o], selected_editable_objects=[o]):
            op(**kw)
    finally:
        o.select_set(False)
        o.hide_set(hidden)
        vl.objects.active = prev


def ensure_world(scene):
    if scene.rigidbody_world is None:
        with bpy.context.temp_override(scene=scene):
            bpy.ops.rigidbody.world_add()
    w = scene.rigidbody_world
    w.enabled = True
    if hasattr(w, 'substeps_per_frame'):
        w.substeps_per_frame = max(w.substeps_per_frame, 10)
    w.solver_iterations = max(w.solver_iterations, 20)
    w.point_cache.frame_start = scene.frame_start
    w.point_cache.frame_end = max(w.point_cache.frame_end, scene.frame_end)
    return w


def build(rig, arm, parent_collection, name):
    """make the rig's bodies, constraints and bone drives for the armature object arm"""
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    world = ensure_world(scene)
    rb_coll, con_coll = _world_collections(world)
    coll = bpy.data.collections.new(name + '_physics')
    parent_collection.children.link(coll)
    objs = {}

    def tail_matrix(bone_name):
        b = arm.data.bones[bone_name]
        return arm.matrix_world @ b.matrix_local @ mathutils.Matrix.Translation((0.0, b.length, 0.0))

    for b in rig.bodies:
        a, e = to_blender(b['a']), to_blender(b['b'])
        r = float(b['radius'])
        if b['shape'] == 'CAPSULE':
            axis = e - a
            length = float(np.linalg.norm(axis))
            me = _capsule_mesh(b['name'], r, length)
            rot = _frame(yure.frame_y(np.asarray(b['b']) - np.asarray(b['a'])))
            rot = rot @ mathutils.Matrix.Rotation(-math.pi * 0.5, 3, 'X')
        else:
            me = _capsule_mesh(b['name'], r, 0.0)
            rot = mathutils.Matrix.Identity(3)
        o = bpy.data.objects.new(b['name'], me)
        coll.objects.link(o)
        M = mathutils.Matrix.Translation(((a + e) * 0.5).tolist()) @ rot.to_4x4()
        if b['parent_bone']:
            o.parent = arm
            o.parent_type = 'BONE'
            o.parent_bone = b['parent_bone']
            T = tail_matrix(b['parent_bone'])
            o.matrix_parent_inverse = T.inverted()
            o.matrix_basis = M
        else:
            o.matrix_world = M
        o.display_type = 'WIRE'
        o.hide_render = True
        o['tmd_physics'] = b['kind']
        o['tmd_label'] = b.get('label', '')
        objs[b['name']] = o
    view_layer.update()

    for b in rig.bodies:
        o = objs[b['name']]
        if o.rigid_body is None:
            rb_coll.objects.link(o)
        if o.rigid_body is None:
            _with_active(o, bpy.ops.rigidbody.object_add, type='ACTIVE')
        if o.rigid_body is None:
            raise RuntimeError('could not make %s a rigid body' % o.name)
        rb = o.rigid_body
        rb.type = 'ACTIVE'
        rb.kinematic = b['kind'] != 'dynamic'
        rb.collision_shape = b['shape']
        rb.mesh_source = 'BASE'
        rb.mass = float(b['mass'])
        rb.friction = 0.1
        rb.restitution = 0.0
        rb.linear_damping, rb.angular_damping = (float(x) for x in b['damping'])
        rb.use_margin = True
        rb.collision_margin = 0.001
        rb.collision_collections = [i in b['collections'] for i in range(20)]
        rb.use_deactivation = False

    empties = []
    body_ref = {b['name']: b['parent_bone'] for b in rig.bodies}
    for c in rig.constraints:
        o = bpy.data.objects.new(c['name'], None)
        coll.objects.link(o)
        o.empty_display_type = 'ARROWS' if c['type'] == 'GENERIC_SPRING' else 'PLAIN_AXES'
        o.empty_display_size = 0.01
        M = mathutils.Matrix.Translation(to_blender(c['pivot']).tolist()) @ _frame(np.asarray(c['frame'])).to_4x4()
        ref = body_ref.get(c['b']) or body_ref.get(c['a'])
        if ref:
            o.parent = arm
            o.parent_type = 'BONE'
            o.parent_bone = ref
            o.matrix_parent_inverse = tail_matrix(ref).inverted()
            o.matrix_basis = M
        else:
            o.matrix_world = M
        o['tmd_label'] = c.get('label', '')
        empties.append((o, c))
    view_layer.update()
    for o, c in empties:
        if o.rigid_body_constraint is None:
            con_coll.objects.link(o)
        if o.rigid_body_constraint is None:
            _with_active(o, bpy.ops.rigidbody.constraint_add, type=c['type'])
        k = o.rigid_body_constraint
        if k is None:
            raise RuntimeError('could not add a rigid body constraint to %s' % o.name)
        k.type = c['type']
        k.object1, k.object2 = objs[c['a']], objs[c['b']]
        k.disable_collisions = True
        if c['type'] != 'GENERIC_SPRING':
            continue
        if hasattr(k, 'spring_type'):
            k.spring_type = 'SPRING2'
        for i, ax in enumerate('xyz'):
            if c['lin'] is not None:
                setattr(k, 'use_limit_lin_' + ax, True)
                setattr(k, 'limit_lin_%s_lower' % ax, float(c['lin'][i][0]))
                setattr(k, 'limit_lin_%s_upper' % ax, float(c['lin'][i][1]))
            if c['ang'] is not None:
                setattr(k, 'use_limit_ang_' + ax, True)
                setattr(k, 'limit_ang_%s_lower' % ax, float(c['ang'][i][0]))
                setattr(k, 'limit_ang_%s_upper' % ax, float(c['ang'][i][1]))
            if c['lin_k'][i] > 0:
                setattr(k, 'use_spring_' + ax, True)
                setattr(k, 'spring_stiffness_' + ax, float(c['lin_k'][i]))
                setattr(k, 'spring_damping_' + ax, float(c['damping']))
            if c['ang_k'][i] > 0:
                setattr(k, 'use_spring_ang_' + ax, True)
                setattr(k, 'spring_stiffness_ang_' + ax, float(c['ang_k'][i]))
                setattr(k, 'spring_damping_ang_' + ax, float(c['damping']))

    for d in rig.drives:
        pb = arm.pose.bones.get(d['bone'])
        if pb is None:
            continue
        if d['kind'] == 'DAMPED_TRACK':
            con = pb.constraints.new('DAMPED_TRACK')
            con.track_axis = 'TRACK_Y'
        else:
            con = pb.constraints.new('COPY_LOCATION')
            con.target_space = con.owner_space = 'WORLD'
        con.name = PREFIX
        con.target = objs[d['body']]

    for o in list(objs.values()) + [e for e, _ in empties]:
        o.hide_set(True)
    arm['tmd_physics_collection'] = coll.name
    return coll


def set_enabled(arm, on):
    for pb in arm.pose.bones:
        for con in pb.constraints:
            if con.name.startswith(PREFIX):
                con.influence = 1.0 if on else 0.0
    w = bpy.context.scene.rigidbody_world
    if w is not None:
        w.enabled = on
        if on:
            w.point_cache.frame_start = bpy.context.scene.frame_start


def remove(arm):
    for pb in arm.pose.bones:
        for con in list(pb.constraints):
            if con.name.startswith(PREFIX):
                pb.constraints.remove(con)
    coll = bpy.data.collections.get(arm.get('tmd_physics_collection', ''))
    if coll is not None:
        for o in list(coll.objects):
            me = o.data if o.type == 'MESH' else None
            bpy.data.objects.remove(o)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        bpy.data.collections.remove(coll)
    if 'tmd_physics_collection' in arm:
        del arm['tmd_physics_collection']


def reset_cache(scene):
    """mark the rigid body cache outdated (as editing a world setting does, rna_RigidBodyWorld_reset), so the
    simulation is rebuilt from the current pose when it next starts"""
    w = scene.rigidbody_world
    if w is not None:
        w.solver_iterations = w.solver_iterations


def _scene_has_tmd_physics(scene):
    return any(o.type == 'ARMATURE' and o.get('tmd_physics_collection') and o.get('tmd_physics_on', True)
               for o in scene.objects)


@bpy.app.handlers.persistent
def frame_change_pre(scene, *args):
    """at the simulation's start frame, rebuild from the current pose: the particles are parented to their
    reference bones, so a character moved or posed since the last run starts with its strands attached"""
    w = scene.rigidbody_world
    if w is None or not w.enabled or scene.frame_current != w.point_cache.frame_start:
        return
    if _scene_has_tmd_physics(scene):
        reset_cache(scene)


def register_handlers():
    if frame_change_pre not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frame_change_pre)


def unregister_handlers():
    while frame_change_pre in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frame_change_pre)
