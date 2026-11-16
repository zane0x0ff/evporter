import math
import numpy as np
import bpy
import mathutils

PREFIX = 'TMD IK'
COLLECTION = 'IK controls'
LIMBS = (
    ('leg_L', 'groin_L', 'knee_L', 'ankle_L', +1.0),
    ('leg_R', 'groin_R', 'knee_R', 'ankle_R', +1.0),
    ('arm_L', 'shoulder_L', 'elbow_L', 'hand_L', -1.0),
    ('arm_R', 'shoulder_R', 'elbow_R', 'hand_R', -1.0),
)


def _v(x):
    return np.array(list(x), float)


def has_ik(arm):
    return any(b.get('tmd_ik') for b in arm.data.bones)


def _top_bone(arm):
    for b in arm.data.bones:
        if b.parent is None and not b.get('tmd_ik'):
            return b.name
    return None


def _forward(eb):
    """the character's facing in armature space: left x up (Blender is right-handed; she faces -Y after import)"""
    up = np.array([0.0, 0.0, 1.0])
    if 'groin_L' in eb and 'groin_R' in eb:
        left = _v(eb['groin_L'].head) - _v(eb['groin_R'].head)
    elif 'shoulder_L' in eb and 'shoulder_R' in eb:
        left = _v(eb['shoulder_L'].head) - _v(eb['shoulder_R'].head)
    else:
        return np.array([0.0, -1.0, 0.0])
    f = np.cross(left, up)
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])


def _signed_angle(u, v, normal):
    a = u.angle(v)
    c = u.cross(v)
    if c.length > 1e-9 and c.angle(normal) < 1:
        a = -a
    return a


def _pole_angle(base, mid, pole_head):
    """Blender's pole angle that keeps the chain in its rest pose (base: chain root edit bone, mid: the IK bone)"""
    pole_normal = (mid.tail - base.head).cross(pole_head - base.head)
    axis = pole_normal.cross(base.tail - base.head)
    if axis.length < 1e-9:
        return 0.0
    return _signed_angle(base.x_axis, axis, base.tail - base.head)


def setup(arm, context=None):
    """adds the IK controls; returns (limbs set up, notes)"""
    context = context or bpy.context
    if has_ik(arm):
        remove(arm, context)
    notes = []
    view_layer = context.view_layer
    prev_active, prev_mode = view_layer.objects.active, (context.object.mode if context.object else 'OBJECT')
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    view_layer.objects.active = arm
    top = _top_bone(arm)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    coll = None
    if hasattr(arm.data, 'collections'):
        coll = arm.data.collections.get(COLLECTION) or arm.data.collections.new(COLLECTION)
    fwd = mathutils.Vector(_forward(eb).tolist())
    made = []
    for name, root, mid, end, side in LIMBS:
        if not all(n in eb for n in (root, mid, end)):
            notes.append('%s: bones missing, skipped' % name)
            continue
        r, m, e = eb[root], eb[mid], eb[end]
        if m.parent is None or m.parent.name != root or e.parent is None or e.parent.name != mid:
            notes.append('%s: not a root -> mid -> end chain, skipped' % name)
            continue
        if (m.tail - e.head).length > 1e-3:
            notes.append('%s: %s tail is %.1f mm from %s, control placed at the tail'
                         % (name, mid, (m.tail - e.head).length * 1000, end))
        c = eb.new('IK_%s' % name)
        c.head = m.tail.copy()
        c.tail = m.tail + (e.tail - e.head)
        c.roll = e.roll
        c.use_deform = False
        c.parent = eb[top] if top else None
        length = (r.head - m.head).length + (m.head - m.tail).length
        p = eb.new('IK_%s_pole' % name)
        p.head = m.head + fwd * (side * max(length, 0.1))
        p.tail = p.head + mathutils.Vector((0.0, 0.0, 0.05))
        p.use_deform = False
        p.parent = eb[top] if top else None
        angle = _pole_angle(r, m, p.head)
        for b, pal in ((c, 'THEME09'), (p, 'THEME03')):
            if coll is not None:
                coll.assign(b)
            if hasattr(b, 'color'):
                b.color.palette = pal
        made.append((name, root, mid, end, c.name, p.name, angle))
    bpy.ops.object.mode_set(mode='POSE')
    for name, root, mid, end, cn, pn, angle in made:
        arm.data.bones[cn]['tmd_ik'] = 'control'
        arm.data.bones[pn]['tmd_ik'] = 'pole'
        k = arm.pose.bones[mid].constraints.new('IK')
        k.name = PREFIX
        k.target, k.subtarget = arm, cn
        k.pole_target, k.pole_subtarget = arm, pn
        k.pole_angle = angle
        k.chain_count = 2
        k.use_tail = True
        k.use_stretch = False
        cr = arm.pose.bones[end].constraints.new('COPY_ROTATION')
        cr.name = PREFIX
        cr.target, cr.subtarget = arm, cn
        cr.target_space = cr.owner_space = 'WORLD'
        for b in (cn, pn):
            arm.pose.bones[b].rotation_mode = 'QUATERNION'
    arm['tmd_ik_on'] = True
    _calibrate_poles(arm, made, context, notes)
    bpy.ops.object.mode_set(mode='OBJECT')
    view_layer.objects.active = prev_active
    if prev_active is not None and prev_mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode=prev_mode)
        except RuntimeError:
            pass
    return [m[0] for m in made], notes


def _chain_error(arm, names):
    err = 0.0
    for n in names:
        pb = arm.pose.bones[n]
        d = pb.matrix - arm.data.bones[n].matrix_local
        err = max(err, max(abs(d[i][j]) for i in range(3) for j in range(4)))
    return err


def _calibrate_poles(arm, made, context, notes):
    """rest pose + controls at rest: keep the pole angle candidate that reproduces the rest pose"""
    ad = arm.animation_data
    action = ad.action if ad else None
    saved = {pb.name: pb.matrix_basis.copy() for pb in arm.pose.bones}
    try:
        if ad is not None:
            ad.action = None
        for pb in arm.pose.bones:
            pb.matrix_basis = mathutils.Matrix.Identity(4)
        for name, root, mid, end, cn, pn, angle in made:
            k = arm.pose.bones[mid].constraints.get(PREFIX)
            best = None
            for a in (angle, angle + math.pi, -angle, math.pi - angle):
                a = math.atan2(math.sin(a), math.cos(a))
                k.pole_angle = a
                context.view_layer.update()
                e = _chain_error(arm, (root, mid, end))
                if best is None or e < best[0]:
                    best = (e, a)
            k.pole_angle = best[1]
            if best[0] > 1e-3:
                notes.append('%s: the IK chain moves %.3f from its rest pose with the controls at rest' % (name, best[0]))
    finally:
        if ad is not None and action is not None:
            from .anim import assign_action
            assign_action(arm, action)
        for pb in arm.pose.bones:
            if pb.name in saved:
                pb.matrix_basis = saved[pb.name]
        context.view_layer.update()


def constraints(arm):
    return [c for pb in arm.pose.bones for c in pb.constraints if c.name.startswith(PREFIX)]


def set_enabled(arm, on, snap=True, context=None):
    """IK on / off. Switching on with snap moves the controls to the current (FK) pose first, so nothing jumps"""
    context = context or bpy.context
    if on and snap:
        match_controls(arm, context)
    for c in constraints(arm):
        c.mute = not on
    arm['tmd_ik_on'] = bool(on)
    for b in arm.data.bones:
        if b.get('tmd_ik'):
            b.hide = not on
    context.view_layer.update()


def match_controls(arm, context=None):
    """puts each limb's foot / hand control and pole where the limb is in the current pose without IK"""
    context = context or bpy.context
    cons = constraints(arm)
    muted = [c.mute for c in cons]
    for c in cons:
        c.mute = True
    context.view_layer.update()
    try:
        targets = []
        for mid in arm.pose.bones:
            k = mid.constraints.get(PREFIX)
            if k is None or k.type != 'IK':
                continue
            root = mid.parent
            ctrl, pole = arm.pose.bones.get(k.subtarget), arm.pose.bones.get(k.pole_subtarget)
            end = next((ch for ch in mid.children if ch.constraints.get(PREFIX) is not None), None)
            if ctrl is None or end is None or root is None:
                continue
            M = end.matrix.copy()
            M.translation = mid.tail
            P = None
            if pole is not None:
                rb = arm.data.bones[root.name]
                rest_off = arm.data.bones[pole.name].head_local - rb.head_local
                R = root.matrix.to_3x3() @ rb.matrix_local.to_3x3().inverted()
                P = pole.matrix.copy()
                P.translation = root.head + R @ rest_off
            targets.append((ctrl, M, pole, P))
        for ctrl, M, pole, P in targets:
            ctrl.matrix = M
            if pole is not None and P is not None:
                pole.matrix = P
            context.view_layer.update()
    finally:
        for c, m in zip(cons, muted):
            c.mute = m
        context.view_layer.update()


def remove(arm, context=None):
    context = context or bpy.context
    for pb in arm.pose.bones:
        for c in [c for c in pb.constraints if c.name.startswith(PREFIX)]:
            pb.constraints.remove(c)
    names = [b.name for b in arm.data.bones if b.get('tmd_ik')]
    if not names:
        return 0
    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for n in names:
        eb = arm.data.edit_bones.get(n)
        if eb is not None:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')
    if hasattr(arm.data, 'collections'):
        coll = arm.data.collections.get(COLLECTION)
        if coll is not None:
            arm.data.collections.remove(coll)
    if 'tmd_ik_on' in arm:
        del arm['tmd_ik_on']
    view_layer.objects.active = prev_active
    return len(names)
