import numpy as np
import bpy

from . import motion as MO
from . import formats as F

GAME_FPS = 60.0


def _np(m):
    return np.array([list(r) for r in m], float)


def game_bones(arm):
    """the armature's game bones (IK controls and other added bones left out), parent-first"""
    return [b for b in arm.data.bones if not b.get('tmd_ik')]


def rig_from_armature(arm):
    """motion.Rig of an imported armature (bone custom properties written by build.make_armature)"""
    bones, missing = [], 0
    A3 = MO.AI[:3, :3]
    for b in game_bones(arm):
        rest = b.get('tmd_rest')
        if rest is not None:
            W = np.array(list(rest), float).reshape(4, 4)
        else:
            missing += 1
            W = np.eye(4)
            W[3, :3] = A3 @ np.array(b.head_local, float)
        h = b.get('tmd_hash')
        off = b.get('tmd_offset')
        adj = b.get('tmd_adjust')
        bones.append(dict(name=b.name, parent=b.parent.name if b.parent else None,
                          hash=int(h, 16) if isinstance(h, str) else F.tamsoft_hash(b.name),
                          model=b.get('tmd_model', ''), rest=W,
                          offset=None if off is None else np.array(list(off), float),
                          adjust=None if adj is None else np.array(list(adj), float).reshape(4, 4)))
    rig = MO.Rig(bones, arm.get('tmd_body_model', ''))
    rig.missing_rest = missing
    return rig


def catalogue(path):
    """[(name, Tmo)] of a motion catalogue / .tmo2 file"""
    with open(path, 'rb') as f:
        data = f.read()
    import os
    return MO.motions_in(data, os.path.basename(path))


def _container(action, arm, id_type='OBJECT'):
    """(fcurves collection, group getter, slot) for the action; Blender 4.4+ layered actions or the legacy API.
    id_type: the slot's ID type ('OBJECT', 'CAMERA' for camera data, ...)"""
    if hasattr(action, 'slots') and hasattr(action, 'layers'):
        slot = action.slots.new(id_type, arm.name) if len(action.slots) == 0 else action.slots[0]
        layer = action.layers[0] if len(action.layers) else action.layers.new('Layer')
        strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type='KEYFRAME')
        bag = strip.channelbag(slot, ensure=True)
        groups = {}

        def group(name):
            g = groups.get(name)
            if g is None:
                g = groups[name] = bag.groups.new(name)
            return g

        def new(path, index, gname):
            fc = bag.fcurves.new(path, index=index)
            fc.group = group(gname)
            return fc
        return new, slot

    def new(path, index, gname):
        return action.fcurves.new(path, index=index, action_group=gname)
    return new, None


def _keys(fc, frames, values):
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set('co', co)
    fc.keyframe_points.foreach_set('interpolation', np.ones(n, np.int32))
    fc.update()


def import_motion(arm, rig, name, base, companions, root_motion=True, frame_scale=1.0, start=1, assign=True):
    """one action named after the motion; returns it. The game's files key frames 0..numFrames (the closing frame of
    a loop too), so the action gets numFrames + 1 keys"""
    nf = base.num_frames + 1 if base.num_frames > 0 else 1
    W = rig.evaluate(base, companions, np.arange(nf), root_motion)
    bl = game_bones(arm)
    ML = np.stack([_np(b.matrix_local) for b in bl])
    index = {b.name: i for i, b in enumerate(bl)}
    parents = [index[b.parent.name] if b.parent else -1 for b in bl]
    basis, _ = MO.pose_basis(rig, W, ML, parents)
    loc, quat, scale = MO.decompose(basis)
    action = bpy.data.actions.new(name)
    action['tmd_motion'] = name
    action['tmd_companions'] = ','.join(sorted(companions))
    new, slot = _container(action, arm)
    frames = start + np.arange(nf) * frame_scale
    ident = np.eye(4)
    keyed = 0
    for i, b in enumerate(bl):
        if np.abs(basis[:, i] - ident).max() < 1e-6:
            continue
        pb = arm.pose.bones.get(b.name)
        if pb is None:
            continue
        pb.rotation_mode = 'QUATERNION'
        base_path = 'pose.bones["%s"].' % bpy.utils.escape_identifier(b.name)
        for k in range(3):
            _keys(new(base_path + 'location', k, b.name), frames, loc[:, i, k])
        for k in range(4):
            _keys(new(base_path + 'rotation_quaternion', k, b.name), frames, quat[:, i, k])
        if np.abs(scale[:, i] - 1).max() > 1e-5:
            for k in range(3):
                _keys(new(base_path + 'scale', k, b.name), frames, scale[:, i, k])
        keyed += 1
    action['tmd_keyed_bones'] = keyed
    action.use_fake_user = True
    try:
        action.frame_range = (frames[0], frames[-1])
        action.use_frame_range = True
    except (AttributeError, TypeError):
        pass
    if assign:
        assign_action(arm, action)
    return action


def assign_action(arm, action):
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action
    if hasattr(arm.animation_data, 'action_slot') and hasattr(action, 'slots') and len(action.slots):
        try:
            arm.animation_data.action_slot = action.slots[0]
        except (AttributeError, TypeError, RuntimeError):
            pass


def import_motions(arm, path, names, companions=True, root_motion=True, fps_mode='SCENE_60', reset_pose=True):
    """import the named motions of a catalogue as actions on the armature; the last one is assigned.
    fps_mode: SCENE_60 sets the scene to 60 fps (one key per game frame); RESAMPLE keeps the scene rate and places
    the game frames at their time (60 fps -> scene fps)"""
    scene = bpy.context.scene
    mots = dict(catalogue(path))
    rig = rig_from_armature(arm)
    if fps_mode == 'SCENE_60':
        scene.render.fps, scene.render.fps_base = 60, 1.0
        scale = 1.0
    else:
        scale = (scene.render.fps / scene.render.fps_base) / GAME_FPS
    if reset_pose:
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)
    actions = []
    for n in names:
        base = mots.get(n)
        if base is None:
            continue
        comp = {s: mots[n + s] for s in MO.COMPANIONS if companions and n + s in mots}
        actions.append(import_motion(arm, rig, n, base, comp, root_motion, scale, scene.frame_start, assign=False))
    if actions:
        a = actions[-1]
        assign_action(arm, a)
        last = int(np.ceil(scene.frame_start + (mots[a['tmd_motion']].num_frames - 1) * scale))
        scene.frame_end = max(scene.frame_start + 1, last)
        w = scene.rigidbody_world
        if w is not None:
            w.point_cache.frame_end = max(w.point_cache.frame_end, scene.frame_end)
        scene.frame_set(scene.frame_start)
    return actions, rig


def physics_bones(arm):
    """names of the bones the rigid body rig drives (rigid.py constraints) and their descendants"""
    from .rigid import PREFIX
    out = set()
    for b in arm.data.bones:
        pb = arm.pose.bones.get(b.name)
        if (b.parent is not None and b.parent.name in out) or \
                (pb is not None and any(c.name.startswith(PREFIX) for c in pb.constraints)):
            out.add(b.name)
    return out


def export_motion(arm, path, frame_start, frame_end, timing='TIME', visual=False, physics=False, split=True,
                  all_bones=True):
    """write the armature's animation over [frame_start, frame_end] as a game motion (.tmo2).
    timing: TIME samples at 60 fps by time (scene frame rate respected), FRAMES takes one scene frame per game frame.
    visual: the evaluated pose (constraints, IK); otherwise the keyed pose (matrix_basis) only.
    physics: keep what the rigid body rig does to its bones (visual only); otherwise they get their keyed pose.
    split: hair / face bones go to <name>_hr.tmo2 / <name>_hd.tmo2 (as the game's companion motions), only when
    they move; otherwise into the main file. Weapon bones are not written.
    all_bones: a track for every bone of the part (as the game's files) or only for the bones that move.
    Returns [(path, frames, tracks)]."""
    import os
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    step = 1.0 if timing == 'FRAMES' else fps / GAME_FPS
    n = int(np.floor((frame_end - frame_start) / step + 1e-6)) + 1
    if n < 2:
        raise ValueError("the frame range needs at least two frames")
    times = frame_start + np.arange(n) * step
    rig = rig_from_armature(arm)
    bl = game_bones(arm)
    nb = len(bl)
    ML = np.stack([_np(b.matrix_local) for b in bl])
    index = {b.name: i for i, b in enumerate(bl)}
    parents = [index[b.parent.name] if b.parent else -1 for b in bl]
    pbs = [arm.pose.bones[b.name] for b in bl]
    phys = set() if physics and visual else {index[nm] for nm in physics_bones(arm)}
    world = scene.rigidbody_world
    rb_enabled = world.enabled if world is not None else None
    if world is not None and not (visual and physics):
        world.enabled = False
    frame0, sub0 = scene.frame_current, scene.frame_subframe
    pose = np.zeros((n, nb, 4, 4))
    basis = np.zeros((n, nb, 4, 4))
    try:
        for k, t in enumerate(times):
            f = int(np.floor(t + 1e-6))
            scene.frame_set(f, subframe=float(t - f) if t - f > 1e-6 else 0.0)
            for i, pb in enumerate(pbs):
                basis[k, i] = _np(pb.matrix_basis)
                if visual:
                    pose[k, i] = _np(pb.matrix)
    finally:
        if world is not None:
            world.enabled = rb_enabled
        scene.frame_set(frame0, subframe=sub0)
    if visual:
        for i in range(nb):
            if i in phys:
                p = parents[i]
                rel = ML[i] if p < 0 else np.linalg.inv(ML[p]) @ ML[i]
                pose[:, i] = (rel[None] @ basis[:, i]) if p < 0 else pose[:, p] @ rel @ basis[:, i]
    else:
        pose = MO.pose_from_basis(basis, ML, parents)
    W = MO.world_from_pose(rig, pose, ML)
    L, moved = MO.local_tracks(rig, W)
    base, _ = os.path.splitext(path)
    groups = {'': [], '_hr': [], '_hd': []}
    for i, b in enumerate(rig.bones):
        part = rig.part(b)
        if part == 'weapon':
            continue
        key = {'hair': '_hr', 'face': '_hd'}.get(part, '') if split else ''
        groups[key].append(i)
    written = []
    for suffix, ids in groups.items():
        if suffix and not any(moved[i] for i in ids):
            continue
        if not all_bones:
            ids = [i for i in ids if moved[i]]
        data = MO.encode_tmo(MO.tracks_of(rig, L, ids), n - 1)
        out = base + suffix + '.tmo2'
        with open(out, 'wb') as fh:
            fh.write(data)
        written.append((out, n - 1, len(ids)))
    return written
