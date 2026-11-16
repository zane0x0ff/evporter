import os
import struct
import numpy as np
import bpy

from . import formats as F, motion as MO, character, build, anim, camera, camera_io, ik


def parse_tmv(b):
    """.tmv mesh visibility (game class CTmv, loader sub_14004e000): {f32 frames, i32 count, 8 bytes 0} then count
    rows of frames bytes; returns (frames, rows (count, frames) uint8). Row i drives mesh i of the model: the game
    (sub_14004e110, per frame from sub_1400eb630) writes hidden = (row[i][min(frame, frames - 1)] == 0) into the
    model slot's per-mesh hide flags (the same array the mesh tags fill), for i < min(count, the model's meshes)."""
    nf = int(struct.unpack_from('<f', b, 0)[0])
    count = struct.unpack_from('<i', b, 4)[0]
    if nf <= 0 or count <= 0 or 16 + nf * count > len(b):
        raise ValueError('not a .tmv')
    return nf, np.frombuffer(b, np.uint8, count=nf * count, offset=16).reshape(count, nf)


def apply_tmv(objs, rows, start, scale):
    """keys hide_viewport / hide_render of the model's mesh objects (by tmd2 mesh index) where the visibility
    changes; returns the number of meshes that are hidden at some frame"""
    changed = 0
    for o in objs:
        t = getattr(o.data, 'tmd_material', None) if o.type == 'MESH' else None
        if t is None or o.get('tmd_outline_of') or t.mesh_index >= len(rows):
            continue
        row = rows[t.mesh_index]
        if row.min() != 0:
            continue
        changed += 1
        cuts = [0] + [int(f) for f in np.nonzero(np.diff(row.astype(int)))[0] + 1]
        for f in cuts:
            hidden = bool(row[f] == 0)
            o.hide_viewport = o.hide_render = hidden
            o.keyframe_insert('hide_viewport', frame=start + f * scale)
            o.keyframe_insert('hide_render', frame=start + f * scale)
        o.hide_viewport = o.hide_render = bool(row[0] == 0)
    return changed


def motions(data):
    """[(name without .tmo2, Tmo)]: the catalogue's motions (cameras left out)"""
    return [(n[:-5] if n.lower().endswith('.tmo2') else n, t) for n, t in MO.motions_in(data)]


def scan(data):
    """(character motions, object motions, models, cameras) of a catalogue: names"""
    entries = F.parse_cat(data)
    models = [e[0] for e in entries if e[1] == 'tmd']
    mots = [n for n, _ in motions(data)]
    cams = [n for n, _ in camera.cameras_in(data)]
    obj = [n for n in mots if n.endswith('_obj') or n in models]
    char = [n for n in mots if n not in obj]
    return char, obj, models, cams, entries


def _object_motion(model, obj_mots):
    """the object motion that drives a model: <model>, the model name without its last _part, or the only one"""
    for n in obj_mots:
        if n == model or model.startswith(n + '_'):
            return n
    return obj_mots[0] if len(obj_mots) == 1 else None


def import_scene(path, arm=None, fps_mode='SCENE_60', models=True, cameras=True, camera_mode='AIM', pack=True,
                 shading='TOON', full_size=True):
    """returns a list of report lines"""
    with open(path, 'rb') as fh:
        data = fh.read()
    char, obj, model_names, cams, entries = scan(data)
    mots = dict(motions(data))
    scene = bpy.context.scene
    if fps_mode == 'SCENE_60':
        scene.render.fps, scene.render.fps_base = 60, 1.0
    scale = (scene.render.fps / scene.render.fps_base) / anim.GAME_FPS
    start = scene.frame_start
    report, end = [], start
    if arm is not None and char:
        if ik.has_ik(arm) and arm.get('tmd_ik_on', True):
            ik.set_enabled(arm, False, snap=False)
            report.append('IK switched off on %s' % arm.name)
        rig = anim.rig_from_armature(arm)
        if full_size:
            for b in rig.bones:
                if b['hash'] == F.tamsoft_hash('offset') and b.get('adjust') is not None:
                    b['adjust'] = None
                    report.append("%s: the 'offset' body scale is left out so she lands in place" % arm.name)
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)
        n = char[0]
        act = anim.import_motion(arm, rig, n, mots[n], {}, True, scale, start, assign=True)
        end = max(end, start + mots[n].num_frames * scale)
        report.append('%s: %s (%d frames)' % (arm.name, n, mots[n].num_frames))
    elif char:
        report.append('character motion %s not imported: select an imported character first' % ', '.join(char))
    if models:
        for mn in model_names:
            plan = character.scene_plan(data, mn, entries)
            prop_arm, objs = build.build(plan, '', pack, shading, False, False, None)
            line = '%s: %d meshes' % (mn, len(objs))
            mo = _object_motion(mn, obj)
            tmv = next((e for e in entries if e[1] == 'blob' and e[0].lower().endswith('.tmv')
                        and (mo and e[0][:-4] == mo or mn.startswith(e[0][:-4]))), None)
            if tmv is not None:
                try:
                    _, rows = parse_tmv(data[tmv[2]:tmv[2] + tmv[3]])
                    k = apply_tmv(objs, rows, start, scale)
                    line += ', visibility %s (%d of %d meshes switch)' % (tmv[0], k, len(objs))
                except ValueError:
                    pass
            if prop_arm is not None and mo is not None:
                rig = anim.rig_from_armature(prop_arm)
                anim.import_motion(prop_arm, rig, mo, mots[mo], {}, True, scale, start, assign=True)
                end = max(end, start + mots[mo].num_frames * scale)
                line += ', motion %s' % mo
            report.append(line)
    if cameras:
        for cn, ct in camera.cameras_in(data, path):
            cam_data = bpy.data.cameras.new(cn)
            cam = bpy.data.objects.new(cn, cam_data)
            scene.collection.objects.link(cam)
            camera_io.import_camera(cam, ct, cn, camera_mode, float(start), scale)
            scene.camera = cam
            end = max(end, start + ct.num_frames * scale)
            report.append('camera %s: %d frames' % (cn, ct.num_frames))
    scene.frame_start = int(start)
    scene.frame_end = int(np.ceil(end))
    w = scene.rigidbody_world
    if w is not None:
        w.point_cache.frame_end = max(w.point_cache.frame_end, scene.frame_end)
    scene.frame_set(scene.frame_start)
    return report


def match_motions(data, model_names, motion_list):
    """{model: [motion names]}: a motion goes to the model whose skeleton shares the most bone hashes with its
    tracks (ties: the model of the same name)"""
    hashes = {}
    for mn in model_names:
        try:
            m = F.model_from_cat(data, mn)
            hashes[mn] = {b['hash'] for b in m.bones}
        except Exception:
            hashes[mn] = set()
    out = {mn: [] for mn in model_names}
    for n, t in motion_list:
        tr = set(t.hashes)
        best = max(model_names, key=lambda mn: (len(tr & hashes[mn]), n == mn or n.startswith(mn) or mn.startswith(n)))
        if tr & hashes[best]:
            out[best].append(n)
    return out


def import_models(path, which='', with_motions=True, fps_mode='SCENE_60', pack=True, shading='TOON'):
    """every model (or the named ones, comma separated) of a catalogue as armature + meshes with its textures, and
    the catalogue's motions as actions on the model they fit; returns (report lines, armatures)"""
    with open(path, 'rb') as fh:
        data = fh.read()
    entries = F.parse_cat(data)
    names = [e[0] for e in entries if e[1] == 'tmd']
    if not names:
        raise ValueError('%s holds no model' % os.path.basename(path))
    want = [w.strip() for w in which.split(',') if w.strip()]
    if want:
        missing = [w for w in want if w not in names]
        if missing:
            raise ValueError('not in the catalogue: %s (models: %s)' % (', '.join(missing), ', '.join(names)))
        names = want
    mots = motions(data) if with_motions else []
    plan_mots = match_motions(data, names, mots) if mots else {}
    by_name = dict(mots)
    scene = bpy.context.scene
    if mots and fps_mode == 'SCENE_60':
        scene.render.fps, scene.render.fps_base = 60, 1.0
    scale = (scene.render.fps / scene.render.fps_base) / anim.GAME_FPS
    start, end = scene.frame_start, scene.frame_start
    report, arms = [], []
    for mn in names:
        plan = character.scene_plan(data, mn, entries)
        arm, objs = build.build(plan, '', pack, shading, shading == 'TOON', False, None)
        line = '%s: %d meshes, %d bones' % (mn, len(objs), len(plan.bones))
        ms = plan_mots.get(mn, [])
        if arm is not None and ms:
            rig = anim.rig_from_armature(arm)
            acts = [anim.import_motion(arm, rig, n, by_name[n], {}, True, scale, start, assign=False) for n in ms]
            anim.assign_action(arm, acts[0])
            end = max(end, start + by_name[ms[0]].num_frames * scale)
            line += ', %d motion%s (%s)' % (len(ms), '' if len(ms) == 1 else 's', ', '.join(ms[:6]) + (' ...' if len(ms) > 6 else ''))
        if arm is not None:
            arms.append(arm)
        report.append(line)
    if mots:
        unused = [n for n, _ in mots if not any(n in v for v in plan_mots.values())]
        if unused:
            report.append('motions without a matching skeleton: %s' % ', '.join(unused[:8]))
        if end > start:
            scene.frame_end = int(np.ceil(end))
    return report, arms
