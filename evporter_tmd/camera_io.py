import math
import numpy as np
import bpy
import mathutils

from . import camera as C
from .anim import _container, _keys, assign_action

GAME_FPS = 60.0
CONSTRAINT = 'TMD camera aim'


def _to_blender(v):
    return np.asarray(v) @ C.A3.T


def _to_game(v):
    return np.asarray(v) @ C.A3


def _axis_curves(ct, track, start, scale, unit):
    """per Blender axis (frames, values) in metres: x = game x, y = -game z, z = game y"""
    ch = ct.curves.get(track) or [None] * 9
    out = []
    for gi, sign in ((0, 1.0), (2, -1.0), (1, 1.0)):
        c = ch[gi]
        if c is None:
            out.append((np.array([start]), np.array([0.0])))
        else:
            out.append((start + c[0] * scale, c[1] * unit * sign))
    return out


def _key_object(obj, name, curves=None, rot=None, frames=None, id_type='OBJECT'):
    action = bpy.data.actions.new(name)
    new, _ = _container(action, obj, id_type)
    if curves is not None:
        for k, (fr, va) in enumerate(curves):
            _keys(new('location', k, 'Object Transforms'), fr, va)
    if rot is not None:
        for k in range(3):
            _keys(new('rotation_euler', k, 'Object Transforms'), frames, rot[:, k])
    assign_action(obj, action)
    return action


def import_camera(cam_obj, ct, name, mode='AIM', start=1.0, scale=1.0):
    """keys the game camera motion ct (camera.CameraTmo) onto cam_obj; returns a short report"""
    cam = cam_obj.data
    cam.type = 'PERSP'
    cam.sensor_fit = 'VERTICAL'
    cam.sensor_height = ct.film_mm
    cam.sensor_width = ct.film_mm * (ct.block[12] if ct.block[12] > 0 else 16 / 9)
    for c in [c for c in cam_obj.constraints if c.name == CONSTRAINT]:
        cam_obj.constraints.remove(c)
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    nf = ct.num_frames
    if mode == 'AIM':
        _key_object(cam_obj, name, _axis_curves(ct, 'pos', start, scale, ct.unit))
        aim_name = name + '_aim'
        aim = bpy.data.objects.get(aim_name)
        if aim is None or aim.type != 'EMPTY':
            aim = bpy.data.objects.new(aim_name, None)
            aim.empty_display_type = 'SPHERE'
            aim.empty_display_size = 0.1
            (cam_obj.users_collection[0] if cam_obj.users_collection else bpy.context.scene.collection).objects.link(aim)
        _key_object(aim, aim_name, _axis_curves(ct, 'aim', start, scale, ct.unit))
        k = cam_obj.constraints.new('TRACK_TO')
        k.name = CONSTRAINT
        k.target = aim
        k.track_axis = 'TRACK_NEGATIVE_Z'
        k.up_axis = 'UP_Y'
        k.use_target_z = False
    else:
        fr = np.arange(nf + 1, dtype=float)
        eye = _to_blender(ct.vec('pos', fr))
        at = _to_blender(ct.vec('aim', fr))
        rot = np.zeros((len(fr), 3))
        prev = None
        for i in range(len(fr)):
            m = mathutils.Matrix(C.look_rotation(eye[i], at[i]).tolist())
            e = m.to_euler('XYZ', prev) if prev is not None else m.to_euler('XYZ')
            rot[i] = tuple(e)
            prev = e
        frames = start + fr * scale
        _key_object(cam_obj, name, [(frames, eye[:, k]) for k in range(3)], rot, frames)
    foca = (ct.curves.get('foca') or [None] * 9)[2]
    if foca is not None:
        lf = start + foca[0] * scale
        lv = ct.focal(foca[0])
    else:
        lf, lv = np.array([start]), ct.focal(np.array([0.0]))
    cam.lens = float(lv[0])
    action = bpy.data.actions.new(name + '_lens')
    new, _ = _container(action, cam, 'CAMERA')
    _keys(new('lens', 0, 'Camera'), lf, lv)
    assign_action(cam, action)
    cam_obj['tmd_camera'] = name
    cam_obj['tmd_block'] = [float(x) for x in ct.block]
    scene = bpy.context.scene
    scene.frame_start = int(round(start))
    scene.frame_end = int(round(start + nf * scale))
    if ct.block[10] > 0 and ct.block[11] > 0:
        scene.render.resolution_x, scene.render.resolution_y = int(ct.block[11]), int(ct.block[10])
    return '%s: %d frames, tracks in %s, focal %.0f-%.0f mm' % (name, nf, 'cm' if ct.unit < 1 else 'm', lv.min(), lv.max())


def _aim_point(cam_obj, eye, fwd, distance):
    for c in cam_obj.constraints:
        if c.type in ('TRACK_TO', 'DAMPED_TRACK', 'LOCKED_TRACK') and not c.mute and c.influence > 0 and c.target is not None:
            t = c.target
            if getattr(c, 'subtarget', '') and t.type == 'ARMATURE' and c.subtarget in t.pose.bones:
                return np.array(t.matrix_world @ t.pose.bones[c.subtarget].head)
            return np.array(t.matrix_world.translation)
    dof = cam_obj.data.dof
    if dof.focus_object is not None:
        return np.array(dof.focus_object.matrix_world.translation)
    d = dof.focus_distance if dof.use_dof and dof.focus_distance > 0 else distance
    return eye + fwd * d


def vertical_fov(cam, aspect):
    """Blender's vertical field of view of a perspective camera for a render aspect (width / height)"""
    fit = cam.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if aspect >= 1 else 'VERTICAL'
        size = cam.sensor_width
    else:
        size = cam.sensor_width if fit == 'HORIZONTAL' else cam.sensor_height
    a = 2 * math.atan(size / (2 * cam.lens))
    if fit == 'HORIZONTAL':
        return 2 * math.atan(math.tan(a / 2) / aspect)
    return a


def export_camera(cam_obj, path, frame_start, frame_end, timing='TIME', distance=10.0):
    """writes the camera's animation as a game camera motion (.tmo2); returns (frames, keys, max roll in degrees)"""
    if cam_obj.data.type != 'PERSP':
        raise ValueError('only perspective cameras can be exported')
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    step = 1.0 if timing == 'FRAMES' else fps / GAME_FPS
    n = int(np.floor((frame_end - frame_start) / step + 1e-6)) + 1
    if n < 1:
        raise ValueError('empty frame range')
    r = scene.render
    aspect = (r.resolution_x * r.pixel_aspect_x) / max(1e-9, r.resolution_y * r.pixel_aspect_y)
    film = C.FILM_IN * 25.4
    eye = np.zeros((n, 3))
    aim = np.zeros((n, 3))
    focal = np.zeros(n)
    roll = 0.0
    frame0, sub0 = scene.frame_current, scene.frame_subframe
    try:
        for k in range(n):
            t = frame_start + k * step
            f = int(math.floor(t + 1e-6))
            scene.frame_set(f, subframe=float(t - f) if t - f > 1e-6 else 0.0)
            M = np.array(cam_obj.matrix_world)
            R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
            e, fwd = M[:3, 3], -R[:, 2]
            a = _aim_point(cam_obj, e, fwd, distance)
            eye[k], aim[k] = e, a
            L = C.look_rotation(e, a)
            roll = max(roll, math.degrees(math.acos(max(-1.0, min(1.0, float(L[:, 1] @ R[:, 1]))))))
            fy = vertical_fov(cam_obj.data, aspect)
            focal[k] = film / (2 * math.tan(fy / 2))
    finally:
        scene.frame_set(frame0, subframe=sub0)
    ge, ga = _to_game(eye), _to_game(aim)
    block = C.make_block(ge[0], ga[0], focal[0], r.resolution_x, r.resolution_y)
    old = cam_obj.get('tmd_block')
    if old is not None and len(old) == 20:
        block[13], block[14] = float(old[13]), float(old[14])
    data = C.encode(np.arange(n, dtype=float), ge, ga, focal, block)
    with open(path, 'wb') as fh:
        fh.write(data)
    ct = C.CameraTmo(data, path)
    keys = sum(len(c[0]) for chans in ct.curves.values() for c in chans if c is not None)
    return n - 1, keys, roll
