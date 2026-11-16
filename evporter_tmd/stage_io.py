import os
import numpy as np
import bpy

from . import formats as F, character, build, anim, camera, camera_io, scene_io

TOUCH_COLOURS = {'wall': (1.0, 0.25, 0.2, 1.0), 'floor': (0.3, 1.0, 0.35, 1.0), 'switch wall': (1.0, 0.85, 0.15, 1.0),
                 'water': (0.25, 0.6, 1.0, 1.0)}


def _child_collection(parent, coll):
    """move coll (linked to the scene root by build.build) under parent"""
    sc = bpy.context.scene.collection
    if coll.name in sc.children:
        sc.children.unlink(coll)
    if coll.name not in parent.children:
        parent.children.link(coll)


def import_touch(data, entries, parent, name):
    """the collision meshes as wireframe objects (not rendered); returns report lines"""
    coll = bpy.data.collections.new(name + '_collision')
    parent.children.link(coll)
    lines = []
    for e in entries:
        kind = F.touch_kind(e[0])
        if kind is None:
            continue
        try:
            verts, parts = F.parse_touch(data[e[2]:e[2] + e[3]])
        except Exception as ex:
            lines.append('%s: %s' % (e[0], ex))
            continue
        polys = [p for _, ps in parts for p in ps]
        me = bpy.data.meshes.new(e[0])
        me.from_pydata(build.to_blender(verts).tolist(), [], [list(p) for p in polys])
        me.validate(clean_customdata=False)
        obj = bpy.data.objects.new(e[0], me)
        coll.objects.link(obj)
        obj.display_type = 'WIRE'
        obj.show_in_front = False
        obj.hide_render = True
        obj.color = TOUCH_COLOURS.get(kind, (0.9, 0.4, 1.0, 1.0))
        obj['tmd_touch'] = kind
        ids = np.concatenate([np.full(len(ps), pid, np.int32) for pid, ps in parts]) if parts else np.zeros(0, np.int32)
        num = np.concatenate([np.full(len(ps), k, np.int32) for k, (_, ps) in enumerate(parts)]) if parts else np.zeros(0, np.int32)
        if len(me.polygons) == len(ids):
            at = me.attributes.new('tmd_part_id', 'INT', 'FACE')
            at.data.foreach_set('value', ids)
            at = me.attributes.new('tmd_part', 'INT', 'FACE')
            at.data.foreach_set('value', num)
        obj['tmd_entry'] = e[0]
        lines.append('collision %s (%s): %d polygons' % (e[0], kind, len(polys)))
    coll.hide_render = True
    return lines


def import_stage(path, which='', cameras=True, camera_mode='AIM', motions=True, collision=False,
                 fps_mode='SCENE_60', pack=True, shading='TOON', clip_end=3000.0):
    """returns (report lines, camera objects)"""
    with open(path, 'rb') as fh:
        data = fh.read()
    entries = F.parse_cat(data)
    stage_name = os.path.splitext(os.path.basename(path))[0]
    names = [e[0] for e in entries if e[1] == 'tmd']
    want = [w.strip() for w in which.split(',') if w.strip()]
    if want:
        missing = [w for w in want if w not in names]
        if missing:
            raise ValueError('not in the catalogue: %s (models: %s)' % (', '.join(missing), ', '.join(names)))
        names = want
    scene = bpy.context.scene
    if fps_mode == 'SCENE_60':
        scene.render.fps, scene.render.fps_base = 60, 1.0
    scale = (scene.render.fps / scene.render.fps_base) / anim.GAME_FPS
    start = scene.frame_start
    end = start
    parent = bpy.data.collections.new(stage_name)
    scene.collection.children.link(parent)
    parent['tmd_source'] = os.path.abspath(path)
    report, kinds = [], {}
    mots = [(n, t) for n, t in scene_io.motions(data) if t.num_frames > 0 and len(t.hashes)] if motions else []
    for mn in names:
        plan = character.scene_plan(data, mn, entries)
        arm, objs = build.build(plan, '', pack, shading, False, shading == 'TOON', None)
        for o in objs:
            if o.type == 'MESH':
                o.data['tmd_source_path'] = os.path.abspath(path)
        coll = bpy.data.collections.get(plan.name)
        if objs and objs[0].users_collection:
            coll = objs[0].users_collection[0]
        if coll is not None:
            _child_collection(parent, coll)
        for v in plan.materials.values():
            k = v['stage']['kind'] if v.get('stage') else v.get('shader', 'PLAIN')
            kinds[k] = kinds.get(k, 0) + 1
        nv = sum(len(o['positions']) for o in plan.objects)
        line = '%s: %d meshes, %d vertices' % (mn, len(objs), nv)
        if arm is not None and mots:
            fit = scene_io.match_motions(data, [mn], mots).get(mn, [])
            if fit:
                by = dict(mots)
                rig = anim.rig_from_armature(arm)
                acts = [anim.import_motion(arm, rig, n, by[n], {}, True, scale, start, assign=False) for n in fit]
                anim.assign_action(arm, acts[0])
                end = max(end, start + by[fit[0]].num_frames * scale)
                line += ', motion %s' % ', '.join(fit)
        report.append(line)
    report.append('materials: ' + ', '.join('%s x%d' % kv for kv in sorted(kinds.items())))
    cams = []
    if cameras:
        for cn, ct in camera.cameras_in(data, path):
            cam_data = bpy.data.cameras.new(cn)
            cam_data.clip_start = 0.1
            cam_data.clip_end = clip_end
            cam = bpy.data.objects.new(cn, cam_data)
            parent.objects.link(cam)
            camera_io.import_camera(cam, ct, cn, camera_mode, float(start), scale)
            aim = bpy.data.objects.get(cn + '_aim')
            if aim is not None and aim.name not in parent.objects:
                for c in list(aim.users_collection):
                    c.objects.unlink(aim)
                parent.objects.link(aim)
            if scene.camera is None or not cams:
                scene.camera = cam
            cams.append(cam)
            end = max(end, start + ct.num_frames * scale)
            report.append('camera %s: %d frames' % (cn, ct.num_frames))
    if collision:
        report += import_touch(data, entries, parent, stage_name)
    if end > start:
        scene.frame_start = int(start)
        scene.frame_end = int(np.ceil(end))
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D' and sp.clip_end < clip_end:
                        sp.clip_end = clip_end
    return report, cams


def collision_targets(objs):
    """{'floor': object, 'wall': object} of the imported collision meshes for the stage of these objects"""
    stems = set()
    for o in objs:
        t = getattr(o.data, 'tmd_material', None)
        if t is not None and t.model:
            stems.add(t.model.split('_')[0])
    found = {}
    for o in bpy.context.scene.objects:
        kind = o.get('tmd_touch')
        if o.type != 'MESH' or kind not in ('floor', 'wall'):
            continue
        name = o.get('tmd_entry') or o.name
        better = not stems or any(name.startswith(s) for s in stems)
        if kind not in found or (better and not found[kind][1]):
            found[kind] = (o, better)
    return {k: v[0] for k, v in found.items()}


def _common_part(me):
    """(part number, part id) used by most faces of a collision mesh (new faces go there)"""
    at, ai = me.attributes.get('tmd_part'), me.attributes.get('tmd_part_id')
    n = len(me.polygons)
    if at is None or not n:
        return 0, 0
    a = np.zeros(n, np.int32)
    at.data.foreach_get('value', a)
    v, c = np.unique(a, return_counts=True)
    part = int(v[np.argmax(c)])
    pid = 0
    if ai is not None:
        b = np.zeros(n, np.int32)
        ai.data.foreach_get('value', b)
        pid = int(b[np.argmax(a == part)])
    return part, pid


def collision_from_objects(objs, targets, mode='AUTO', slope_deg=45.0, apply_modifiers=True):
    """add the faces of objs (world space, modifiers applied) to the collision objects: with mode AUTO a face whose
    normal is within slope_deg of straight up goes to the floor (touch_yuka), every other face (walls, slopes,
    ceilings) to the walls (touch_kabe); FLOOR / WALL put all faces in one. New faces join the part most faces of
    that collision mesh use. Returns report lines"""
    import bmesh
    from .costume_io import _evaluated_mesh
    cos_lim = np.cos(np.radians(slope_deg))
    add = {'floor': [], 'wall': []}
    for o in objs:
        me, cleanup = _evaluated_mesh(o, apply_modifiers)
        try:
            M = np.array(o.matrix_world)
            nv = len(me.vertices)
            co = np.zeros(nv * 3, np.float32)
            me.vertices.foreach_get('co', co)
            co = co.reshape(nv, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
            flip = np.linalg.det(M[:3, :3]) < 0
            for p in me.polygons:
                vi = list(p.vertices)
                if flip:
                    vi = vi[::-1]
                pts = co[vi]
                n = np.zeros(3)
                for k in range(len(pts)):
                    a, b = pts[k], pts[(k + 1) % len(pts)]
                    n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
                ln = np.linalg.norm(n)
                if ln < 1e-12:
                    continue
                up = n[2] / ln
                kind = 'floor' if mode == 'FLOOR' or (mode == 'AUTO' and up >= cos_lim) else 'wall'
                add[kind].append(pts)
        finally:
            cleanup()
    lines = []
    for kind, polys in add.items():
        if not polys:
            continue
        tgt = targets.get(kind)
        if tgt is None:
            lines.append('%d faces for the %s collision skipped: no %s collision object (Import stage with Collision '
                         'meshes)' % (len(polys), kind, kind))
            continue
        part, pid = _common_part(tgt.data)
        Minv = np.linalg.inv(np.array(tgt.matrix_world))
        bm = bmesh.new()
        bm.from_mesh(tgt.data)
        lp = bm.faces.layers.int.get('tmd_part') or bm.faces.layers.int.new('tmd_part')
        li = bm.faces.layers.int.get('tmd_part_id') or bm.faces.layers.int.new('tmd_part_id')
        added, new_verts = 0, []
        for pts in polys:
            local = pts @ Minv[:3, :3].T + Minv[:3, 3]
            vs = [bm.verts.new(tuple(x)) for x in local]
            new_verts += vs
            try:
                f = bm.faces.new(vs)
            except ValueError:
                for v in vs:
                    bm.verts.remove(v)
                    new_verts.remove(v)
                continue
            f[lp], f[li] = part, pid
            added += 1
        bmesh.ops.remove_doubles(bm, verts=new_verts, dist=1e-5)
        bm.to_mesh(tgt.data)
        bm.free()
        tgt.data.update()
        lines.append('%d faces added to %s (part %d)' % (added, tgt.name, part))
    return lines
