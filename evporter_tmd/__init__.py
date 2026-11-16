bl_info = {
    "name": "EVPorter (SK Estival Versus)",
    "author": "zane0xff",
    "version": (0, 18, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > EVPorter",
    "description": "Import and export playable characters (costume, face, hair, underwear, accessories, skeleton), stages, props, motions and cameras from the game's GameData",
    "category": "Import-Export",
}

if "bpy" in locals():
    import importlib
    from . import formats, character, toon, yure, rigid, motion, anim, ik, camera, camera_io, scene_io, stage, stage_io, catio, tmdwrite, costume, costume_io, build
    for _m in (formats, character, toon, yure, rigid, motion, anim, ik, camera, camera_io, scene_io, stage, stage_io, catio, tmdwrite, costume, costume_io, build):
        importlib.reload(_m)

try:
    import bpy
    from bpy.props import (StringProperty, IntProperty, BoolProperty, EnumProperty, FloatVectorProperty, FloatProperty,
                           CollectionProperty, PointerProperty)
except ImportError:
    bpy = None

if bpy is not None:
    import os
    import time
    from . import formats, character, toon, yure, rigid, motion, anim, ik, camera, camera_io, scene_io, stage, stage_io, catio, tmdwrite, costume, costume_io, build

    class TMD_MaterialEntry(bpy.types.PropertyGroup):
        material: PointerProperty(type=bpy.types.Material, name="Material")
        index: IntProperty(name="Material index", default=-1, description="Index of the material in the tmd2 file")
        fourcc: StringProperty(name="Shader", description="Shader FourCC of the material (e.g. CSB0, SKC0, HRH0)")
        num_params: IntProperty(name="Channels", description="Number of shader constants (material channels)")
        params: FloatVectorProperty(name="Channels", size=32, description="Shader constants ch0..ch31 (m_Param) at frame 0")
        textures: StringProperty(name="Texture slots", description="Texture slots: slot, name, sampler kind")

    class TMD_MeshProps(bpy.types.PropertyGroup):
        model: StringProperty(name="Model", description="tmd2 model name")
        kind: StringProperty(name="Part", description="body, face, hair, pants, accessory or weapon")
        mesh_index: IntProperty(name="Mesh index", description="Index of the mesh node in the tmd2 file")
        tags: StringProperty(name="Layer tags", description="Per-mesh tag string (NTRL BRK0 HIN, INF4 NNJ ORIG, ...)")
        xform_type: IntProperty(name="Transform type", description="0 none, 1 static, 2 animated (3 skinned in practice)")
        mesh_flags: IntProperty(name="Mesh flags", min=0, max=255,
                                description="Mesh +0x1A, written on export. bit0 = 1: back faces culled (most meshes); "
                                            "0: probably two-sided (distant scenery, barriers, costume cloth layers)")
        vertex_flags: IntProperty(name="Vertex flags", description="tmd2 header vertex format flags")
        bounds: FloatVectorProperty(name="Bounds", size=6, description="Bounding box min xyz, max xyz (game space)")
        hidden_reason: StringProperty(name="Hidden", description="Why the game would not draw this mesh in the imported state")
        materials: CollectionProperty(type=TMD_MaterialEntry)
        export: BoolProperty(name="Export", default=True,
                             description="Write this mesh with Export costume (off: the mesh is left out, as if it "
                                         "were not in the scene)")

    def _prefs(context):
        a = context.preferences.addons.get(__package__)
        return a.preferences if a else None

    class TMD_Preferences(bpy.types.AddonPreferences):
        bl_idname = __package__
        gamedata: StringProperty(name="GameData folder", subtype='DIR_PATH',
                                 description="The game's GameData folder (or the game folder that contains it)")
        cache_dir: StringProperty(name="Texture cache", subtype='DIR_PATH',
                                  description="Where the DDS textures are written before Blender loads them (empty: system temp)")

        def draw(self, context):
            self.layout.prop(self, "gamedata")
            self.layout.prop(self, "cache_dir")

    _STATE = {"gd": None, "cos": None, "root": "", "chara_items": [], "costume_items": {}, "names": []}

    def _gamedata(context):
        p = _prefs(context)
        root = bpy.path.abspath(p.gamedata) if p and p.gamedata else ""
        if root and (_STATE["gd"] is None or _STATE["root"] != root):
            gd = formats.GameData(root)
            if gd.ok:
                _STATE.update(gd=gd, root=root, cos=None, costume_items={}, chara_items=[])
                data = gd.read('Binary/Param/Player/cosinfo.bin')
                if data:
                    _STATE["cos"] = formats.CosInfo(data)
                    _STATE["names"] = [gd.asset_name(0x3CD + i).replace('Model/Costume/', '').replace('_00_H.cat', '')
                                       for i in range(_STATE["cos"].ncostume)]
        return _STATE["gd"]

    def _chara_items(self, context):
        cos = _STATE["cos"]
        if cos is None:
            return [("0", "Load the GameData folder first", "")]
        if not _STATE["chara_items"]:
            names = _STATE["names"]
            items = []
            for i, c in enumerate(cos.chars):
                ev, sh = c['costume']
                label = "pl%02d  (%s / %s)" % (i, names[ev] if ev is not None and ev < len(names) else '-',
                                               names[sh] if sh is not None and sh < len(names) else '-')
                items.append((str(i), label, "Playable character %d" % i))
            _STATE["chara_items"] = items
        return _STATE["chara_items"]

    def _costume_items(self, context):
        cos = _STATE["cos"]
        if cos is None:
            return [("OWN0", "-", "")]
        ch = int(self.chara or 0)
        if ch not in _STATE["costume_items"]:
            names = _STATE["names"]
            rec = cos.chars[ch]
            items = []

            def label(cid):
                f = cos.costume(cid)
                return "%3d  %s" % (cid, names[cid] if cid < len(names) else '?') + \
                    ("  (no damage)" if f['no_damage'] else "") + ("  (no underwear)" if f['no_underwear'] else "")
            ev, sh = rec['costume']
            if ev is not None and ev != sh:
                items.append(("OWN0", "Everyday: " + label(ev).strip(), "The character's everyday (uniform) costume"))
            if sh is not None:
                items.append(("OWN1", "Shinobi: " + label(sh).strip(), "The character's shinobi (ninja) costume"))
            for cid in range(cos.ncostume):
                rng = 'Uniform' if cid < 29 else 'Ninja' if cid < 74 else 'FreeCos'
                items.append(("C%03d" % cid, label(cid), "%s costume %d, worn with the character's everyday hair and underwear" % (rng, cid)))
            _STATE["costume_items"][ch] = items
        return _STATE["costume_items"][ch]

    class TMD_MotionItem(bpy.types.PropertyGroup):
        name: StringProperty()
        frames: IntProperty()
        tracks: IntProperty()
        companions: StringProperty(description="Companion motions in the catalogue (_hr hair, _hd head, _wpKK weapons)")
        select: BoolProperty(name="Import", default=False)

    class TMD_SceneProps(bpy.types.PropertyGroup):
        chara: EnumProperty(name="Character", items=_chara_items)
        costume: EnumProperty(name="Outfit", items=_costume_items)
        layer: EnumProperty(name="Outfit layer", default='NTRL', items=[
            ('NTRL', "NTRL intact", ""), ('BRK0', "BRK0 damage 1", ""), ('BRK1', "BRK1 damage 2", ""),
            ('BRK2', "BRK2 damage 3", ""), ('BRK3', "BRK3 damage 4", ""), ('BRKS', "BRKS transformation", "")])
        accessories: BoolProperty(name="Accessories", default=True, description="Default accessories from cosinfo.bin")
        weapons: BoolProperty(name="Weapons", default=False, description="Weapon models of weaponinfo.bin (Model/Playable/wpnNN)")
        proportions: BoolProperty(name="Body proportions", default=True,
                                  description="Apply the character's plbodyscl.bin bone scales to the rest pose")
        hidden_meshes: BoolProperty(name="Hidden layers", default=False,
                                    description="Also import meshes the chosen layer does not show (hidden in the viewport)")
        face_overlays: BoolProperty(name="Face overlays", default=False,
                                    description="Import the face's expression overlay meshes 2-6 (blush, tears, eye shapes) as hidden objects")
        pack: BoolProperty(name="Pack textures", default=True, description="Pack the images into the .blend file")
        shading: EnumProperty(name="Materials", default='TOON', items=[
            ('TOON', "Game shaders", "Node trees emulating the game's shaders: characters (ramp shade, rim, "
                                    "sheen / specular, hair ring) and stages (baked lightmaps, reflections, water), "
                                    "driven by the TMD material parameters"),
            ('PRINCIPLED', "Principled BSDF", "Plain Principled BSDF with the colour and normal maps")])
        outlines: BoolProperty(name="Outlines", default=True,
                               description="Inverted hull objects for the game's outline pass (materials with an _outline texture)")
        ik: BoolProperty(name="IK controls for posing", default=False,
                         description="Add foot / hand IK controls with knee / elbow poles (extra non-deforming bones) "
                                     "for posing; they can be added or removed later in the Posing panel")
        physics: BoolProperty(name="Physics (rigid bodies)", default=True,
                              description="Rigid bodies from the game's yure / collision files: skirt (yure_cate_*), hair, "
                                          "costume parts and bust, with the body collision shapes")
        phys_radius: FloatProperty(name="Particle radius (cm)", default=2.0, min=0.2, max=6.0,
                                   description="Collision radius of the chain particles; about the cloth thickness "
                                               "(2 cm matched the game's solver best in tests)")
        phys_stiffness: FloatProperty(name="Stiffness", default=1.0, min=0.0, max=20.0,
                                      description="Scale of the joint rotation springs (1 = the game's spring / gravity balance)")
        phys_bust: BoolProperty(name="Bust", default=True, description="Sprung bust bodies from BustInfo.bin")
        view_standard: BoolProperty(name="Standard view transform", default=True,
                                    description="Set the scene's view transform to Standard so the toon colours show as in the game")
        status: StringProperty()
        motion_path: StringProperty(name="Motion file", subtype='FILE_PATH',
                                    description="A motion catalogue (Motion/Player/plNN_act.cat, Motion/Adv/Adv_Motion.cat, "
                                                "Motion/Room/Dress_Motion.cat ...) or a single .tmo2")
        motions: CollectionProperty(type=TMD_MotionItem)
        motion_index: IntProperty()
        motion_companions: BoolProperty(name="Companion motions", default=True,
                                        description="Also apply <motion>_hr (hair), _hd (head) and _wpKK (weapon) motions "
                                                    "of the catalogue to those parts")
        motion_show_companions: BoolProperty(name="List companions", default=False,
                                             description="Show the companion motions in the list too")
        motion_root: BoolProperty(name="Root motion", default=True,
                                  description="Keep the character's travel (off: she stays in place, height kept)")
        motion_fps: EnumProperty(name="Timing", default='SCENE_60', items=[
            ('SCENE_60', "Scene to 60 fps", "Set the scene to 60 fps: one key per game frame"),
            ('RESAMPLE', "Keep scene fps", "Keep the scene frame rate and place the game frames at their time")])
        motion_status: StringProperty()

    class TMD_OT_load_gamedata(bpy.types.Operator):
        bl_idname = "evporter.load_gamedata"
        bl_label = "Load GameData"
        bl_description = "Read cosinfo.bin and the file table from the GameData folder"

        def execute(self, context):
            _STATE.update(gd=None, root="")
            gd = _gamedata(context)
            s = context.scene.evporter
            if gd is None:
                s.status = "No Fnames/E/filename.bin below that folder"
                self.report({'ERROR'}, s.status)
                return {'CANCELLED'}
            cos = _STATE["cos"]
            if cos is None:
                s.status = "Binary/Param/Player/cosinfo.bin not found"
                self.report({'ERROR'}, s.status)
                return {'CANCELLED'}
            s.status = "cosinfo.bin: %d characters, %d costumes" % (cos.nchara, cos.ncostume)
            self.report({'INFO'}, s.status)
            return {'FINISHED'}

    class TMD_OT_import_character(bpy.types.Operator):
        bl_idname = "evporter.import_character"
        bl_label = "Import character"
        bl_description = "Import the character in the chosen outfit: skeleton, body, face, hair, underwear, accessories"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            gd = _gamedata(context)
            s = context.scene.evporter
            if gd is None or _STATE["cos"] is None:
                self.report({'ERROR'}, "Set the GameData folder and press Load GameData first")
                return {'CANCELLED'}
            t0 = time.time()
            ch = int(s.chara)
            rec = _STATE["cos"].chars[ch]
            side = None
            if s.costume.startswith("OWN"):
                side = int(s.costume[3])
                cid = rec['costume'][side]
            else:
                cid = int(s.costume[1:])
            try:
                c = character.Character(gd, ch)
                plan = c.build(cid, side=side, layer=s.layer, accessories=s.accessories, weapons=s.weapons,
                               proportions=s.proportions, hidden_meshes=s.hidden_meshes, face_overlays=s.face_overlays)
                p = _prefs(context)
                cache = bpy.path.abspath(p.cache_dir) if p and p.cache_dir else ''
                rig = None
                if s.physics:
                    rig = yure.build_rig(gd, plan, radius=s.phys_radius * 0.01, stiffness=s.phys_stiffness, bust=s.phys_bust)
                    for line in rig.files + rig.notes:
                        print('evporter physics:', line)
                arm, objs = build.build(plan, cache, s.pack, s.shading, s.outlines, s.view_standard, rig)
                if s.ik and arm is not None:
                    limbs, ik_notes = ik.setup(arm, context)
                    plan.notes.extend('IK: ' + n for n in ik_notes)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            msg = "%s: %d bones, %d meshes, %d materials in %.1f s" % (plan.name, len(plan.bones), len(objs),
                                                                         len(plan.materials), time.time() - t0)
            for n in plan.notes:
                print('evporter:', n)
            s.status = msg + ("  (%d notes in the console)" % len(plan.notes) if plan.notes else "")
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    class TMD_OT_light_from_sun(bpy.types.Operator):
        bl_idname = "evporter.light_from_sun"
        bl_label = "Light from active Sun"
        bl_description = "Aim the toon light (TMD Globals) along the active light object (a Sun lamp shines along its -Z)"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            o = context.active_object
            return o is not None and o.type == 'LIGHT'

        def execute(self, context):
            import mathutils
            m = context.active_object.matrix_world
            d = (m.to_3x3() @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()
            toon.set_light(tuple(d))
            return {'FINISHED'}

    class TMD_OT_reset_light(bpy.types.Operator):
        bl_idname = "evporter.reset_light"
        bl_label = "Game light"
        bl_description = "The game's character light direction (m_LightDirection)"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            toon.set_light(toon.LIGHT)
            return {'FINISHED'}

    def _active_armature(context):
        o = context.active_object
        while o is not None and o.type != 'ARMATURE':
            o = o.parent
        return o

    class TMD_OT_motion_open(bpy.types.Operator):
        bl_idname = "evporter.motion_open"
        bl_label = "Open motion file"
        bl_description = "Choose a motion catalogue (.cat) or a .tmo2 file and list its motions"
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat;*.tmo2;*.tmo", options={'HIDDEN'})

        def invoke(self, context, event):
            s = context.scene.evporter
            if s.motion_path:
                self.filepath = bpy.path.abspath(s.motion_path)
            else:
                gd = _gamedata(context)
                if gd is not None:
                    self.filepath = os.path.join(gd.root, 'Motion', '')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def execute(self, context):
            context.scene.evporter.motion_path = self.filepath
            return bpy.ops.evporter.motion_read()

    class TMD_OT_motion_read(bpy.types.Operator):
        bl_idname = "evporter.motion_read"
        bl_label = "Read motions"
        bl_description = "List the motions of the motion file"

        def execute(self, context):
            s = context.scene.evporter
            path = bpy.path.abspath(s.motion_path)
            try:
                mots = anim.catalogue(path)
            except (OSError, ValueError) as e:
                s.motion_status = str(e)
                self.report({'ERROR'}, s.motion_status)
                return {'CANCELLED'}
            names = {n for n, _ in mots}
            s.motions.clear()
            for n, m in mots:
                it = s.motions.add()
                it.name, it.frames, it.tracks = n, max(0, m.num_frames), max(0, m.num_tracks)
                it.companions = ' '.join(x for x in motion.COMPANIONS if n + x in names)
            s.motion_index = 0
            ncomp = sum(1 for n, _ in mots if motion.is_companion(n))
            s.motion_status = "%d motions (%d companions) in %s" % (len(mots), ncomp, os.path.basename(path))
            return {'FINISHED'}

    class TMD_OT_motion_select(bpy.types.Operator):
        bl_idname = "evporter.motion_select"
        bl_label = "Select motions"
        bl_description = "Tick or untick every listed motion"
        state: BoolProperty(default=True)

        def execute(self, context):
            s = context.scene.evporter
            for it in s.motions:
                if s.motion_show_companions or not motion.is_companion(it.name):
                    it.select = self.state
            return {'FINISHED'}

    class TMD_OT_motion_import(bpy.types.Operator):
        bl_idname = "evporter.motion_import"
        bl_label = "Import motions"
        bl_description = ("Import the ticked motions (or the highlighted one) as actions on the selected character's "
                          "armature; the last one is assigned")
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return _active_armature(context) is not None and len(context.scene.evporter.motions) > 0

        def execute(self, context):
            s = context.scene.evporter
            arm = _active_armature(context)
            names = [it.name for it in s.motions if it.select]
            if not names and 0 <= s.motion_index < len(s.motions):
                names = [s.motions[s.motion_index].name]
            if not names:
                self.report({'ERROR'}, "No motion chosen")
                return {'CANCELLED'}
            ik_off = ik.has_ik(arm) and arm.get('tmd_ik_on', True)
            if ik_off:
                ik.set_enabled(arm, False, snap=False, context=context)
            t0 = time.time()
            try:
                actions, rig = anim.import_motions(arm, bpy.path.abspath(s.motion_path), names, s.motion_companions,
                                                   s.motion_root, s.motion_fps)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            msg = "%d action(s) on %s in %.1f s" % (len(actions), arm.name, time.time() - t0)
            if ik_off:
                msg += "; IK switched off (Posing panel)"
            if rig.missing_rest:
                msg += " (%d bones without import data: re-import the character for exact poses)" % rig.missing_rest
            if arm.get('tmd_physics_collection'):
                rigid.reset_cache(context.scene)
            s.motion_status = msg
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    class TMD_OT_motion_export(bpy.types.Operator):
        bl_idname = "evporter.motion_export"
        bl_label = "Export motion (.tmo2)"
        bl_description = ("Write the selected character's animation (its action, or the scene range) as a game motion "
                          "file (.tmo2, 60 fps, float keys like the game's own motions)")
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.tmo2", options={'HIDDEN'})
        frames: EnumProperty(name="Frames", default='ACTION', items=(
            ('ACTION', "Action range", "The active action's frame range"),
            ('SCENE', "Scene range", "The scene's start / end frames")))
        timing: EnumProperty(name="Timing", default='TIME', items=(
            ('TIME', "By time (60 fps)", "Sample at the game's 60 fps, whatever the scene frame rate"),
            ('FRAMES', "One scene frame = one game frame", "Take every scene frame as one game frame")))
        visual: BoolProperty(name="Visual pose (constraints, IK)", default=False,
                             description="Write the evaluated pose including bone constraints; otherwise the keyed pose")
        physics: BoolProperty(name="Bake physics bones", default=False,
                              description="With Visual pose: keep what the rigid body simulation does to the skirt / "
                                          "hair / bust bones (the game simulates them itself)")
        split: BoolProperty(name="Hair / face as _hr / _hd files", default=True,
                            description="Hair and face bones go to <name>_hr.tmo2 / <name>_hd.tmo2 companion motions "
                                        "(as the game's files) when they move; otherwise into the main file")
        all_bones: BoolProperty(name="Track for every bone", default=True,
                                description="Write a track for every bone (as the game's files); otherwise only for the "
                                            "bones that move")

        @classmethod
        def poll(cls, context):
            return _active_armature(context) is not None

        def invoke(self, context, event):
            arm = _active_armature(context)
            act = arm.animation_data.action if arm.animation_data else None
            name = bpy.path.clean_name(act.get('tmd_motion', act.name) if act else arm.name)
            folder = os.path.dirname(bpy.path.abspath(context.scene.evporter.motion_path or '')) or os.path.dirname(bpy.data.filepath)
            self.filepath = os.path.join(folder, name + '.tmo2')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def draw(self, context):
            col = self.layout.column()
            col.prop(self, "frames")
            col.prop(self, "timing")
            col.prop(self, "visual")
            sub = col.column()
            sub.enabled = self.visual
            sub.prop(self, "physics")
            col.prop(self, "split")
            col.prop(self, "all_bones")

        def execute(self, context):
            arm = _active_armature(context)
            scene = context.scene
            act = arm.animation_data.action if arm.animation_data else None
            if self.frames == 'ACTION' and act is not None:
                f0, f1 = act.frame_range
            else:
                f0, f1 = scene.frame_start, scene.frame_end
            path = bpy.path.ensure_ext(bpy.path.abspath(self.filepath), '.tmo2')
            if context.object and context.object.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            t0 = time.time()
            try:
                written = anim.export_motion(arm, path, float(f0), float(f1), self.timing, self.visual, self.physics,
                                             self.split, self.all_bones)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            msg = ", ".join("%s (%d frames, %d tracks)" % (os.path.basename(p), n, t) for p, n, t in written)
            msg += " in %.1f s" % (time.time() - t0)
            scene.evporter.motion_status = msg
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    class TMD_UL_motions(bpy.types.UIList):
        def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
            row = layout.row(align=True)
            row.prop(item, "select", text="")
            row.label(text=item.name, icon='ACTION')
            sub = row.row()
            sub.alignment = 'RIGHT'
            sub.label(text=("%d f" % item.frames) + ("  +" + item.companions.replace('_', '') if item.companions else ""))

        def filter_items(self, context, data, propname):
            items = getattr(data, propname)
            flt = [self.bitflag_filter_item] * len(items)
            show = context.scene.evporter.motion_show_companions
            pat = self.filter_name.lower()
            for i, it in enumerate(items):
                if (not show and motion.is_companion(it.name)) or (pat and pat not in it.name.lower()):
                    flt[i] = 0
            return flt, []

    def _tmd_armature(context):
        o = context.active_object
        while o is not None and o.type != 'ARMATURE':
            o = o.parent
        return o if o is not None and o.get('tmd_physics_collection') else None

    class TMD_OT_physics_enable(bpy.types.Operator):
        bl_idname = "evporter.physics_enable"
        bl_label = "Physics on / off"
        bl_description = "Switch the character's rigid body physics (bone constraints and the scene's rigid body world)"
        bl_options = {'REGISTER', 'UNDO'}
        enable: BoolProperty(default=True)

        @classmethod
        def poll(cls, context):
            return _tmd_armature(context) is not None

        def execute(self, context):
            arm = _tmd_armature(context)
            rigid.set_enabled(arm, self.enable)
            arm['tmd_physics_on'] = self.enable
            context.scene.frame_set(context.scene.frame_start)
            return {'FINISHED'}

    class TMD_OT_physics_reset(bpy.types.Operator):
        bl_idname = "evporter.physics_reset"
        bl_label = "Reset to current pose"
        bl_description = ("Restart the simulation from the character's current pose: go to the start frame and "
                          "rebuild the rigid bodies there (also done automatically whenever the start frame is shown)")
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            sc = context.scene
            w = sc.rigidbody_world
            if w is None:
                return {'CANCELLED'}
            rigid.reset_cache(sc)
            sc.frame_set(w.point_cache.frame_start)
            return {'FINISHED'}

    class TMD_OT_physics_remove(bpy.types.Operator):
        bl_idname = "evporter.physics_remove"
        bl_label = "Remove physics"
        bl_description = "Delete the character's rigid bodies, constraints and the bone constraints that follow them"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return _tmd_armature(context) is not None

        def execute(self, context):
            rigid.remove(_tmd_armature(context))
            return {'FINISHED'}

    class TMD_PT_main(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "EV Porter"

        def draw(self, context):
            pass

    class TMD_PT_config(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Game data"
        bl_parent_id = "TMD_PT_main"

        def draw(self, context):
            p = _prefs(context)
            col = self.layout.column()
            if p:
                col.prop(p, "gamedata", text="")
            col.operator("evporter.load_gamedata", icon='FILE_REFRESH')
            if context.scene.evporter.status:
                col.label(text=context.scene.evporter.status)

    class TMD_PT_import(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Import character"
        bl_parent_id = "TMD_PT_main"

        def draw(self, context):
            s = context.scene.evporter
            col = self.layout.column()
            if _STATE["cos"] is None:
                col.label(text="Load the GameData folder first", icon='INFO')
                return
            col.prop(s, "chara")
            col.prop(s, "costume")
            col.prop(s, "layer")
            g = col.column(heading="Include", align=True)
            g.prop(s, "accessories")
            g.prop(s, "weapons")
            g.prop(s, "proportions")
            g.prop(s, "hidden_meshes")
            g.prop(s, "face_overlays")
            g.prop(s, "pack")
            col.prop(s, "shading")
            if s.shading == 'TOON':
                g = col.column(align=True)
                g.prop(s, "outlines")
                g.prop(s, "view_standard")
            col.prop(s, "ik")
            col.prop(s, "physics")
            if s.physics:
                g = col.column(align=True)
                g.prop(s, "phys_radius")
                g.prop(s, "phys_stiffness")
                g.prop(s, "phys_bust")
            col.operator("evporter.import_character", icon='IMPORT')

    class TMD_PT_motion(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Motions"
        bl_parent_id = "TMD_PT_main"

        def draw(self, context):
            s = context.scene.evporter
            col = self.layout.column()
            row = col.row(align=True)
            row.prop(s, "motion_path", text="")
            row.operator("evporter.motion_open", text="", icon='FILEBROWSER')
            row.operator("evporter.motion_read", text="", icon='FILE_REFRESH')
            if s.motion_status:
                col.label(text=s.motion_status)
            if len(s.motions):
                col.template_list("TMD_UL_motions", "", s, "motions", s, "motion_index", rows=8)
                r = col.row(align=True)
                r.operator("evporter.motion_select", text="All").state = True
                r.operator("evporter.motion_select", text="None").state = False
                r.prop(s, "motion_show_companions", toggle=True)
            g = col.column(align=True)
            g.prop(s, "motion_companions")
            g.prop(s, "motion_root")
            g.prop(s, "motion_fps", text="")
            arm = _active_armature(context)
            col.label(text=("Target: " + arm.name) if arm else "Select the character's armature", icon='ARMATURE_DATA')
            col.operator("evporter.motion_import", icon='IMPORT')
            if arm is not None and arm.animation_data is not None:
                col.prop(arm.animation_data, "action", text="Action")
            col.operator("evporter.motion_export", icon='EXPORT')

    class TMD_OT_ik_setup(bpy.types.Operator):
        bl_idname = "evporter.ik_setup"
        bl_label = "Add IK controls"
        bl_description = ("Add foot / hand IK controls with knee / elbow poles to the selected character (extra "
                          "non-deforming bones in the bone collection 'IK controls')")
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            return _active_armature(context) is not None

        def execute(self, context):
            arm = _active_armature(context)
            try:
                limbs, notes = ik.setup(arm, context)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            for n in notes:
                print('evporter IK:', n)
            self.report({'WARNING'} if notes else {'INFO'}, "IK on %s: %s%s" % (
                arm.name, ", ".join(limbs) or "no limbs found", ("  (%d notes in the console)" % len(notes)) if notes else ""))
            return {'FINISHED'}

    class TMD_OT_ik_remove(bpy.types.Operator):
        bl_idname = "evporter.ik_remove"
        bl_label = "Remove IK controls"
        bl_description = "Remove the IK controls and constraints (the game bones keep their current keys)"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            a = _active_armature(context)
            return a is not None and ik.has_ik(a)

        def execute(self, context):
            n = ik.remove(_active_armature(context), context)
            self.report({'INFO'}, "%d IK bones removed" % n)
            return {'FINISHED'}

    class TMD_OT_ik_enable(bpy.types.Operator):
        bl_idname = "evporter.ik_enable"
        bl_label = "IK on / off"
        bl_description = ("Switch the IK constraints on (the controls first move to the current pose, so nothing "
                          "jumps) or off (the keyed / imported pose plays as is)")
        bl_options = {'REGISTER', 'UNDO'}
        enable: BoolProperty(default=True)

        @classmethod
        def poll(cls, context):
            a = _active_armature(context)
            return a is not None and ik.has_ik(a)

        def execute(self, context):
            ik.set_enabled(_active_armature(context), self.enable, snap=True, context=context)
            return {'FINISHED'}

    class TMD_OT_ik_match(bpy.types.Operator):
        bl_idname = "evporter.ik_match"
        bl_label = "Match controls to pose"
        bl_description = "Move the foot / hand controls and poles to where the limbs are without IK"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            a = _active_armature(context)
            return a is not None and ik.has_ik(a)

        def execute(self, context):
            ik.match_controls(_active_armature(context), context)
            return {'FINISHED'}

    class TMD_PT_ik(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Posing (IK)"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            arm = _active_armature(context)
            if arm is None:
                col.label(text="Select an imported character", icon='INFO')
                return
            if not ik.has_ik(arm):
                col.operator("evporter.ik_setup", icon='CON_KINEMATIC')
                return
            on = arm.get('tmd_ik_on', True)
            col.label(text="%s: IK %s" % (arm.name, "on" if on else "off"))
            r = col.row(align=True)
            r.operator("evporter.ik_enable", text="On", icon='CON_KINEMATIC', depress=on).enable = True
            r.operator("evporter.ik_enable", text="Off", icon='BONE_DATA', depress=not on).enable = False
            col.operator("evporter.ik_match", icon='SNAP_ON')
            col.label(text="Pose the IK_* bones in Pose Mode", icon='INFO')
            col.label(text="Export IK poses with 'Visual pose'", icon='INFO')
            col.operator("evporter.ik_remove", icon='TRASH')

    def _active_camera(context):
        o = context.active_object
        return o if o is not None and o.type == 'CAMERA' else None

    class TMD_OT_camera_import(bpy.types.Operator):
        bl_idname = "evporter.camera_import"
        bl_label = "Import camera motion"
        bl_description = ("Key a game camera motion (a stage .cat such as bg06.cat, or a camera .tmo2) onto the "
                          "selected camera")
        bl_options = {'REGISTER', 'UNDO'}
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat;*.tmo2;*.tmo", options={'HIDDEN'})
        which: StringProperty(name="Camera", default="",
                              description="Camera motion name inside a catalogue (empty: the first one)")
        mode: EnumProperty(name="Keys", default='AIM', items=(
            ('AIM', "Aim empty (game keys)", "Eye keys on the camera, look-at keys on an empty, Track To constraint: "
                                            "the game's own keys and cuts"),
            ('BAKE', "Bake every frame", "Location and rotation keyed on every frame, no extra objects")))
        timing: EnumProperty(name="Timing", default='SCENE_60', items=(
            ('SCENE_60', "Scene to 60 fps", "Set the scene to 60 fps: one scene frame per game frame"),
            ('RESAMPLE', "Keep scene fps", "Keep the scene frame rate and place the game frames at their time")))

        @classmethod
        def poll(cls, context):
            return _active_camera(context) is not None

        def invoke(self, context, event):
            gd = _gamedata(context)
            if gd is not None and not self.filepath:
                self.filepath = os.path.join(gd.root, '')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def draw(self, context):
            col = self.layout.column()
            col.prop(self, "which")
            col.prop(self, "mode")
            col.prop(self, "timing")

        def execute(self, context):
            cam = _active_camera(context)
            try:
                with open(bpy.path.abspath(self.filepath), 'rb') as fh:
                    data = fh.read()
                found = camera.cameras_in(data, self.filepath)
                if not found:
                    raise ValueError("no camera motion in %s" % os.path.basename(self.filepath))
                pick = [x for x in found if x[0] == self.which] or found[:1] if self.which else found[:1]
                name, ct = pick[0]
                scene = context.scene
                if self.timing == 'SCENE_60':
                    scene.render.fps, scene.render.fps_base = 60, 1.0
                fps = scene.render.fps / scene.render.fps_base
                msg = camera_io.import_camera(cam, ct, name, self.mode, float(scene.frame_start or 1), fps / 60.0)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if len(found) > 1:
                msg += "  (in the file: %s)" % ", ".join(n for n, _ in found)
            context.scene.evporter.motion_status = msg
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    class TMD_OT_camera_export(bpy.types.Operator):
        bl_idname = "evporter.camera_export"
        bl_label = "Export camera motion (.tmo2)"
        bl_description = ("Write the selected camera's animation as a game camera motion (.tmo2: eye, look-at point, "
                          "focal length at 60 fps)")
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.tmo2", options={'HIDDEN'})
        frames: EnumProperty(name="Frames", default='SCENE', items=(
            ('SCENE', "Scene range", "The scene's start / end frames"),
            ('ACTION', "Action range", "The camera's action frame range")))
        timing: EnumProperty(name="Timing", default='TIME', items=(
            ('TIME', "By time (60 fps)", "Sample at the game's 60 fps, whatever the scene frame rate"),
            ('FRAMES', "One scene frame = one game frame", "Take every scene frame as one game frame")))
        distance: FloatProperty(name="Look-at distance", default=10.0, min=0.01, unit='LENGTH',
                                description="Without a Track To target or focus object: the look-at point this far "
                                            "ahead (or the depth-of-field focus distance when DoF is on)")

        @classmethod
        def poll(cls, context):
            return _active_camera(context) is not None

        def invoke(self, context, event):
            cam = _active_camera(context)
            name = bpy.path.clean_name(cam.get('tmd_camera', 'cam_' + cam.name))
            folder = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ''
            self.filepath = os.path.join(folder, name + '.tmo2')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def execute(self, context):
            cam = _active_camera(context)
            scene = context.scene
            act = cam.animation_data.action if cam.animation_data else None
            if self.frames == 'ACTION' and act is not None:
                f0, f1 = act.frame_range
            else:
                f0, f1 = scene.frame_start, scene.frame_end
            path = bpy.path.ensure_ext(bpy.path.abspath(self.filepath), '.tmo2')
            try:
                n, keys, roll = camera_io.export_camera(cam, path, float(f0), float(f1), self.timing, self.distance)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            msg = "%s: %d frames, %d keys" % (os.path.basename(path), n, keys)
            if roll > 0.5:
                msg += "; the camera differs from a level look-at view by up to %.1f deg (the game has no roll)" % roll
            scene.evporter.motion_status = msg
            self.report({'WARNING'} if roll > 0.5 else {'INFO'}, msg)
            return {'FINISHED'}

    class TMD_OT_model_import(bpy.types.Operator):
        bl_idname = "evporter.model_import"
        bl_label = "Import model (.cat)"
        bl_description = ("Import the models of any catalogue (enemies Model/Zako/enNN_H.cat, objects Model/Obj, props) "
                          "as armatures with meshes and textures; motions inside the catalogue become actions")
        bl_options = {'REGISTER', 'UNDO'}
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat", options={'HIDDEN'})
        which: StringProperty(name="Models", default="",
                              description="Model names inside the catalogue, comma separated (empty: all)")
        with_motions: BoolProperty(name="Motions in the catalogue", default=True,
                                   description="Import the catalogue's motions as actions on the model whose skeleton "
                                               "they fit")
        timing: EnumProperty(name="Timing", default='SCENE_60', items=(
            ('SCENE_60', "Scene to 60 fps", "Set the scene to 60 fps: one scene frame per game frame"),
            ('RESAMPLE', "Keep scene fps", "Keep the scene frame rate and place the game frames at their time")))

        def invoke(self, context, event):
            gd = _gamedata(context)
            if gd is not None and not self.filepath:
                self.filepath = os.path.join(gd.root, 'Model', '')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def execute(self, context):
            s = context.scene.evporter
            t0 = time.time()
            try:
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                lines, arms = scene_io.import_models(bpy.path.abspath(self.filepath), self.which, self.with_motions,
                                                     self.timing, s.pack, s.shading)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if arms:
                for o in context.view_layer.objects:
                    o.select_set(False)
                for a in arms:
                    a.select_set(True)
                context.view_layer.objects.active = arms[-1]
            for line in lines:
                print('evporter model:', line)
            msg = "; ".join(lines) + " (%.1f s)" % (time.time() - t0)
            s.motion_status = msg
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    class TMD_OT_scene_import(bpy.types.Operator):
        bl_idname = "evporter.scene_import"
        bl_label = "Import cut-scene (.cat)"
        bl_description = ("Import a cut-scene catalogue such as Placement/plbg/plbg_dieXX.cat: its character motion on "
                          "the selected character, its models with their motions, and its camera")
        bl_options = {'REGISTER', 'UNDO'}
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat", options={'HIDDEN'})
        models: BoolProperty(name="Scene models", default=True, description="Import the catalogue's models and their motions")
        cameras: BoolProperty(name="Camera", default=True, description="Import the camera motion on a new camera (the scene camera)")
        camera_mode: EnumProperty(name="Camera keys", default='AIM', items=(
            ('AIM', "Aim empty (game keys)", "Eye keys on the camera, look-at keys on an empty, Track To constraint"),
            ('BAKE', "Bake every frame", "Location and rotation keyed on every frame")))
        timing: EnumProperty(name="Timing", default='SCENE_60', items=(
            ('SCENE_60', "Scene to 60 fps", "Set the scene to 60 fps: one scene frame per game frame"),
            ('RESAMPLE', "Keep scene fps", "Keep the scene frame rate and place the game frames at their time")))
        full_size: BoolProperty(name="Place at the scene's spot", default=True,
                                description="Leave out the character's overall body scale (plbodyscl 'offset', e.g. pl00 "
                                            "x 0.94) for this motion: the cut-scene places her in absolute world units, "
                                            "which that scale would shrink (pl00 lands 10 m short in die46)")

        def invoke(self, context, event):
            gd = _gamedata(context)
            if gd is not None and not self.filepath:
                self.filepath = os.path.join(gd.root, 'Placement', 'plbg', '')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def draw(self, context):
            col = self.layout.column()
            arm = _active_armature(context)
            col.label(text=("Character: " + arm.name) if arm else "No character selected: motion skipped",
                      icon='ARMATURE_DATA' if arm else 'INFO')
            col.prop(self, "models")
            col.prop(self, "cameras")
            sub = col.column()
            sub.enabled = self.cameras
            sub.prop(self, "camera_mode")
            col.prop(self, "timing")
            col.prop(self, "full_size")

        def execute(self, context):
            arm = _active_armature(context)
            s = context.scene.evporter
            t0 = time.time()
            try:
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                lines = scene_io.import_scene(bpy.path.abspath(self.filepath), arm, self.timing, self.models,
                                              self.cameras, self.camera_mode, s.pack, s.shading, self.full_size)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            for line in lines:
                print('evporter scene:', line)
            msg = "; ".join(lines) + " (%.1f s)" % (time.time() - t0)
            s.motion_status = msg
            self.report({'WARNING'} if any('not imported' in l for l in lines) else {'INFO'}, msg)
            return {'FINISHED'}

    def _model_sources(context, arm):
        """[(GameData-relative catalogue, number of objects)] of the character's meshes, most objects first"""
        count = {}
        for o in costume_io.character_objects(arm):
            src = o.data.get('tmd_source', '')
            if src:
                count[src] = count.get(src, 0) + 1
        return sorted(count.items(), key=lambda kv: -kv[1])

    _TEMPLATE_ITEMS = []

    def _template_items(self, context):
        """the imported meshes (TMD data, not outlines) as templates"""
        _TEMPLATE_ITEMS.clear()
        for o in sorted(context.scene.objects, key=lambda o: o.name):
            if o.type == 'MESH' and not o.get('tmd_outline_of') and o.data.tmd_material.model:
                t = o.data.tmd_material
                _TEMPLATE_ITEMS.append((o.name, o.name, "%s mesh %d: %s" % (t.model, t.mesh_index, t.tags or '(no tags)')))
        if not _TEMPLATE_ITEMS:
            _TEMPLATE_ITEMS.append(('', "(no imported meshes)", ""))
        return _TEMPLATE_ITEMS

    class TMD_OT_mesh_setup(bpy.types.Operator):
        bl_idname = "evporter.mesh_setup"
        bl_label = "Make costume mesh"
        bl_description = ("Give the selected mesh(es) the TMD data of an imported mesh (the template), so Export "
                          "costume writes them as extra meshes: model, layer tags, materials, armature, bone weights, "
                          "UV maps and colour attribute")
        bl_options = {'REGISTER', 'UNDO'}
        template: EnumProperty(name="Like", items=_template_items,
                               description="Imported mesh to copy the TMD data from (model, mesh record, materials)")
        ntrl: BoolProperty(name="NTRL", default=True, description="Intact costume")
        brk0: BoolProperty(name="BRK0", default=False, description="Damage stage 0")
        brk1: BoolProperty(name="BRK1", default=False, description="Damage stage 1")
        brk2: BoolProperty(name="BRK2", default=False, description="Damage stage 2")
        brk3: BoolProperty(name="BRK3", default=False, description="Damage stage 3")
        brks: BoolProperty(name="BRKS", default=False, description="Transformation (CSBB materials)")
        bust: EnumProperty(name="Bust", default='BOTH', items=(
            ('BOTH', "Both sizes", "No BAKU / HIN tag: drawn for every character"),
            ('BAKU', "Big (BAKU)", "Drawn for big-bust characters only"),
            ('HIN', "Small (HIN)", "Drawn for small-bust characters only (pl08, pl18 everyday)")))
        other: StringProperty(name="Other tags", default="", description="Further tokens kept as they are (INF4 ...)")
        materials: EnumProperty(name="Materials", default='ADD', items=(
            ('ADD', "Add the template's", "Keep the mesh's game materials, add the template's (faces on non-game "
                                          "materials export with the first game material)"),
            ('REPLACE', "Replace with the template's", "Only the template's materials, every face on the first")))
        weights: BoolProperty(name="Transfer bone weights", default=True,
                              description="Bone weights from the template's nearest faces (rest pose); off: keep or "
                                          "paint your own (bone vertex groups)")
        color: BoolProperty(name="Add colour attribute", default=True,
                            description="If the mesh has no 'Col' attribute: the template's most common vertex colour "
                                        "(red = highlight mask)")

        @classmethod
        def poll(cls, context):
            return any(o.type == 'MESH' for o in context.selected_objects)

        def _targets(self, context):
            return [o for o in context.selected_objects if o.type == 'MESH' and o.name != self.template
                    and not o.get('tmd_outline_of')]

        def _load_tags(self, tags):
            st, bust, other = costume_io.split_tags(tags)
            for s_ in costume_io.STATES:
                setattr(self, s_.lower(), s_ in st)
            self.bust = bust
            self.other = ' '.join(other)

        def invoke(self, context, event):
            a = context.active_object
            sel = [o for o in context.selected_objects if o.type == 'MESH']
            if a is not None and a.type == 'MESH' and a.data.tmd_material.model and len(sel) > 1:
                self.template = a.name
            t = context.scene.objects.get(self.template)
            if t is not None:
                self._load_tags(t.data.tmd_material.tags)
            return context.window_manager.invoke_props_dialog(self, width=420)

        def draw(self, context):
            col = self.layout.column()
            col.prop(self, "template")
            names = [o.name for o in self._targets(context)]
            col.label(text="Mesh(es): " + (", ".join(names[:4]) + (" ..." if len(names) > 4 else "") if names else "select them"),
                      icon='MESH_DATA')
            col.label(text="Layer tags (when the game draws it):")
            r = col.row(align=True)
            for s_ in costume_io.STATES:
                r.prop(self, s_.lower(), toggle=True)
            col.prop(self, "bust")
            col.prop(self, "other")
            col.prop(self, "materials")
            col.prop(self, "weights")
            col.prop(self, "color")

        def execute(self, context):
            t = context.scene.objects.get(self.template)
            if t is None:
                self.report({'ERROR'}, "choose a template mesh (an imported costume mesh)")
                return {'CANCELLED'}
            targets = self._targets(context)
            if not targets:
                self.report({'ERROR'}, "select the new mesh(es) besides the template")
                return {'CANCELLED'}
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            tags = costume_io.join_tags([s_ for s_ in costume_io.STATES if getattr(self, s_.lower())], self.bust,
                                        self.other.split())
            lines = []
            for o in targets:
                lines += costume_io.make_costume_mesh(o, t, tags, self.materials, self.weights, self.color)
            for line in lines:
                print('evporter mesh setup:', line)
            self.report({'INFO'}, "%d mesh(es) set up like %s, tags '%s'%s" % (
                len(targets), t.name, tags, (": " + lines[-1]) if lines else ""))
            return {'FINISHED'}

    _SLOT_ITEMS = []

    def _slot_items(self, context):
        _SLOT_ITEMS.clear()
        o = context.active_object
        m = o.active_material if o is not None else None
        imgs = dict(m.get('tmd_images', {})) if m is not None else {}
        roles = costume_io.slot_roles(m.get('tmd_fourcc', '')) if m is not None else []
        for k in sorted(imgs, key=lambda x: int(x)):
            role = roles[int(k)] if int(k) < len(roles) else ''
            _SLOT_ITEMS.append((str(k), "t%s %s: %s" % (k, role, imgs[k]), "Texture slot %s of the material" % k))
        if not _SLOT_ITEMS:
            _SLOT_ITEMS.append(('0', "t0", "Texture slot 0"))
        return _SLOT_ITEMS

    def _set_slot_texture(m, k, img, tex_name):
        """slot k of the New game material m -> image img (texture tex_name or the image's name); the node tree's
        image of that slot is swapped so Blender shows it"""
        imgs = dict(m.get('tmd_images', {}))
        old = bpy.data.images.get(imgs.get(k, ''))
        if old is not None:
            try:
                img.colorspace_settings.name = old.colorspace_settings.name
            except TypeError:
                pass
            for n in m.node_tree.nodes:
                if n.bl_idname == 'ShaderNodeTexImage' and n.image == old:
                    n.image = img
        imgs[k] = img.name
        m['tmd_images'] = imgs
        tex = costume_io.texture_name(tex_name or img.name)
        st = dict(m.get('tmd_slot_textures', {}))
        st[k] = tex
        m['tmd_slot_textures'] = st
        si = dict(m.get('tmd_slot_images', {}))
        si[k] = img.name
        m['tmd_slot_images'] = si
        return "t%s = %s" % (k, tex)

    class TMD_OT_material_texture(bpy.types.Operator):
        bl_idname = "evporter.material_texture"
        bl_label = "New texture for game material"
        bl_description = ("Give the active game material a new texture in one of its slots. The material is first "
                          "duplicated into a New game material (only this object's slot uses it); on export the "
                          "copy and the texture are added to the catalogue")
        bl_options = {'REGISTER', 'UNDO'}
        slot: EnumProperty(name="Slot", items=_slot_items, description="Texture slot (t0 = colour for most shaders)")
        image: StringProperty(name="Image", description="An image already in Blender")
        filepath: StringProperty(name="Or load file", subtype='FILE_PATH',
                                 description="Image file to load (PNG, TGA ...; a .dds is written as it is)")
        tex_name: StringProperty(name="Texture name", default="",
                                 description="Name of the texture in the catalogue (empty: from the image name); an "
                                             "existing name reuses that texture")
        companions: BoolProperty(name="Also its specular / normal map", default=True,
                                 description="When a colour texture file is loaded, also use the files next to it named "
                                             "like its specular (X_spc, X_sp, X_specular) or normal map (X_nml, "
                                             "X_normal) for those slots")

        @classmethod
        def poll(cls, context):
            o = context.active_object
            m = o.active_material if o is not None else None
            return m is not None and m.get('tmd_index') is not None

        def invoke(self, context, event):
            self.image, self.filepath, self.tex_name = "", "", ""
            return context.window_manager.invoke_props_dialog(self, width=460)

        def draw(self, context):
            col = self.layout.column()
            m = context.active_object.active_material
            col.label(text="%s%s" % (m.name, "  (new game material)" if m.get('tmd_new') else "  (will be duplicated)"),
                      icon='MATERIAL')
            col.prop(self, "slot")
            col.prop_search(self, "image", bpy.data, "images")
            col.prop(self, "filepath")
            col.prop(self, "tex_name")
            col.prop(self, "companions")
            col.label(text="Existing texture: pick its image (e.g. non_spc)", icon='INFO')

        def execute(self, context):
            o = context.active_object
            m = o.active_material
            if self.filepath:
                try:
                    img = bpy.data.images.load(bpy.path.abspath(self.filepath), check_existing=True)
                except RuntimeError as e:
                    self.report({'ERROR'}, str(e))
                    return {'CANCELLED'}
            else:
                img = bpy.data.images.get(self.image)
            if img is None:
                self.report({'ERROR'}, "choose an image or a file")
                return {'CANCELLED'}
            if not m.get('tmd_new'):
                new = m.copy()
                new.name = m.name + "_new"
                new['tmd_new'] = True
                new['tmd_base_index'] = int(m['tmd_index'])
                new['tmd_slot_textures'] = {}
                new['tmd_slot_images'] = {}
                o.material_slots[o.active_material_index].material = new
                m = new
            done = [_set_slot_texture(m, str(self.slot), img, self.tex_name)]
            if self.companions and self.filepath and int(self.slot) == 0:
                roles = costume_io.slot_roles(m.get('tmd_fourcc', ''))
                for k, path in costume_io.companion_files(bpy.path.abspath(self.filepath), roles).items():
                    if str(k) in dict(m.get('tmd_images', {})):
                        try:
                            ci = bpy.data.images.load(path, check_existing=True)
                        except RuntimeError:
                            continue
                        done.append(_set_slot_texture(m, str(k), ci, ''))
            self.report({'INFO'}, "%s: %s" % (m.name, ", ".join(done)))
            return {'FINISHED'}

    class TMD_OT_costume_export(bpy.types.Operator):
        bl_idname = "evporter.costume_export"
        bl_label = "Export costume (.cat)"
        bl_description = ("Write the character's edited meshes into a copy of an original catalogue (costume, hair, "
                          "face ...). The original is never changed; replacing game files is up to you")
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat", options={'HIDDEN'})
        template: StringProperty(name="Original catalogue", subtype='FILE_PATH',
                                 description="The game's catalogue the meshes came from: read as the template, not "
                                             "written")
        apply_modifiers: BoolProperty(name="Apply modifiers", default=True,
                                      description="Export the meshes with their modifiers (not the Armature modifier)")
        twins: BoolProperty(name="Same changes to identical copies", default=True,
                            description="A material whose channels were edited passes the changes on to the model's "
                                        "identical copies of it (the torn outfit states: same shader, channels and "
                                        "textures apart from _brk0 / _brk1), unless the scene edits those too")
        keep_missing: BoolProperty(name="Keep meshes not in the scene", default=True,
                                   description="Meshes of the model without an object (the hidden bust / damage "
                                               "variants are not imported by default) are kept unchanged; off: they "
                                               "are emptied")
        dds_dir: StringProperty(name="Replacement textures", subtype='DIR_PATH', default="",
                                description="Optional folder of DDS files named like the catalogue's textures "
                                            "(clothes00_EE.dds ...): they replace those textures")

        @classmethod
        def poll(cls, context):
            return _active_armature(context) is not None

        def invoke(self, context, event):
            arm = _active_armature(context)
            gd = _gamedata(context)
            srcs = _model_sources(context, arm)
            if srcs and not self.template:
                self.template = os.path.join(gd.root, srcs[0][0]) if gd is not None else srcs[0][0]
            folder = os.path.dirname(bpy.data.filepath) or os.path.expanduser('~')
            base = os.path.basename(self.template) if self.template else arm.name + '.cat'
            self.filepath = os.path.join(folder, 'export', base)
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def draw(self, context):
            col = self.layout.column()
            col.prop(self, "template")
            arm = _active_armature(context)
            if arm is not None:
                srcs = _model_sources(context, arm)
                if srcs:
                    box = col.box()
                    box.label(text="This character's catalogues:")
                    for src, n in srcs[:6]:
                        box.label(text="%s (%d objects)" % (src, n))
            col.prop(self, "apply_modifiers")
            col.prop(self, "keep_missing")
            col.prop(self, "twins")
            col.prop(self, "dds_dir")
            col.label(text="The original file is not changed.", icon='INFO')

        def execute(self, context):
            arm = _active_armature(context)
            gd = _gamedata(context)
            try:
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                lines = costume_io.export(arm, bpy.path.abspath(self.template), bpy.path.abspath(self.filepath),
                                          self.apply_modifiers, self.keep_missing,
                                          bpy.path.abspath(self.dds_dir) if self.dds_dir else '',
                                          gd.root if gd is not None else '', self.twins)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            for line in lines:
                print('evporter costume export:', line)
            msg = "%s written: %s" % (os.path.basename(self.filepath), "; ".join(lines[:3]))
            context.scene.evporter.motion_status = msg
            self.report({'INFO'}, msg + (" (+%d more lines in the console)" % (len(lines) - 3) if len(lines) > 3 else ""))
            return {'FINISHED'}

    class TMD_PT_costume_export(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Costume export"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            col.operator("evporter.costume_export", icon='EXPORT')
            col.operator("evporter.mesh_setup", icon='MESH_DATA')
            col.label(text="New mesh: select it and an imported one")
            col.operator("evporter.material_texture", icon='TEXTURE')
            col.label(text="Writes a new .cat: replace game files", icon='ERROR')
            col.label(text="yourself, at your own risk (keep backups)")

    def _stage_source(context):
        o = context.active_object
        cands = [o] if o is not None else []
        cands += [x for x in context.scene.objects if x.type == 'MESH']
        for x in cands:
            if x is not None and x.type == 'MESH' and x.data.get('tmd_source_path'):
                return x.data['tmd_source_path']
        for c in bpy.data.collections:
            if c.get('tmd_source'):
                return c['tmd_source']
        return ''

    class TMD_OT_stage_export(bpy.types.Operator):
        bl_idname = "evporter.stage_export"
        bl_label = "Export stage (.cat)"
        bl_description = ("Write the scene's edited stage models and collision meshes into a copy of the original "
                          "stage catalogue (and of the collision catalogue the game loads). Originals are not changed")
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat", options={'HIDDEN'})
        template: StringProperty(name="Original stage", subtype='FILE_PATH',
                                 description="The game's stage catalogue (Model/Field/bgNN.cat): read, not written")
        apply_modifiers: BoolProperty(name="Apply modifiers", default=True)
        twins: BoolProperty(name="Same changes to identical copies", default=True,
                            description="A material whose channels were edited passes the changes on to the model's "
                                        "identical copies of it (the torn outfit states: same shader, channels and "
                                        "textures apart from _brk0 / _brk1), unless the scene edits those too")
        keep_missing: BoolProperty(name="Keep meshes not in the scene", default=True,
                                   description="Meshes without an object are kept unchanged (off: emptied)")
        collision: BoolProperty(name="Collision meshes", default=True,
                                description="Rebuild the collision entries from the imported collision objects "
                                            "(<stage>_collision); without them the original collision is kept")
        touch_template: StringProperty(name="Collision catalogue", subtype='FILE_PATH',
                                       description="The game loads the collision from Binary/Touch/bgNN_tch.cat: its "
                                                   "edited copy is written next to the stage (empty: skip)")
        dds_dir: StringProperty(name="Replacement textures", subtype='DIR_PATH', default="",
                                description="Optional folder of DDS files named like the catalogue's textures")

        def invoke(self, context, event):
            gd = _gamedata(context)
            if not self.template:
                self.template = _stage_source(context)
            stem = os.path.splitext(os.path.basename(self.template))[0] if self.template else 'stage'
            if not self.touch_template:
                roots = [gd.root] if gd is not None else []
                tp = os.path.abspath(bpy.path.abspath(self.template)) if self.template else ''
                if tp:
                    roots.append(os.path.dirname(os.path.dirname(os.path.dirname(tp))))
                for r in roots:
                    t = os.path.join(r, 'Binary', 'Touch', stem + '_tch.cat')
                    if os.path.isfile(t):
                        self.touch_template = t
                        break
            folder = os.path.dirname(bpy.data.filepath) or os.path.expanduser('~')
            self.filepath = os.path.join(folder, 'export', os.path.basename(self.template) if self.template else 'stage.cat')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def draw(self, context):
            col = self.layout.column()
            col.prop(self, "template")
            col.prop(self, "apply_modifiers")
            col.prop(self, "keep_missing")
            col.prop(self, "twins")
            col.prop(self, "collision")
            sub = col.column()
            sub.enabled = self.collision
            sub.prop(self, "touch_template")
            if self.touch_template:
                stem = os.path.splitext(os.path.basename(self.filepath))[0] or 'stage'
                sub.label(text="Writes %s_tch.cat next to the stage" % stem, icon='FILE')
            else:
                sub.label(text="No collision catalogue: Binary/Touch/bgNN_tch.cat", icon='ERROR')
            col.prop(self, "dds_dir")
            col.label(text="The original files are not changed.", icon='INFO')

        def execute(self, context):
            gd = _gamedata(context)
            out = bpy.path.abspath(self.filepath)
            tt = bpy.path.abspath(self.touch_template) if self.touch_template else ''
            tout = os.path.join(os.path.dirname(out), os.path.splitext(os.path.basename(out))[0] + '_tch.cat') if tt else ''
            try:
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                lines = costume_io.export_stage(bpy.path.abspath(self.template), out, self.apply_modifiers,
                                                self.keep_missing, bpy.path.abspath(self.dds_dir) if self.dds_dir else '',
                                                gd.root if gd is not None else '', self.collision, tt, tout, self.twins)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            for line in lines:
                print('evporter stage export:', line)
            msg = "%s written: %s" % (os.path.basename(out), "; ".join(lines[:3]))
            context.scene.evporter.motion_status = msg
            self.report({'INFO'}, msg + (" (+%d more lines in the console)" % (len(lines) - 3) if len(lines) > 3 else ""))
            return {'FINISHED'}

    class TMD_OT_collision_from_selected(bpy.types.Operator):
        bl_idname = "evporter.collision_from_selected"
        bl_label = "Make collision from selected"
        bl_description = ("Add the faces of the selected meshes to the stage's collision objects: faces pointing up "
                          "to the floor (touch_yuka), the others to the walls (touch_kabe). Export stage writes them")
        bl_options = {'REGISTER', 'UNDO'}
        mode: EnumProperty(name="Faces to", default='AUTO', items=(
            ('AUTO', "Floor or wall by slope", "Faces within the slope limit of straight up: floor; the rest: wall"),
            ('FLOOR', "All to the floor", "Every face into touch_yuka"),
            ('WALL', "All to the walls", "Every face into touch_kabe")))
        slope: FloatProperty(name="Floor slope limit", default=45.0, min=0.0, max=90.0, subtype='NONE',
                             description="Faces tilted less than this from horizontal (degrees) become floor")
        apply_modifiers: BoolProperty(name="Apply modifiers", default=True)

        @classmethod
        def poll(cls, context):
            return any(o.type == 'MESH' and not o.get('tmd_touch') for o in context.selected_objects)

        def invoke(self, context, event):
            return context.window_manager.invoke_props_dialog(self)

        def execute(self, context):
            objs = [o for o in context.selected_objects if o.type == 'MESH' and not o.get('tmd_touch')]
            targets = stage_io.collision_targets(objs)
            if not targets:
                self.report({'ERROR'}, "no collision objects: import the stage with Collision meshes ticked")
                return {'CANCELLED'}
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            lines = stage_io.collision_from_objects(objs, targets, self.mode, self.slope, self.apply_modifiers)
            for line in lines:
                print('evporter collision:', line)
            self.report({'INFO'}, "; ".join(lines) or "no faces")
            return {'FINISHED'}

    class TMD_OT_stage_import(bpy.types.Operator):
        bl_idname = "evporter.stage_import"
        bl_label = "Import stage (.cat)"
        bl_description = ("Import a stage catalogue from Model/Field (bgNN.cat): every model with the game's stage "
                          "shaders (baked lightmaps and vertex light, cube-map reflections, water, emissive maps), the "
                          "stage camera loop and optionally the collision meshes")
        bl_options = {'REGISTER', 'UNDO'}
        filepath: StringProperty(subtype='FILE_PATH')
        filter_glob: StringProperty(default="*.cat", options={'HIDDEN'})
        which: StringProperty(name="Models", default="",
                              description="Model names inside the catalogue, comma separated (empty: all, e.g. "
                                          "bg06_00 arena, _a translucent pieces, _o distant ring, _s sky dome)")
        cameras: BoolProperty(name="Stage camera", default=True,
                              description="Import the stage's camera loop (cam_bgNN_loop00) on a new camera")
        camera_mode: EnumProperty(name="Camera keys", default='AIM', items=(
            ('AIM', "Aim empty (game keys)", "Eye keys on the camera, look-at keys on an empty, Track To constraint"),
            ('BAKE', "Bake every frame", "Location and rotation keyed on every frame")))
        motions: BoolProperty(name="Object motions", default=True,
                              description="Import the models' object motions (only the models with bones have "
                                          "non-empty ones)")
        collision: BoolProperty(name="Collision meshes", default=False,
                                description="Also import the collision meshes (touch_kabe walls, touch_yuka floors, "
                                            "touch_switch barrier walls, touch_water) as wireframe objects, not rendered")
        timing: EnumProperty(name="Timing", default='SCENE_60', items=(
            ('SCENE_60', "Scene to 60 fps", "Set the scene to 60 fps: one scene frame per game frame"),
            ('RESAMPLE', "Keep scene fps", "Keep the scene frame rate and place the game frames at their time")))

        def invoke(self, context, event):
            gd = _gamedata(context)
            if gd is not None and not self.filepath:
                self.filepath = os.path.join(gd.root, 'Model', 'Field', '')
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        def draw(self, context):
            col = self.layout.column()
            col.prop(self, "which")
            col.prop(self, "cameras")
            sub = col.column()
            sub.enabled = self.cameras
            sub.prop(self, "camera_mode")
            col.prop(self, "motions")
            col.prop(self, "collision")
            col.prop(self, "timing")
            if context.scene.evporter.shading != 'TOON':
                col.label(text="Materials: Principled BSDF (not the game shaders)", icon='INFO')

        def execute(self, context):
            s = context.scene.evporter
            t0 = time.time()
            try:
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                lines, cams = stage_io.import_stage(bpy.path.abspath(self.filepath), self.which, self.cameras,
                                                    self.camera_mode, self.motions, self.collision, self.timing,
                                                    s.pack, s.shading)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            for line in lines:
                print('evporter stage:', line)
            msg = "; ".join(lines) + " (%.1f s)" % (time.time() - t0)
            s.motion_status = msg
            self.report({'INFO'}, msg)
            return {'FINISHED'}

    class TMD_PT_stage(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Stages"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            col.operator("evporter.stage_import", icon='WORLD')
            col.operator("evporter.stage_export", icon='EXPORT')
            col.operator("evporter.material_texture", icon='TEXTURE')
            col.operator("evporter.collision_from_selected", icon='MOD_PHYSICS')
            col.label(text="Model/Field/bgNN.cat", icon='FILE_FOLDER')
            if context.scene.evporter.shading != 'TOON':
                col.label(text="Set Materials to Game shaders for the stage shaders", icon='INFO')

    class TMD_PT_scene(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Cut-scenes and other models"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            arm = _active_armature(context)
            col.label(text=("Character: " + arm.name) if arm else "Cut-scene: select the character first",
                      icon='ARMATURE_DATA' if arm else 'INFO')
            col.operator("evporter.scene_import", icon='SEQUENCE')
            col.separator()
            col.operator("evporter.model_import", icon='OUTLINER_OB_ARMATURE')
            col.label(text="e.g. Placement/plbg/plbg_die01.cat", icon='FILE_FOLDER')

    class TMD_PT_camera(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Camera motion"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            cam = _active_camera(context)
            if cam is None:
                col.label(text="Select a camera", icon='INFO')
                return
            col.label(text=cam.name + ((": " + cam['tmd_camera']) if cam.get('tmd_camera') else ""), icon='CAMERA_DATA')
            col.operator("evporter.camera_import", icon='IMPORT')
            col.operator("evporter.camera_export", icon='EXPORT')
            col.label(text="Game cameras look at a point, no roll", icon='INFO')

    class TMD_PT_physics(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Physics"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            arm = _tmd_armature(context)
            if arm is None:
                col.label(text="Select an imported character with physics", icon='INFO')
                return
            on = arm.get('tmd_physics_on', True)
            col.label(text="%s: physics %s" % (arm.name, "on" if on else "off"))
            r = col.row(align=True)
            r.operator("evporter.physics_enable", text="On", icon='PLAY', depress=on).enable = True
            r.operator("evporter.physics_enable", text="Off", icon='PAUSE', depress=not on).enable = False
            w = context.scene.rigidbody_world
            if w is not None:
                col.prop(w.point_cache, "frame_start", text="Simulation start")
                if hasattr(w, "substeps_per_frame"):
                    col.prop(w, "substeps_per_frame")
            col.operator("evporter.physics_reset", icon='FILE_REFRESH')
            col.label(text="Pose / move her, then play from the start frame", icon='INFO')
            col.operator("evporter.physics_remove", icon='TRASH')

    class TMD_PT_toon(bpy.types.Panel):
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "EVPorter"
        bl_label = "Toon shading"
        bl_parent_id = "TMD_PT_main"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            col = self.layout.column()
            ng = bpy.data.node_groups.get('TMD Globals')
            if ng is None:
                col.label(text="Import a character with game toon shaders first", icon='INFO')
                return
            n = ng.nodes.get('tmd_light')
            if n is not None:
                col.label(text="Light (toward the light, world)")
                r = col.row(align=True)
                for k in range(3):
                    r.prop(n.inputs[k], "default_value", text="XYZ"[k])
            r = col.row(align=True)
            r.operator("evporter.light_from_sun", icon='LIGHT_SUN')
            r.operator("evporter.reset_light", icon='LOOP_BACK')
            for name, text in (('tmd_wetness', "Wetness"), ('tmd_rim_power', "Rim power"),
                               ('tmd_hair_screen_scale', "Hair ring scale"), ('tmd_gamma', "Gamma")):
                v = ng.nodes.get(name)
                if v is not None:
                    col.prop(v.outputs[0], "default_value", text=text)

    class TMD_PT_material_textures(bpy.types.Panel):
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "material"
        bl_label = "Game material textures"

        @classmethod
        def poll(cls, context):
            m = context.material
            return m is not None and m.get('tmd_index') is not None

        def draw(self, context):
            m = context.material
            col = self.layout.column()
            col.label(text="%s, %s%s" % (m.get('tmd_model', ''), m.get('tmd_fourcc', ''),
                                         ": new material (copy of %d)" % m['tmd_base_index'] if m.get('tmd_new') else
                                         ": material %d" % m['tmd_index']), icon='MATERIAL')
            roles = costume_io.slot_roles(m.get('tmd_fourcc', ''))
            imgs = dict(m.get('tmd_images', {}))
            new = dict(m.get('tmd_slot_textures', {}))
            for k in sorted(imgs, key=lambda x: int(x)):
                r = col.row(align=True)
                role = roles[int(k)] if int(k) < len(roles) else ''
                name = new.get(k, imgs[k])
                r.label(text="t%s %s" % (k, role))
                r.label(text=name + ("  (new)" if k in new else ""), icon='TEXTURE' if k in new else 'IMAGE_DATA')
                op = r.operator("evporter.material_texture", text="", icon='FILE_REFRESH')
                op.slot = k
            if not m.get('tmd_new'):
                col.label(text="Replacing a texture makes a copy of this material", icon='INFO')

    def _orig_channels(m):
        """the material's channels as imported (older imports: from a mesh that uses it)"""
        if m.get('tmd_params_orig') is not None:
            return [float(x) for x in m['tmd_params_orig']]
        for me in bpy.data.meshes:
            for it in me.tmd_material.materials:
                if it.material == m and it.num_params:
                    return [float(x) for x in it.params[:it.num_params]]
        return None

    class TMD_OT_material_channels_preview(bpy.types.Operator):
        bl_idname = "evporter.material_channels_preview"
        bl_label = "Update preview"
        bl_description = ("Show the edited channels in the Blender shader preview (the export always uses the "
                          "values, with or without this)")
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            m = context.material
            msg = toon.apply_channels(m) if m is not None else 'no material'
            if msg:
                self.report({'WARNING'}, msg)
            return {'FINISHED'}

    class TMD_OT_material_channels_reset(bpy.types.Operator):
        bl_idname = "evporter.material_channels_reset"
        bl_label = "Reset to original"
        bl_description = "Set every channel of this game material back to the imported value"
        bl_options = {'REGISTER', 'UNDO'}

        def execute(self, context):
            m = context.material
            orig = _orig_channels(m) if m is not None else None
            if not orig:
                self.report({'WARNING'}, "the imported values are not known for this material")
                return {'CANCELLED'}
            m['tmd_params'] = orig
            toon.apply_channels(m)
            return {'FINISHED'}

    class TMD_PT_material_channels(bpy.types.Panel):
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "material"
        bl_label = "Game material channels"

        @classmethod
        def poll(cls, context):
            m = context.material
            return m is not None and m.get('tmd_params') is not None

        def draw(self, context):
            m = context.material
            col = self.layout.column()
            vals = list(m['tmd_params'])
            orig = _orig_channels(m)
            names = toon.channel_names(m)
            animated = set(int(k) for k in m.get('tmd_param_keys', []))
            changed = [k for k in range(len(vals)) if orig and k < len(orig) and abs(float(vals[k]) - orig[k]) > 1e-6]
            col.label(text="%s material %s: %d channels%s" % (
                m.get('tmd_fourcc', ''), m.get('tmd_index', '?') if not m.get('tmd_new') else 'new', len(vals),
                ", %d changed" % len(changed) if changed else ""), icon='MATERIAL')
            for k in range(len(vals)):
                r = col.row(align=True)
                r.alert = k in changed
                label = "ch%d" % k + (" %s" % names[k] if k in names else "")
                if k in animated:
                    label += " (animated)"
                r.prop(m, '["tmd_params"]', index=k, text=label)
                if k in animated:
                    r.enabled = False
            r = col.row(align=True)
            r.operator("evporter.material_channels_preview", icon='SHADING_RENDERED')
            r.operator("evporter.material_channels_reset", icon='LOOP_BACK')
            col.label(text="Written on export (Export costume / Export stage)", icon='INFO')
            if m.get('tmd_shader') in ('COSTUME', 'SKIN', 'FACE', 'HAIR'):
                col.label(text="Highlight and sheen need vertex colour R (Col) > 0", icon='INFO')

    class TMD_PT_mesh_data(bpy.types.Panel):
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "data"
        bl_label = "TMD material"

        @classmethod
        def poll(cls, context):
            return context.mesh is not None

        def draw(self, context):
            t = context.mesh.tmd_material
            col = self.layout.column()
            if not t.model:
                col.label(text="Not from a game model.", icon='INFO')
                col.label(text="Select it and an imported mesh, then:")
                col.operator("evporter.mesh_setup", icon='MESH_DATA')
                return
            col.prop(t, "export")
            for prop in ("model", "kind", "mesh_index", "tags", "xform_type", "mesh_flags", "vertex_flags", "hidden_reason"):
                col.prop(t, prop)
            col.prop(t, "bounds")
            for it in t.materials:
                box = self.layout.box()
                box.label(text="%s  (material %d)" % (it.fourcc, it.index), icon='MATERIAL')
                box.prop(it, "material")
                box.prop(it, "textures", text="")
                box.prop(it, "num_params")
                box.label(text="Channels as imported; edit them in the Material tab", icon='INFO')
                grid = box.grid_flow(columns=4, align=True)
                grid.enabled = False
                for k in range(min(it.num_params, 32)):
                    grid.prop(it, "params", index=k, text="ch%d" % k)

    classes = (TMD_MaterialEntry, TMD_MeshProps, TMD_Preferences, TMD_MotionItem, TMD_SceneProps, TMD_OT_load_gamedata,
               TMD_OT_motion_open, TMD_OT_motion_read, TMD_OT_motion_select, TMD_OT_motion_import, TMD_OT_motion_export, TMD_UL_motions,
               TMD_OT_import_character, TMD_OT_light_from_sun, TMD_OT_reset_light, TMD_OT_physics_enable,
               TMD_OT_physics_reset, TMD_OT_physics_remove, TMD_OT_ik_setup, TMD_OT_ik_remove, TMD_OT_ik_enable, TMD_OT_ik_match, TMD_OT_camera_import, TMD_OT_camera_export, TMD_OT_scene_import, TMD_OT_model_import, TMD_OT_stage_import, TMD_OT_costume_export, TMD_OT_mesh_setup, TMD_OT_stage_export, TMD_OT_material_texture, TMD_OT_material_channels_preview, TMD_OT_material_channels_reset, TMD_OT_collision_from_selected, TMD_PT_main, TMD_PT_config, TMD_PT_import, TMD_PT_motion, TMD_PT_ik, TMD_PT_costume_export, TMD_PT_stage, TMD_PT_scene, TMD_PT_camera, TMD_PT_physics, TMD_PT_toon,
               TMD_PT_mesh_data, TMD_PT_material_textures, TMD_PT_material_channels)

    def register():
        for c in classes:
            bpy.utils.register_class(c)
        bpy.types.Mesh.tmd_material = PointerProperty(type=TMD_MeshProps)
        bpy.types.Scene.evporter = PointerProperty(type=TMD_SceneProps)
        rigid.register_handlers()

    def unregister():
        rigid.unregister_handlers()
        del bpy.types.Scene.evporter
        del bpy.types.Mesh.tmd_material
        for c in reversed(classes):
            bpy.utils.unregister_class(c)
