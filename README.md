## EVPorter (Blender add-on)

EVPorter brings **SK Estival Versus** game data into Blender and writes your edits back in the game's
own formats. It reads straight from the game's `GameData` folder. Requires Blender 4.2 or newer.

<img width="1585" height="847" alt="screen2" src="https://github.com/user-attachments/assets/1dc08b79-fb18-40ae-8044-0f671d337fcc" />

**Import**
- **Characters** in any outfit: skeleton, body/costume, face, hair, underwear and bra, accessories placed where the
  game puts them, and the game's mesh variants (damage states, bust sizes).
- **Toon shading:** a port of the game's shaders, driven by each material's own data, with the game's light direction
  (or aim it with a Sun lamp).
- **Physics:** hair, skirt, bust and accessory secondary motion turned into rigid bodies, with a reset-to-pose
  button.
- **Motions** (`.tmo` or motion catalogues) as actions, and **camera motions** onto a Blender camera.
- **Stages** (`Model/Field/bgNN.cat`) with their props and collision meshes, **cut-scenes**, and any other model
  catalogue (enemies, objects, props).

**Posing**
- Optional foot/hand IK controls with knee/elbow poles that can be switched on and off without losing the pose.

**Export** (always to a new file; originals and parameter tables such as `cosinfo` are never modified)
- **Costumes:** edited or new meshes and textures written into a copy of the original catalogue (costume, hair,
  face ...), ready to replace the game file yourself.
- **Material edits:** the game material channels (colours, specular, rim, outline ...) can be edited in the
  sidebar, previewed in Blender, and are written into the exported models.
- **Stages:** edited stage models and collision meshes (make collision from any selected mesh).
- **Motions** and **camera motions** back to `.tmo`.
  
