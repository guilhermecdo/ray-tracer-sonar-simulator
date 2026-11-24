"""
Convert any Blender (.blend) file into a sonar simulator scene (.npz).

Can be run directly from your terminal:
    python blender2scene.py scenes/coral.blend
    python blender2scene.py scenes/coral.blend --out scenes/coral.npz --max-tris 10000
"""

import argparse
import math
import os
import shutil
import subprocess
import sys


def _find_blender_executable() -> str:
    found = shutil.which("blender")
    if found:
        return found
    mac_path = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(mac_path):
        return mac_path
    raise FileNotFoundError(
        "Could not find Blender executable. Install Blender or add it to PATH."
    )


def _run_as_cli_wrapper():
    parser = argparse.ArgumentParser(
        description="Convert a .blend file into a sonar simulator .npz scene."
    )
    parser.add_argument("blend_file", help="Path to input .blend file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz path (default: <blend_dir>/<blend_name>.npz).",
    )
    parser.add_argument(
        "--max-tris",
        type=int,
        default=10000,
        help="Maximum triangles per mesh object; larger meshes are decimated (default: 10000).",
    )
    args = parser.parse_args()

    blend_path = os.path.abspath(args.blend_file)
    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"Input .blend file not found: {blend_path}")

    blender_bin = _find_blender_executable()
    cmd = [
        blender_bin,
        "--background",
        blend_path,
        "--python",
        os.path.abspath(__file__),
        "--",
        "--max-tris",
        str(args.max_tris),
    ]
    if args.out is not None:
        cmd.extend(["--out", os.path.abspath(args.out)])

    subprocess.run(cmd, check=True)


def _run_inside_blender():
    import bpy
    import bmesh
    import numpy as np
    from mathutils import Vector

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-tris", type=int, default=10000)
    args = parser.parse_args(argv)

    blend_path = bpy.data.filepath
    if args.out is not None:
        out_path = args.out
    elif blend_path:
        out_path = os.path.splitext(blend_path)[0] + ".npz"
    else:
        raise ValueError("No .blend filepath found; pass --out <path.npz>.")

    scene = bpy.context.scene

    # 1. Find camera
    cam_obj = scene.camera or bpy.data.objects.get("SonarCamera")
    if cam_obj is None:
        for obj in scene.objects:
            if obj.type == "CAMERA":
                cam_obj = obj
                break
    if cam_obj is None:
        raise RuntimeError("No CAMERA object found in the .blend scene.")

    cam_data = cam_obj.data
    aspect = (scene.render.resolution_y * scene.render.pixel_aspect_y) / max(
        1.0, scene.render.resolution_x * scene.render.pixel_aspect_x
    )
    az_fov_rad = float(cam_data.angle_x)
    el_fov_rad = 2.0 * math.atan(math.tan(az_fov_rad * 0.5) * aspect)
    az_fov_deg = math.degrees(az_fov_rad)
    el_fov_deg = math.degrees(el_fov_rad)
    min_range = float(cam_data.clip_start)
    max_range = float(cam_data.clip_end)

    # 2. Extract camera trajectory (animated frames or single static pose)
    has_anim = bool(cam_obj.animation_data and cam_obj.animation_data.action)
    frames = (
        range(scene.frame_start, scene.frame_end + 1)
        if has_anim
        else [scene.frame_current]
    )

    cam_pos, cam_yaw_deg, cam_pitch_deg, cam_az_deg = [], [], [], []
    for f in frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        pos = cam_obj.matrix_world.translation.copy()
        world_fwd = -(cam_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
        yaw = math.degrees(math.atan2(world_fwd.y, world_fwd.x))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, world_fwd.z))))
        orbit_az = math.degrees(math.atan2(pos.y, pos.x)) % 360.0

        cam_pos.append([pos.x, pos.y, pos.z])
        cam_yaw_deg.append(yaw)
        cam_pitch_deg.append(pitch)
        cam_az_deg.append(orbit_az)

    # 3. Extract visible MESH objects in world space (decimating if > max_tris)
    mesh_objs = [
        obj for obj in scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]
    if not mesh_objs:
        raise RuntimeError("No visible MESH objects found in the .blend scene.")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    v0_list, v1_list, v2_list, n_list, refl_list, obj_id_list = [], [], [], [], [], []
    bmin_list, bmax_list = [], []

    for obj_id, obj in enumerate(mesh_objs):
        eval_obj = obj.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()

        # Estimate triangle count (each n-gon becomes n-2 tris)
        est_tris = sum(max(1, len(p.vertices) - 2) for p in temp_mesh.polygons)
        temp_obj = None
        if args.max_tris > 0 and est_tris > args.max_tris:
            ratio = max(0.0005, float(args.max_tris) / float(est_tris))
            dec_mesh = temp_mesh.copy()
            temp_obj = bpy.data.objects.new("_tmp_decimate", dec_mesh)
            bpy.context.collection.objects.link(temp_obj)
            mod = temp_obj.modifiers.new(name="Decimate", type="DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            bpy.context.view_layer.update()
            eval_dec = temp_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            bm = bmesh.new()
            bm.from_mesh(eval_dec.to_mesh())
            eval_dec.to_mesh_clear()
        else:
            bm = bmesh.new()
            bm.from_mesh(temp_mesh)

        eval_obj.to_mesh_clear()
        if temp_obj is not None:
            dec_data = temp_obj.data
            bpy.data.objects.remove(temp_obj, do_unlink=True)
            bpy.data.meshes.remove(dec_data)

        bmesh.ops.transform(bm, matrix=obj.matrix_world, verts=bm.verts)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.normal_update()

        refl = float(obj.get("reflectivity", 0.9))
        obj_pts = []
        for face in bm.faces:
            if len(face.verts) != 3:
                continue
            p0 = np.array(face.verts[0].co, dtype=np.float32)
            p1 = np.array(face.verts[1].co, dtype=np.float32)
            p2 = np.array(face.verts[2].co, dtype=np.float32)
            n = np.array(face.normal, dtype=np.float32)
            n_len = np.linalg.norm(n)
            if n_len < 1e-7:
                continue
            n /= n_len

            v0_list.append(p0)
            v1_list.append(p1)
            v2_list.append(p2)
            n_list.append(n)
            refl_list.append(refl)
            obj_id_list.append(obj_id)
            obj_pts.extend([p0, p1, p2])

        if obj_pts:
            pts_arr = np.asarray(obj_pts, dtype=np.float32)
            bmin_list.append(pts_arr.min(axis=0))
            bmax_list.append(pts_arr.max(axis=0))

        bm.free()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        v0=np.asarray(v0_list, dtype=np.float32),
        v1=np.asarray(v1_list, dtype=np.float32),
        v2=np.asarray(v2_list, dtype=np.float32),
        normals=np.asarray(n_list, dtype=np.float32),
        refl=np.asarray(refl_list, dtype=np.float32),
        obj_ids=np.asarray(obj_id_list, dtype=np.int32),
        box_min=np.asarray(bmin_list, dtype=np.float32),
        box_max=np.asarray(bmax_list, dtype=np.float32),
        cam_pos=np.asarray(cam_pos, dtype=np.float32),
        cam_yaw_deg=np.asarray(cam_yaw_deg, dtype=np.float32),
        cam_pitch_deg=np.asarray(cam_pitch_deg, dtype=np.float32),
        cam_az_deg=np.asarray(cam_az_deg, dtype=np.float32),
        cam_pitch_offset_deg=np.zeros(len(cam_pos), dtype=np.float32),
        az_fov_deg=np.float32(az_fov_deg),
        el_fov_deg=np.float32(el_fov_deg),
        min_range=np.float32(min_range),
        max_range=np.float32(max_range),
    )
    print(
        f"[blender2scene] Exported {len(v0_list)} tris ({len(mesh_objs)} objects), "
        f"{len(cam_pos)} views -> {out_path}"
    )


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
        _INSIDE_BLENDER = True
    except ImportError:
        _INSIDE_BLENDER = False

    if _INSIDE_BLENDER:
        _run_inside_blender()
    else:
        _run_as_cli_wrapper()
