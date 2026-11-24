"""
Blender script to create 3 sonar scenarios (cube, sphere, concave_half_sphere)
with 0.25 m radius targets on a seafloor, a 36x3 (108-view) orbiting camera
with -10/0/+10 deg pitch variations and 0.5-5.0 m clip range, and export them
to scenes/*.npz for GPUFLSSimulator.

Usage:
  1. Open in Blender's Scripting tab and click Run Script, OR
  2. Run headless:
     blender --background --python blender/create_and_export_scenes.py
"""

import math
import os
import bpy
import bmesh
import numpy as np
from mathutils import Vector


# Output directory relative to this script's repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "blender" else SCRIPT_DIR
OUT_DIR = os.path.join(REPO_ROOT, "scenes")

# Scenario parameters
OBJECT_RADIUS = 0.25
TARGET_CENTER = Vector((0.0, 0.0, OBJECT_RADIUS))
SEAFLOOR_SIZE = 8.0

# Camera / Sonar parameters
ORBIT_RADIUS = 2.0
CAMERA_HEIGHT = 0.20
AZ_STEP_DEG = 2.0          # 360 / 10 = 36 azimuth views
PITCH_OFFSETS_DEG = (0.0)  # 36 views total (no pitch variation)
AZ_FOV_DEG = 30.0
EL_FOV_DEG = 20.0
MIN_RANGE = 0.5
MAX_RANGE = 5.0


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)


def create_seafloor():
    bpy.ops.mesh.primitive_plane_add(size=SEAFLOOR_SIZE, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = "Seafloor"
    obj["reflectivity"] = 1.0
    return obj


def create_cube_target():
    bpy.ops.mesh.primitive_cube_add(size=2.0 * OBJECT_RADIUS, location=TARGET_CENTER)
    obj = bpy.context.active_object
    obj.name = "Target_Cube"
    obj["reflectivity"] = 0.9
    return obj


def create_sphere_target():
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=16,
        radius=OBJECT_RADIUS,
        location=TARGET_CENTER,
    )
    obj = bpy.context.active_object
    obj.name = "Target_Sphere"
    obj["reflectivity"] = 0.9
    return obj


def create_concave_half_sphere_target(
    outer_radius: float = OBJECT_RADIUS,
    inner_radius: float = 0.19,
    az_segments: int = 24,
    el_segments: int = 12,
):
    """Creates a thick-walled carved/concave half-sphere opening toward -X."""
    mesh = bpy.data.meshes.new("Target_ConcaveHalfSphere_Mesh")
    obj = bpy.data.objects.new("Target_ConcaveHalfSphere", mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = TARGET_CENTER
    obj["reflectivity"] = 0.9

    bm = bmesh.new()

    def add_dome_verts(radius):
        # Pole at +X, rings from alpha=0..pi/2
        pole = bm.verts.new((radius, 0.0, 0.0))
        rings = []
        for i in range(1, el_segments + 1):
            alpha = (0.5 * math.pi) * (i / el_segments)
            ca, sa = math.cos(alpha), math.sin(alpha)
            ring = []
            for j in range(az_segments):
                beta = (2.0 * math.pi) * (j / az_segments)
                ring.append(bm.verts.new((radius * ca, radius * sa * math.cos(beta), radius * sa * math.sin(beta))))
            rings.append(ring)
        return pole, rings

    o_pole, o_rings = add_dome_verts(outer_radius)
    i_pole, i_rings = add_dome_verts(inner_radius)

    # Outer dome faces
    for j in range(az_segments):
        jn = (j + 1) % az_segments
        bm.faces.new((o_pole, o_rings[0][j], o_rings[0][jn]))
        bm.faces.new((i_pole, i_rings[0][jn], i_rings[0][j]))

    for i in range(el_segments - 1):
        for j in range(az_segments):
            jn = (j + 1) % az_segments
            bm.faces.new((o_rings[i][j], o_rings[i + 1][j], o_rings[i + 1][jn], o_rings[i][jn]))
            bm.faces.new((i_rings[i][j], i_rings[i][jn], i_rings[i + 1][jn], i_rings[i + 1][j]))

    # Annular rim connecting outer and inner domes at alpha = pi/2
    o_rim = o_rings[-1]
    i_rim = i_rings[-1]
    for j in range(az_segments):
        jn = (j + 1) % az_segments
        bm.faces.new((i_rim[j], o_rim[j], o_rim[jn], i_rim[jn]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def setup_sonar_camera():
    """Creates and configures the Blender camera to match the sonar FOV and range."""
    scene = bpy.context.scene
    # Set render aspect ratio so angle_x=AZ_FOV_DEG and angle_y=EL_FOV_DEG match exactly
    scene.render.resolution_x = 300
    scene.render.resolution_y = int(round(300 * math.tan(math.radians(EL_FOV_DEG * 0.5)) / math.tan(math.radians(AZ_FOV_DEG * 0.5))))
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    cam_data = bpy.data.cameras.new("SonarCamera")
    cam_data.lens_unit = "FOV"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(AZ_FOV_DEG)
    cam_data.clip_start = MIN_RANGE
    cam_data.clip_end = MAX_RANGE
    cam_data.show_limits = True

    cam_obj = bpy.data.objects.new("SonarCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def build_camera_orbit(cam_obj):
    """Generates 36x3=108 orbiting camera poses and keyframes them on the timeline."""
    num_az = int(round(360.0 / AZ_STEP_DEG))
    total_frames = num_az * len(PITCH_OFFSETS_DEG)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames

    cam_pos = []
    cam_yaw_deg = []
    cam_pitch_deg = []
    cam_az_deg = []
    cam_pitch_offset_deg = []

    frame = 1
    for az_idx in range(num_az):
        az_deg = az_idx * AZ_STEP_DEG
        az_rad = math.radians(az_deg)
        pos = Vector((
            ORBIT_RADIUS * math.cos(az_rad),
            ORBIT_RADIUS * math.sin(az_rad),
            CAMERA_HEIGHT,
        ))
        base_dir = (TARGET_CENTER - pos).normalized()
        base_yaw = math.atan2(base_dir.y, base_dir.x)
        base_pitch = math.asin(max(-1.0, min(1.0, base_dir.z)))

        for p_off_deg in PITCH_OFFSETS_DEG:
            total_pitch = base_pitch + math.radians(p_off_deg)
            fwd = Vector((
                math.cos(total_pitch) * math.cos(base_yaw),
                math.cos(total_pitch) * math.sin(base_yaw),
                math.sin(total_pitch),
            )).normalized()

            cam_obj.location = pos
            cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
            cam_obj.keyframe_insert(data_path="location", frame=frame)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

            # Extract world forward (-Z in Blender camera frame) for exact simulator alignment
            bpy.context.view_layer.update()
            world_fwd = -(cam_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
            yaw = math.degrees(math.atan2(world_fwd.y, world_fwd.x))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, world_fwd.z))))

            cam_pos.append([pos.x, pos.y, pos.z])
            cam_yaw_deg.append(yaw)
            cam_pitch_deg.append(pitch)
            cam_az_deg.append(az_deg)
            cam_pitch_offset_deg.append(p_off_deg)
            frame += 1

    return {
        "cam_pos": np.asarray(cam_pos, dtype=np.float32),
        "cam_yaw_deg": np.asarray(cam_yaw_deg, dtype=np.float32),
        "cam_pitch_deg": np.asarray(cam_pitch_deg, dtype=np.float32),
        "cam_az_deg": np.asarray(cam_az_deg, dtype=np.float32),
        "cam_pitch_offset_deg": np.asarray(cam_pitch_offset_deg, dtype=np.float32),
    }


def extract_mesh_objects_world(mesh_objects):
    """Triangulates Blender mesh objects in world space and computes per-object AABBs."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    v0_list, v1_list, v2_list, n_list, refl_list, obj_id_list = [], [], [], [], [], []
    bmin_list, bmax_list = [], []

    for obj_id, obj in enumerate(mesh_objects):
        eval_obj = obj.evaluated_get(depsgraph)
        bm = bmesh.new()
        bm.from_mesh(eval_obj.to_mesh())
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

        pts_arr = np.asarray(obj_pts, dtype=np.float32)
        bmin_list.append(pts_arr.min(axis=0))
        bmax_list.append(pts_arr.max(axis=0))

        bm.free()
        eval_obj.to_mesh_clear()

    return {
        "v0": np.asarray(v0_list, dtype=np.float32),
        "v1": np.asarray(v1_list, dtype=np.float32),
        "v2": np.asarray(v2_list, dtype=np.float32),
        "normals": np.asarray(n_list, dtype=np.float32),
        "refl": np.asarray(refl_list, dtype=np.float32),
        "obj_ids": np.asarray(obj_id_list, dtype=np.int32),
        "box_min": np.asarray(bmin_list, dtype=np.float32),
        "box_max": np.asarray(bmax_list, dtype=np.float32),
    }


def export_scenario(name: str, target_builder):
    clear_scene()
    seafloor = create_seafloor()
    target = target_builder()
    cam_obj = setup_sonar_camera()

    cam_data = build_camera_orbit(cam_obj)
    mesh_data = extract_mesh_objects_world([seafloor, target])

    os.makedirs(OUT_DIR, exist_ok=True)
    blend_path = os.path.join(OUT_DIR, f"{name}.blend")
    bpy.context.scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    out_path = os.path.join(OUT_DIR, f"{name}.npz")
    np.savez(
        out_path,
        **mesh_data,
        **cam_data,
        az_fov_deg=np.float32(AZ_FOV_DEG),
        el_fov_deg=np.float32(EL_FOV_DEG),
        min_range=np.float32(MIN_RANGE),
        max_range=np.float32(MAX_RANGE),
    )
    print(f"[Blender Export] {name}: {len(mesh_data['v0'])} tris, {len(cam_data['cam_pos'])} views -> {out_path}")


def main():
    scenarios = [
        ("cube", create_cube_target),
        ("sphere", create_sphere_target),
        ("concave_half_sphere", create_concave_half_sphere_target),
    ]
    for name, builder in scenarios:
        export_scenario(name, builder)

    # Leave the concave_half_sphere scenario on frame 1 in the Blender viewport
    bpy.context.scene.frame_set(1)


if __name__ == "__main__":
    main()
