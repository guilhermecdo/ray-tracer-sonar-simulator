import numpy as np
import taichi as ti


class SonarScene:
    def __init__(self, max_tris: int = 4096, max_boxes: int = 32):
        self.max_tris = max_tris
        self.max_boxes = max_boxes
        self.num_tris = 0
        self.num_boxes = 0

        self.v0 = ti.Vector.field(3, dtype=ti.f32, shape=max_tris)
        self.v1 = ti.Vector.field(3, dtype=ti.f32, shape=max_tris)
        self.v2 = ti.Vector.field(3, dtype=ti.f32, shape=max_tris)
        self.tri_normal = ti.Vector.field(3, dtype=ti.f32, shape=max_tris)
        self.tri_refl = ti.field(dtype=ti.f32, shape=max_tris)
        self.tri_box_id = ti.field(dtype=ti.i32, shape=max_tris)

        self.box_min = ti.Vector.field(3, dtype=ti.f32, shape=max_boxes)
        self.box_max = ti.Vector.field(3, dtype=ti.f32, shape=max_boxes)

        self._v0_list = []
        self._v1_list = []
        self._v2_list = []
        self._n_list = []
        self._refl_list = []
        self._box_id_list = []
        self._bmin_list = []
        self._bmax_list = []

    def clear(self):
        """Clears all staged geometry so the scene can be rebuilt and re-uploaded."""
        self.num_tris = 0
        self.num_boxes = 0
        self._v0_list.clear()
        self._v1_list.clear()
        self._v2_list.clear()
        self._n_list.clear()
        self._refl_list.clear()
        self._box_id_list.clear()
        self._bmin_list.clear()
        self._bmax_list.clear()

    def add_box_mesh(self, center, size, reflectivity: float = 0.8, yaw_deg: float = 0.0):
        """Adds a 3D box (12 triangles), optionally rotated around Z by yaw_deg, and registers its AABB."""
        if self.num_boxes >= self.max_boxes:
            raise ValueError("Exceeded max_boxes")

        c = np.asarray(center, dtype=np.float32)
        s = np.asarray(size, dtype=np.float32)
        hx, hy, hz = s[0] * 0.5, s[1] * 0.5, s[2] * 0.5

        local_pts = np.array([
            [-hx, -hy, -hz],  # 0
            [+hx, -hy, -hz],  # 1
            [+hx, +hy, -hz],  # 2
            [-hx, +hy, -hz],  # 3
            [-hx, -hy, +hz],  # 4
            [+hx, -hy, +hz],  # 5
            [+hx, +hy, +hz],  # 6
            [-hx, +hy, +hz],  # 7
        ], dtype=np.float32)

        yaw = np.radians(yaw_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot_z = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        pts = (local_pts @ rot_z.T) + c

        # 6 faces -> 12 CCW triangles with outward normals
        quads = [
            ([0, 3, 2, 1], [0.0, 0.0, -1.0]),  # -Z
            ([4, 5, 6, 7], [0.0, 0.0, 1.0]),   # +Z
            ([0, 1, 5, 4], [0.0, -1.0, 0.0]),  # -Y
            ([3, 7, 6, 2], [0.0, 1.0, 0.0]),   # +Y
            ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),  # -X
            ([1, 2, 6, 5], [1.0, 0.0, 0.0]),   # +X
        ]

        box_id = self.num_boxes
        for idx_quad, normal in quads:
            rot_normal = rot_z @ np.asarray(normal, dtype=np.float32)
            a, b, d_pt, e_pt = pts[idx_quad[0]], pts[idx_quad[1]], pts[idx_quad[2]], pts[idx_quad[3]]
            for tri in [(a, b, d_pt), (a, d_pt, e_pt)]:
                if len(self._v0_list) >= self.max_tris:
                    raise ValueError("Exceeded max_tris")
                self._v0_list.append(tri[0])
                self._v1_list.append(tri[1])
                self._v2_list.append(tri[2])
                self._n_list.append(rot_normal)
                self._refl_list.append(reflectivity)
                self._box_id_list.append(box_id)

        self._bmin_list.append(pts.min(axis=0))
        self._bmax_list.append(pts.max(axis=0))
        self.num_boxes += 1
        self.num_tris = len(self._v0_list)

    def _add_triangle(self, p0, p1, p2, normal, reflectivity: float, box_id: int):
        if len(self._v0_list) >= self.max_tris:
            raise ValueError("Exceeded max_tris")
        self._v0_list.append(p0)
        self._v1_list.append(p1)
        self._v2_list.append(p2)
        self._n_list.append(normal)
        self._refl_list.append(reflectivity)
        self._box_id_list.append(box_id)

    def _add_quad_auto_normal(self, p0, p1, p2, p3, outward_hint, reflectivity: float, box_id: int):
        """Splits a quad (p0, p1, p2, p3) into 2 triangles with normals aligned to outward_hint."""
        for a, b, c in ((p0, p1, p2), (p0, p2, p3)):
            n = np.cross(b - a, c - a)
            norm_len = np.linalg.norm(n)
            if norm_len < 1e-7:
                continue
            n = n / norm_len
            if np.dot(n, outward_hint) < 0.0:
                b, c = c, b
                n = -n
            self._add_triangle(a, b, c, n.astype(np.float32), reflectivity, box_id)

    def add_carved_half_cylinder_mesh(
        self,
        center,
        outer_radius: float = 0.8,
        inner_radius: float = 0.6,
        height: float = 1.4,
        segments: int = 24,
        reflectivity: float = 0.9,
        yaw_deg: float = 0.0,
    ):
        """Adds a thick-walled carved half-cylinder (open toward -X at yaw_deg=0)."""
        if self.num_boxes >= self.max_boxes:
            raise ValueError("Exceeded max_boxes")

        c = np.asarray(center, dtype=np.float32)
        hz = height * 0.5
        yaw = np.radians(yaw_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot_z = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        def to_world(pt):
            return (rot_z @ np.asarray(pt, dtype=np.float32)) + c

        # Arc spans -pi/2 to +pi/2 in (+X, Y) so the carved cavity opens toward -X
        thetas = np.linspace(-0.5 * np.pi, 0.5 * np.pi, segments + 1)
        box_id = self.num_boxes
        all_pts = []

        for i in range(segments):
            t0, t1 = thetas[i], thetas[i + 1]
            c0, s0 = np.cos(t0), np.sin(t0)
            c1, s1 = np.cos(t1), np.sin(t1)
            mid_dir = rot_z @ np.array([np.cos(0.5 * (t0 + t1)), np.sin(0.5 * (t0 + t1)), 0.0], dtype=np.float32)

            ob0 = to_world([outer_radius * c0, outer_radius * s0, -hz])
            ob1 = to_world([outer_radius * c1, outer_radius * s1, -hz])
            ot0 = to_world([outer_radius * c0, outer_radius * s0, +hz])
            ot1 = to_world([outer_radius * c1, outer_radius * s1, +hz])

            ib0 = to_world([inner_radius * c0, inner_radius * s0, -hz])
            ib1 = to_world([inner_radius * c1, inner_radius * s1, -hz])
            it0 = to_world([inner_radius * c0, inner_radius * s0, +hz])
            it1 = to_world([inner_radius * c1, inner_radius * s1, +hz])

            all_pts.extend([ob0, ob1, ot0, ot1, ib0, ib1, it0, it1])

            # Outer convex wall (normal points outward)
            self._add_quad_auto_normal(ob0, ob1, ot1, ot0, mid_dir, reflectivity, box_id)
            # Inner carved concave wall (normal points inward)
            self._add_quad_auto_normal(ib0, it0, it1, ib1, -mid_dir, reflectivity, box_id)
            # Top & bottom annular caps
            self._add_quad_auto_normal(it0, ot0, ot1, it1, np.array([0.0, 0.0, 1.0], dtype=np.float32), reflectivity, box_id)
            self._add_quad_auto_normal(ib0, ib1, ob1, ob0, np.array([0.0, 0.0, -1.0], dtype=np.float32), reflectivity, box_id)

        # Close the two cut faces at theta = -pi/2 and +pi/2
        for t, hint_local in [(-0.5 * np.pi, [0.0, -1.0, 0.0]), (0.5 * np.pi, [0.0, 1.0, 0.0])]:
            ct, st = np.cos(t), np.sin(t)
            ob = to_world([outer_radius * ct, outer_radius * st, -hz])
            ot = to_world([outer_radius * ct, outer_radius * st, +hz])
            ib = to_world([inner_radius * ct, inner_radius * st, -hz])
            it = to_world([inner_radius * ct, inner_radius * st, +hz])
            hint = rot_z @ np.asarray(hint_local, dtype=np.float32)
            self._add_quad_auto_normal(ib, ob, ot, it, hint, reflectivity, box_id)

        pts_arr = np.asarray(all_pts, dtype=np.float32)
        self._bmin_list.append(pts_arr.min(axis=0))
        self._bmax_list.append(pts_arr.max(axis=0))
        self.num_boxes += 1
        self.num_tris = len(self._v0_list)

    def add_triangle_mesh(self, v0, v1, v2, normals, reflectivity: float = 0.9):
        """Adds an arbitrary triangulated mesh (N triangles) as one AABB-culled object."""
        if self.num_boxes >= self.max_boxes:
            raise ValueError("Exceeded max_boxes")

        v0_arr = np.asarray(v0, dtype=np.float32)
        v1_arr = np.asarray(v1, dtype=np.float32)
        v2_arr = np.asarray(v2, dtype=np.float32)
        n_arr = np.asarray(normals, dtype=np.float32)

        n_tris = len(v0_arr)
        if n_tris == 0:
            return
        if len(self._v0_list) + n_tris > self.max_tris:
            raise ValueError(f"Exceeded max_tris ({len(self._v0_list) + n_tris} > {self.max_tris})")

        box_id = self.num_boxes
        refl_arr = np.full(n_tris, float(reflectivity), dtype=np.float32) if np.ndim(reflectivity) == 0 else np.asarray(reflectivity, dtype=np.float32)

        self._v0_list.extend(v0_arr)
        self._v1_list.extend(v1_arr)
        self._v2_list.extend(v2_arr)
        self._n_list.extend(n_arr)
        self._refl_list.extend(refl_arr.tolist())
        self._box_id_list.extend([box_id] * n_tris)

        all_pts = np.concatenate([v0_arr, v1_arr, v2_arr], axis=0)
        self._bmin_list.append(all_pts.min(axis=0))
        self._bmax_list.append(all_pts.max(axis=0))
        self.num_boxes += 1
        self.num_tris = len(self._v0_list)

    def load_blender_scene(self, npz_path: str) -> dict:
        """Loads a scene and camera trajectory exported by blender/create_and_export_scenes.py."""
        data = np.load(npz_path, allow_pickle=False)
        self.clear()

        v0 = data["v0"].astype(np.float32)
        v1 = data["v1"].astype(np.float32)
        v2 = data["v2"].astype(np.float32)
        normals = data["normals"].astype(np.float32)
        refl = data["refl"].astype(np.float32)
        obj_ids = data["obj_ids"].astype(np.int32)
        box_min = data["box_min"].astype(np.float32)
        box_max = data["box_max"].astype(np.float32)

        n_tris = len(v0)
        n_boxes = len(box_min)
        if n_tris > self.max_tris:
            raise ValueError(f"Scene has {n_tris} tris, exceeds max_tris={self.max_tris}")
        if n_boxes > self.max_boxes:
            raise ValueError(f"Scene has {n_boxes} objects, exceeds max_boxes={self.max_boxes}")

        self._v0_list = list(v0)
        self._v1_list = list(v1)
        self._v2_list = list(v2)
        self._n_list = list(normals)
        self._refl_list = refl.tolist()
        self._box_id_list = obj_ids.tolist()
        self._bmin_list = list(box_min)
        self._bmax_list = list(box_max)
        self.num_tris = n_tris
        self.num_boxes = n_boxes
        self.upload()

        return {
            "cam_pos": data["cam_pos"].astype(np.float32),
            "cam_yaw_deg": data["cam_yaw_deg"].astype(np.float32),
            "cam_pitch_deg": data["cam_pitch_deg"].astype(np.float32),
            "cam_az_deg": data["cam_az_deg"].astype(np.float32),
            "cam_pitch_offset_deg": data["cam_pitch_offset_deg"].astype(np.float32),
            "az_fov_deg": float(data["az_fov_deg"]),
            "el_fov_deg": float(data["el_fov_deg"]),
            "min_range": float(data["min_range"]),
            "max_range": float(data["max_range"]),
        }

    def add_carved_hemisphere_mesh(
        self,
        center,
        outer_radius: float = 0.85,
        inner_radius: float = 0.65,
        az_segments: int = 20,
        el_segments: int = 10,
        reflectivity: float = 0.9,
        yaw_deg: float = 0.0,
    ):
        """Adds a thick-walled carved half-sphere (concave bowl opening toward -X at yaw_deg=0)."""
        if self.num_boxes >= self.max_boxes:
            raise ValueError("Exceeded max_boxes")

        c = np.asarray(center, dtype=np.float32)
        yaw = np.radians(yaw_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot_z = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        def sphere_pt(r, alpha, beta):
            # alpha in [0, pi/2] from +X pole toward YZ-plane rim; beta in [0, 2*pi] around X axis
            ca, sa = np.cos(alpha), np.sin(alpha)
            cb, sb = np.cos(beta), np.sin(beta)
            local = np.array([r * ca, r * sa * cb, r * sa * sb], dtype=np.float32)
            return (rot_z @ local) + c

        alphas = np.linspace(0.0, 0.5 * np.pi, el_segments + 1)
        betas = np.linspace(0.0, 2.0 * np.pi, az_segments + 1)
        box_id = self.num_boxes
        all_pts = []

        for i in range(el_segments):
            a0, a1 = alphas[i], alphas[i + 1]
            am = 0.5 * (a0 + a1)
            for j in range(az_segments):
                b0, b1 = betas[j], betas[j + 1]
                bm = 0.5 * (b0 + b1)
                mid_local = np.array([np.cos(am), np.sin(am) * np.cos(bm), np.sin(am) * np.sin(bm)], dtype=np.float32)
                mid_dir = rot_z @ mid_local

                o00 = sphere_pt(outer_radius, a0, b0)
                o01 = sphere_pt(outer_radius, a0, b1)
                o10 = sphere_pt(outer_radius, a1, b0)
                o11 = sphere_pt(outer_radius, a1, b1)

                i00 = sphere_pt(inner_radius, a0, b0)
                i01 = sphere_pt(inner_radius, a0, b1)
                i10 = sphere_pt(inner_radius, a1, b0)
                i11 = sphere_pt(inner_radius, a1, b1)

                all_pts.extend([o00, o01, o10, o11, i00, i01, i10, i11])

                # Outer convex dome
                self._add_quad_auto_normal(o00, o10, o11, o01, mid_dir, reflectivity, box_id)
                # Inner carved concave bowl
                self._add_quad_auto_normal(i00, i01, i11, i10, -mid_dir, reflectivity, box_id)

        # Annular rim at alpha = pi/2 (facing -X in local frame)
        rim_hint = rot_z @ np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        for j in range(az_segments):
            b0, b1 = betas[j], betas[j + 1]
            ro0 = sphere_pt(outer_radius, 0.5 * np.pi, b0)
            ro1 = sphere_pt(outer_radius, 0.5 * np.pi, b1)
            ri0 = sphere_pt(inner_radius, 0.5 * np.pi, b0)
            ri1 = sphere_pt(inner_radius, 0.5 * np.pi, b1)
            self._add_quad_auto_normal(ri0, ro0, ro1, ri1, rim_hint, reflectivity, box_id)

        pts_arr = np.asarray(all_pts, dtype=np.float32)
        self._bmin_list.append(pts_arr.min(axis=0))
        self._bmax_list.append(pts_arr.max(axis=0))
        self.num_boxes += 1
        self.num_tris = len(self._v0_list)

    def add_triangle_mesh(self, v0, v1, v2, normals, reflectivity: float = 0.9):
        """Adds an arbitrary triangulated mesh (N triangles) as one AABB-culled object."""
        if self.num_boxes >= self.max_boxes:
            raise ValueError("Exceeded max_boxes")

        v0_arr = np.asarray(v0, dtype=np.float32)
        v1_arr = np.asarray(v1, dtype=np.float32)
        v2_arr = np.asarray(v2, dtype=np.float32)
        n_arr = np.asarray(normals, dtype=np.float32)

        n_tris = len(v0_arr)
        if n_tris == 0:
            return
        if len(self._v0_list) + n_tris > self.max_tris:
            raise ValueError(f"Exceeded max_tris ({len(self._v0_list) + n_tris} > {self.max_tris})")

        box_id = self.num_boxes
        refl_arr = np.full(n_tris, float(reflectivity), dtype=np.float32) if np.ndim(reflectivity) == 0 else np.asarray(reflectivity, dtype=np.float32)

        self._v0_list.extend(v0_arr)
        self._v1_list.extend(v1_arr)
        self._v2_list.extend(v2_arr)
        self._n_list.extend(n_arr)
        self._refl_list.extend(refl_arr.tolist())
        self._box_id_list.extend([box_id] * n_tris)

        all_pts = np.concatenate([v0_arr, v1_arr, v2_arr], axis=0)
        self._bmin_list.append(all_pts.min(axis=0))
        self._bmax_list.append(all_pts.max(axis=0))
        self.num_boxes += 1
        self.num_tris = len(self._v0_list)

    def load_blender_scene(self, npz_path: str) -> dict:
        """Loads a scene and camera trajectory exported by blender/create_and_export_scenes.py."""
        data = np.load(npz_path, allow_pickle=False)
        self.clear()

        v0 = data["v0"].astype(np.float32)
        v1 = data["v1"].astype(np.float32)
        v2 = data["v2"].astype(np.float32)
        normals = data["normals"].astype(np.float32)
        refl = data["refl"].astype(np.float32)
        obj_ids = data["obj_ids"].astype(np.int32)
        box_min = data["box_min"].astype(np.float32)
        box_max = data["box_max"].astype(np.float32)

        n_tris = len(v0)
        n_boxes = len(box_min)
        if n_tris > self.max_tris:
            raise ValueError(f"Scene has {n_tris} tris, exceeds max_tris={self.max_tris}")
        if n_boxes > self.max_boxes:
            raise ValueError(f"Scene has {n_boxes} objects, exceeds max_boxes={self.max_boxes}")

        self._v0_list = list(v0)
        self._v1_list = list(v1)
        self._v2_list = list(v2)
        self._n_list = list(normals)
        self._refl_list = refl.tolist()
        self._box_id_list = obj_ids.tolist()
        self._bmin_list = list(box_min)
        self._bmax_list = list(box_max)
        self.num_tris = n_tris
        self.num_boxes = n_boxes
        self.upload()

        return {
            "cam_pos": data["cam_pos"].astype(np.float32),
            "cam_yaw_deg": data["cam_yaw_deg"].astype(np.float32),
            "cam_pitch_deg": data["cam_pitch_deg"].astype(np.float32),
            "cam_az_deg": data["cam_az_deg"].astype(np.float32),
            "cam_pitch_offset_deg": data["cam_pitch_offset_deg"].astype(np.float32),
            "az_fov_deg": float(data["az_fov_deg"]),
            "el_fov_deg": float(data["el_fov_deg"]),
            "min_range": float(data["min_range"]),
            "max_range": float(data["max_range"]),
        }

    def upload(self):
        if self.num_tris > 0:
            self.v0.from_numpy(np.asarray(self._v0_list, dtype=np.float32).ljust_pad(self.max_tris)
                               if hasattr(np, "ljust_pad") else self._pad_vec3(self._v0_list, self.max_tris))
            self.v1.from_numpy(self._pad_vec3(self._v1_list, self.max_tris))
            self.v2.from_numpy(self._pad_vec3(self._v2_list, self.max_tris))
            self.tri_normal.from_numpy(self._pad_vec3(self._n_list, self.max_tris))
            self.tri_refl.from_numpy(self._pad_scalar(self._refl_list, self.max_tris, np.float32))
            self.tri_box_id.from_numpy(self._pad_scalar(self._box_id_list, self.max_tris, np.int32))

        if self.num_boxes > 0:
            self.box_min.from_numpy(self._pad_vec3(self._bmin_list, self.max_boxes))
            self.box_max.from_numpy(self._pad_vec3(self._bmax_list, self.max_boxes))

    @staticmethod
    def _pad_vec3(items, target_len):
        arr = np.zeros((target_len, 3), dtype=np.float32)
        if items:
            src = np.asarray(items, dtype=np.float32)
            arr[:len(src)] = src
        return arr

    @staticmethod
    def _pad_scalar(items, target_len, dtype):
        arr = np.zeros(target_len, dtype=dtype)
        if items:
            src = np.asarray(items, dtype=dtype)
            arr[:len(src)] = src
        return arr