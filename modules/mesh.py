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