import math
import numpy as np
import taichi as ti


@ti.data_oriented
class GPUFLSSimulator:
    def __init__(
        self,
        scene,
        az_rays: int = 256,
        el_rays: int = 64,
        az_bins: int = 256,
        range_bins: int = 512,
        az_fov_deg: float = 60.0,
        el_fov_deg: float = 20.0,
        min_range: float = 0.5,
        max_range: float = 15.0,
        attenuation_db_per_km: float = 10.0,
        multipath_gain: float = 1.5,
        max_bounces: int = 3,
        roughness: float = 0.22,
        log_gain: float = 12.0,
    ):
        self.scene = scene
        self.az_rays = az_rays
        self.el_rays = el_rays
        self.az_bins = az_bins
        self.range_bins = range_bins

        self.half_az = math.radians(az_fov_deg) * 0.5
        self.half_el = math.radians(el_fov_deg) * 0.5
        self.min_range = float(min_range)
        self.max_range = float(max_range)

        # Convert dB/km -> Np/km -> Np/m: gamma = 0.0115 * alpha (dB/km), then divide by 1000 for meters
        gamma_np_per_km = 0.0115 * float(attenuation_db_per_km)
        self.gamma_np_per_m = gamma_np_per_km / 1000.0
        self.multipath_gain = float(multipath_gain)
        self.max_bounces = int(max_bounces)
        self.roughness = float(roughness)
        self.log_gain = float(log_gain)

        self.polar_img = ti.field(dtype=ti.f32, shape=(az_bins, range_bins))
        self.noisy_img = ti.field(dtype=ti.f32, shape=(az_bins, range_bins))
        self.depth_img = ti.field(dtype=ti.f32, shape=(az_rays, el_rays))

    @ti.func
    def _hit_aabb(self, orig, dir_vec, bmin, bmax, t_max) -> ti.i32:
        hit = 1
        for k in ti.static(range(3)):
            if ti.abs(dir_vec[k]) < 1e-7:
                if orig[k] < bmin[k] or orig[k] > bmax[k]:
                    hit = 0
            else:
                inv_d = 1.0 / dir_vec[k]
                t1 = (bmin[k] - orig[k]) * inv_d
                t2 = (bmax[k] - orig[k]) * inv_d
                t_near = ti.min(t1, t2)
                t_far = ti.max(t1, t2)
                if t_near > 0.0 and t_near > t_max:
                    hit = 0
                if t_far < 0.0:
                    hit = 0
        return hit

    @ti.func
    def _moller_trumbore(self, orig, dir_vec, p0, p1, p2) -> ti.f32:
        e1 = p1 - p0
        e2 = p2 - p0
        h = dir_vec.cross(e2)
        a = e1.dot(h)
        t_out = -1.0
        if ti.abs(a) > 1e-6:
            f = 1.0 / a
            s = orig - p0
            u = f * s.dot(h)
            if 0.0 <= u <= 1.0:
                q = s.cross(e1)
                v = f * dir_vec.dot(q)
                if v >= 0.0 and (u + v) <= 1.0:
                    t_cand = f * e2.dot(q)
                    if t_cand > 1e-3:
                        t_out = t_cand
        return t_out

    @ti.func
    def _trace_closest(self, orig, dir_vec, max_dist):
        best_t = max_dist
        hit_tri = -1
        for t_idx in range(self.scene.num_tris):
            b_id = self.scene.tri_box_id[t_idx]
            if self._hit_aabb(orig, dir_vec, self.scene.box_min[b_id], self.scene.box_max[b_id], best_t) == 1:
                t_hit = self._moller_trumbore(
                    orig, dir_vec,
                    self.scene.v0[t_idx], self.scene.v1[t_idx], self.scene.v2[t_idx]
                )
                if 0.0 < t_hit < best_t:
                    best_t = t_hit
                    hit_tri = t_idx
        return best_t, hit_tri

    @ti.func
    def _attenuate(self, intensity, dist_m) -> ti.f32:
        # TVG/RVG compensates 1/R^2 spherical spreading (Aykin & Negahdaripour 2016, Eq. 5/36);
        # only exponential water absorption exp(-2 * gamma * R) remains.
        return intensity * ti.exp(-2.0 * self.gamma_np_per_m * dist_m)

    @ti.func
    def _deposit_polar(self, az_rad, range_m, intensity):
        if self.min_range <= range_m <= self.max_range and intensity > 0.0:
            az_norm = (az_rad + self.half_az) / (2.0 * self.half_az)
            r_norm = (range_m - self.min_range) / (self.max_range - self.min_range)
            b_az = int(az_norm * self.az_bins)
            b_r = int(r_norm * self.range_bins)
            if 0 <= b_az < self.az_bins and 0 <= b_r < self.range_bins:
                # Integral over elevation patches: sum_j k * cos(alpha) * cos(phi) * d_phi
                # Convolved with finite pulse width (Delta_pl ~ 3 * delta_R, Sec. III-B / IV-A)
                val = intensity * (16.0 / float(self.el_rays))
                w = 1.0 / 3.0
                for dr in ti.static(range(3)):
                    rr = b_r + dr - 1
                    if 0 <= rr < self.range_bins:
                        ti.atomic_add(self.polar_img[b_az, rr], val * w)

    @ti.kernel
    def step_ping(self, sonar_pos: ti.types.vector(3, ti.f32), yaw_rad: ti.f32, pitch_rad: ti.f32):
        for i, j in ti.ndrange(self.az_rays, self.el_rays):
            u_az = ((float(i) + 0.5) / float(self.az_rays)) * 2.0 - 1.0
            u_el = ((float(j) + 0.5) / float(self.el_rays)) * 2.0 - 1.0
            az = u_az * self.half_az
            el = u_el * self.half_el

            # Sensor frame: +X forward, +Y left, +Z up
            dir_local = ti.Vector([
                ti.cos(el) * ti.cos(az),
                ti.cos(el) * ti.sin(az),
                ti.sin(el),
            ])

            # Pitch (+pitch = nose up, -pitch = nose down), then Yaw around +Z
            cp, sp = ti.cos(pitch_rad), ti.sin(pitch_rad)
            cy, sy = ti.cos(yaw_rad), ti.sin(yaw_rad)
            dir_pitched = ti.Vector([
                cp * dir_local.x - sp * dir_local.z,
                dir_local.y,
                sp * dir_local.x + cp * dir_local.z,
            ])
            dir_world = ti.Vector([
                cy * dir_pitched.x - sy * dir_pitched.y,
                sy * dir_pitched.x + cy * dir_pitched.y,
                dir_pitched.z,
            ]).normalized()

            # 1. Primary ray hit
            r1, tri1 = self._trace_closest(sonar_pos, dir_world, self.max_range)
            self.depth_img[i, j] = r1 if tri1 >= 0 else 0.0
            if tri1 >= 0:
                p1 = sonar_pos + dir_world * r1
                n1 = self.scene.tri_normal[tri1].normalized()
                refl1 = self.scene.tri_refl[tri1]

                # Aykin & Negahdaripour (2016) Eq. (5)/(36): I = k * cos(alpha) * cos(phi) * d_phi
                cos_inc1 = ti.max(n1.dot(-dir_world), 0.0)
                cos_phi = ti.cos(el)
                i1 = self._attenuate(refl1 * cos_inc1 * cos_phi, r1)
                self._deposit_polar(az, r1, i1)

                # 2. Multi-bounce multipath & acoustic reverberation
                cur_pos = p1
                cur_dir = dir_world
                cur_n = n1
                cum_refl = refl1
                total_path = r1
                bounce_decay = 1.0

                for _b in range(self.max_bounces):
                    refl_dir = (cur_dir - 2.0 * cur_dir.dot(cur_n) * cur_n).normalized()
                    jitter = ti.Vector([ti.randn(ti.f32), ti.randn(ti.f32), ti.randn(ti.f32)])
                    bounce_dir = (refl_dir + self.roughness * jitter).normalized()
                    if bounce_dir.dot(cur_n) < 0.0:
                        bounce_dir = (bounce_dir - 2.0 * bounce_dir.dot(cur_n) * cur_n).normalized()

                    orig_next = cur_pos + cur_n * 0.005
                    r_next, tri_next = self._trace_closest(orig_next, bounce_dir, self.max_range)
                    if tri_next < 0:
                        break

                    p_next = orig_next + bounce_dir * r_next
                    total_path += r_next
                    d_back = (sonar_pos - p_next).norm()
                    r_multi = 0.5 * (total_path + d_back)

                    rel_next = p_next - sonar_pos
                    az_multi = ti.atan2(rel_next.y, rel_next.x) - yaw_rad

                    n_next = self.scene.tri_normal[tri_next].normalized()
                    refl_next = self.scene.tri_refl[tri_next]
                    cos_inc_next = ti.max(n_next.dot(-bounce_dir), 0.0)
                    cum_refl *= refl_next

                    i_multi = self._attenuate(
                        self.multipath_gain * bounce_decay * cum_refl * cos_inc_next * cos_phi,
                        r_multi,
                    )
                    self._deposit_polar(az_multi, r_multi, i_multi)

                    cur_pos = p_next
                    cur_dir = bounce_dir
                    cur_n = n_next
                    bounce_decay *= 0.68

    @ti.kernel
    def apply_speckle_and_clear(self, seed_offset: ti.i32):
        for i, j in ti.ndrange(self.az_bins, self.range_bins):
            clean_val = self.polar_img[i, j]
            # Granular multiplicative + small additive noise, then log compression (Sec. III-A)
            rng = ti.abs(ti.randn(ti.f32))
            mult_noise = 0.4 + 0.9 * rng
            add_noise = 0.015 * ti.abs(ti.randn(ti.f32))
            noisy_lin = ti.max(0.0, clean_val * mult_noise + add_noise)
            log_val = ti.log(1.0 + self.log_gain * noisy_lin) / ti.log(1.0 + self.log_gain)
            self.noisy_img[i, j] = ti.min(1.0, log_val)
            self.polar_img[i, j] = 0.0

    def render_ping(self, sonar_pos, yaw_deg: float = 0.0, pitch_deg: float = -12.0, seed: int = 0) -> np.ndarray:
        pos_vec = ti.Vector(np.asarray(sonar_pos, dtype=np.float32))
        self.step_ping(pos_vec, math.radians(yaw_deg), math.radians(pitch_deg))
        self.apply_speckle_and_clear(seed)
        return self.noisy_img.to_numpy()

    def render_ping_with_depth(self, sonar_pos, yaw_deg: float = 0.0, pitch_deg: float = -12.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Returns (polar_sonar_img, gt_raycast_depth_m) where 0.0 in depth indicates no hit within max_range."""
        polar = self.render_ping(sonar_pos, yaw_deg=yaw_deg, pitch_deg=pitch_deg, seed=seed)
        depth = self.depth_img.to_numpy()
        return polar, depth