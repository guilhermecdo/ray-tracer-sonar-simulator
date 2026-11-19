import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from modules.mesh import SonarScene
from modules.sonar_sim import GPUFLSSimulator


def polar_to_cartesian_fan(polar_img: np.ndarray, az_fov_deg: float, min_range: float, max_range: float, out_size: int = 512):
    """Remaps (az_bins, range_bins) polar intensity into a 2D Cartesian FLS fan image."""
    az_bins, range_bins = polar_img.shape
    half_az = np.radians(az_fov_deg) * 0.5

    ys = np.linspace(0.0, max_range, out_size)
    xs = np.linspace(-max_range * np.sin(half_az), max_range * np.sin(half_az), out_size)
    xx, yy = np.meshgrid(xs, ys)

    rng = np.sqrt(xx ** 2 + yy ** 2)
    az = np.arctan2(xx, yy)

    fan = np.zeros((out_size, out_size), dtype=np.float32)
    mask = (rng >= min_range) & (rng <= max_range) & (np.abs(az) <= half_az)

    u = ((az[mask] + half_az) / (2.0 * half_az)) * (az_bins - 1)
    v = ((rng[mask] - min_range) / (max_range - min_range)) * (range_bins - 1)

    u_idx = np.clip(np.round(u).astype(np.int32), 0, az_bins - 1)
    v_idx = np.clip(np.round(v).astype(np.int32), 0, range_bins - 1)
    fan[mask] = polar_img[u_idx, v_idx]
    return fan


def main():
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    scene = SonarScene(max_tris=512, max_boxes=16)

    def update_scene(box_yaw_deg: float):
        scene.clear()
        # Seafloor plane-like slab below sensor
        scene.add_box_mesh(center=[6.0, 0.0, -2.5], size=[14.0, 14.0, 0.2], reflectivity=0.45)
        # High-reflectivity target block on seafloor (rotated to simulate 360 deg sonar orbit)
        scene.add_box_mesh(
            center=[6.5, 0.8, -1.8],
            size=[1.2, 1.2, 1.4],
            reflectivity=0.95,
            yaw_deg=box_yaw_deg,
        )
        scene.upload()

    update_scene(0.0)

    az_fov_deg = 60.0
    min_range = 0.5
    max_range = 12.0

    sim = GPUFLSSimulator(
        scene=scene,
        az_rays=256,
        el_rays=64,
        az_bins=256,
        range_bins=512,
        az_fov_deg=az_fov_deg,
        el_fov_deg=20.0,
        min_range=min_range,
        max_range=max_range,
        attenuation_db_per_km=12.0,
        multipath_gain=0.6,
    )

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    polar_img = sim.render_ping(sonar_pos=[0.0, 0.0, 0.0], yaw_deg=0.0, pitch_deg=-12.0, seed=0)
    fan_img = polar_to_cartesian_fan(polar_img, az_fov_deg, min_range, max_range, out_size=512)

    im_polar = axes[0].imshow(polar_img.T, origin="lower", aspect="auto", cmap="bone", vmin=0.0, vmax=1.0)
    axes[0].set_title("Polar Image (Azimuth vs. Range)")
    axes[0].set_xlabel("Azimuth Bin")
    axes[0].set_ylabel("Range Bin")

    im_fan = axes[1].imshow(fan_img, origin="lower", cmap="bone", vmin=0.0, vmax=1.0)
    axes[1].set_title("Cartesian FLS Fan Display")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y Forward Range (m)")

    plt.tight_layout()

    num_steps = 72
    for step in range(num_steps):
        if not plt.fignum_exists(fig.number):
            break

        angle_deg = (360.0 * step) / num_steps
        # Rotate box by -angle_deg so it appears as if the sonar orbits +angle_deg around the box
        update_scene(box_yaw_deg=-angle_deg)

        polar_img = sim.render_ping(sonar_pos=[0.0, 0.0, 0.0], yaw_deg=0.0, pitch_deg=-12.0, seed=step)
        fan_img = polar_to_cartesian_fan(polar_img, az_fov_deg, min_range, max_range, out_size=512)

        im_polar.set_data(polar_img.T)
        im_fan.set_data(fan_img)
        fig.suptitle(f"360° Orbit Around Box — Angle: {angle_deg:.1f}°")
        fig.canvas.draw_idle()
        plt.pause(0.03)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()