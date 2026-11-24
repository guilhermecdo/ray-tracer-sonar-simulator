import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from modules.mesh import SonarScene
from modules.sonar_sim import GPUFLSSimulator


def polar_to_cartesian_fan(
    polar_img: np.ndarray,
    az_fov_deg: float,
    min_range: float,
    max_range: float,
    out_size: int = 512,
) -> np.ndarray:
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_scenario(scenario_name: str, cfg: dict):
    scenes_dir = cfg.get("scenes_dir", "scenes")
    npz_path = os.path.join(scenes_dir, f"{scenario_name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Missing exported scene: {npz_path}. "
            "Run blender/blender2scene.py on your .blend file first."
        )

    limits = cfg.get("scene_limits", {})
    scene = SonarScene(
        max_tris=int(limits.get("max_tris", 32768)),
        max_boxes=int(limits.get("max_boxes", 64)),
    )
    meta = scene.load_blender_scene(npz_path)

    sonar_cfg = cfg.get("sonar", {})
    sim = GPUFLSSimulator(
        scene=scene,
        az_rays=int(sonar_cfg.get("az_rays", 256)),
        el_rays=int(sonar_cfg.get("el_rays", 128)),
        az_bins=int(sonar_cfg.get("az_bins", 256)),
        range_bins=int(sonar_cfg.get("range_bins", 512)),
        az_fov_deg=meta["az_fov_deg"],
        el_fov_deg=meta["el_fov_deg"],
        min_range=meta["min_range"],
        max_range=meta["max_range"],
        attenuation_db_per_km=float(sonar_cfg.get("attenuation_db_per_km", 12.0)),
        multipath_gain=float(sonar_cfg.get("multipath_gain", 2.0)),
        max_bounces=int(sonar_cfg.get("max_bounces", 3)),
        roughness=float(sonar_cfg.get("roughness", 0.25)),
        log_gain=float(sonar_cfg.get("log_gain", 12.0)),
    )

    disp_cfg = cfg.get("display", {})
    out_size = int(disp_cfg.get("fan_out_size", 512))
    cmap_name = disp_cfg.get("cmap", "bone")
    pause_sec = float(disp_cfg.get("pause_sec", 0.5))
    save_dir = disp_cfg.get("save_dir", None)

    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, scenario_name), exist_ok=True)

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    cam_pos = meta["cam_pos"]
    cam_yaw = meta["cam_yaw_deg"]
    cam_pitch = meta["cam_pitch_deg"]
    cam_az = meta["cam_az_deg"]

    polar_img, depth_img = sim.render_ping_with_depth(
        cam_pos[0], yaw_deg=float(cam_yaw[0]), pitch_deg=float(cam_pitch[0]), seed=0
    )
    fan_img = polar_to_cartesian_fan(
        polar_img, meta["az_fov_deg"], meta["min_range"], meta["max_range"], out_size=out_size
    )
    depth_vis = np.where(depth_img > 0.0, depth_img, np.nan)

    im_fan = axes[0].imshow(fan_img, origin="lower", cmap=cmap_name, vmin=0.0, vmax=1.0)
    #axes[0].set_title("Cartesian FLS Fan Display")
    #axes[0].set_xlabel("X (m)")
    #axes[0].set_ylabel("Y Forward Range (m)")

    cmap_depth = plt.get_cmap(cmap_name).copy()
    cmap_depth.set_bad(color="black")
    im_depth = axes[1].imshow(
        depth_vis.T,
        origin="lower",
        aspect="auto",
        cmap=cmap_depth,
        vmin=meta["min_range"],
        vmax=meta["max_range"],
    )
    #axes[1].set_title("GT Raycast Depth (Azimuth vs. Elevation)")
    #axes[1].set_xlabel("Azimuth Ray")
    #axes[1].set_ylabel("Elevation Ray")
    fig.colorbar(im_depth, ax=axes[1], label="Range (m)")

    plt.tight_layout()

    num_views = len(cam_pos)
    for idx in range(num_views):
        if not plt.fignum_exists(fig.number):
            break

        polar_img, depth_img = sim.render_ping_with_depth(
            cam_pos[idx],
            yaw_deg=float(cam_yaw[idx]),
            pitch_deg=float(cam_pitch[idx]),
            seed=idx,
        )
        fan_img = polar_to_cartesian_fan(
            polar_img, meta["az_fov_deg"], meta["min_range"], meta["max_range"], out_size=out_size
        )
        depth_vis = np.where(depth_img > 0.0, depth_img, np.nan)

        im_fan.set_data(fan_img)
        im_depth.set_data(depth_vis.T)
        #fig.suptitle(
        #    f"Scenario: {scenario_name} | View {idx + 1}/{num_views} | Orbit Az: {cam_az[idx]:.0f}°"
        #)
        fig.canvas.draw_idle()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, scenario_name, f"view_{idx:03d}.png"), dpi=120)

        plt.pause(pause_sec)

    plt.ioff()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run FLS sonar simulation from config_simulation.json.")
    parser.add_argument(
        "--config",
        default="config_simulation.json",
        help="Path to simulation JSON config (default: config_simulation.json).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    arch_str = cfg.get("taichi", {}).get("arch", "gpu").lower()
    arch = ti.cpu if arch_str == "cpu" else ti.gpu
    ti.init(arch=arch, default_fp=ti.f32)

    scenarios = cfg.get("scenarios", ["coral"])
    if isinstance(scenarios, str):
        scenarios = [scenarios]

    for name in scenarios:
        run_scenario(name, cfg)


if __name__ == "__main__":
    main()
