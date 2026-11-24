# Ray-Tracer Sonar Simulator

GPU-accelerated 2-D Forward-Looking Sonar (FLS) ray-tracer built with [Taichi](https://www.taichi-lang.org/), featuring:
- **DIDSON Radiometric & Multipath Model**: Implements the 2-D lens-based diffuse reflectance model ($I = k \cos\alpha \cos\phi\,\delta\phi$, TVG compensation, finite pulse-width smearing, and logarithmic compression) with multi-bounce multipath and rough-surface acoustic scattering.
- **Ground-Truth Raycast Depth**: Displays both the simulated Cartesian FLS fan image and the per-ray ground-truth depth map.
- **Blender Integration**: Author 3D underwater scenes and camera orbits in Blender and export them directly to the simulator.

## Installation

### 1. Conda Environment
```bash
conda create -n sonar-sim python=3.11 -y
conda activate sonar-sim
pip install -r requirements.txt
```

### 2. Blender (Optional, for `.blend` scene authoring/export)
Install [Blender](https://www.blender.org/download/) if you want to create or convert `.blend` scenes. On macOS, `blender/blender2scene.py` automatically detects `/Applications/Blender.app/Contents/MacOS/Blender` (or `blender` on `PATH`).

## Usage

### 1. Configure and Run the Simulation
All simulation parameters, scenarios, and display options are configured in `config_simulation.json`.

Run the simulator:
```bash
python simulate_sonar.py
# Or specify a custom config file:
python simulate_sonar.py --config config_simulation.json
```

Key fields in `config_simulation.json`:
- `scenes_dir`: Directory containing exported `.npz` scenes (default: `"scenes"`).
- `scenarios`: List of scene names to run, e.g. `["coral"]` or `["cube", "sphere", "concave_half_sphere", "coral"]`.
- `taichi.arch`: `"gpu"` or `"cpu"`.
- `sonar`: Ray counts (`az_rays`, `el_rays`), bin resolution (`az_bins`, `range_bins`), attenuation (`attenuation_db_per_km`), and multipath/reverberation parameters (`multipath_gain`, `max_bounces`, `roughness`, `log_gain`).
- `display`: `fan_out_size`, `cmap`, `pause_sec`, and optional `save_dir` for saving rendered PNG frames.

### 2. Convert Any `.blend` File to a Scene
Use `blender/blender2scene.py` to export meshes, camera FOV/clip range, and animated (or static) camera poses from any `.blend` file into `scenes/<name>.npz`:

```bash
python blender/blender2scene.py scenes/coral.blend
# Optional: control per-mesh triangle budget (default: 10000)
python blender/blender2scene.py scenes/my_scene.blend --max-tris 10000 --out scenes/my_scene.npz
```

### 3. Generate the Built-in Sample Blender Scenes
Creates `cube`, `sphere`, and `concave_half_sphere` (`.blend` and `.npz`) in `scenes/`:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python blender/create_and_export_scenes.py
```

## Project Structure
- `simulate_sonar.py` — Main entry point that loads `config_simulation.json` and runs the FLS simulation.
- `config_simulation.json` — Central configuration file for scenarios, Taichi backend, sonar physics, and display settings.
- `requirements.txt` — Python package dependencies (`numpy`, `matplotlib`, `taichi`).
- `modules/mesh.py` — `SonarScene` geometry container and `.npz` loader.
- `modules/sonar_sim.py` — `GPUFLSSimulator` Taichi ray-tracer, multipath bounce loop, and DIDSON intensity model.
- `blender/blender2scene.py` — CLI converter from any `.blend` file to `.npz` (with automatic mesh decimation).
- `blender/create_and_export_scenes.py` — Generates the `cube`, `sphere`, and `concave_half_sphere` `.blend`/`.npz` scenarios.
- `meshes/` — Source 3D mesh assets (e.g. `coral.stl`).
- `scenes/` — Exported `.blend` and `.npz` scenario files.
