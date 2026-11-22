# RPL Demo Worth — Isaac Lab Project & Custom Assets

Isaac Lab extension and custom scanned object assets (`bowl`, `box`, `glass1`, `glass2`, `mug`, `mug_rack`) normalized for simulation (**meters scale, Z-up, bottom-centered origin at `(0, 0, 0)`**).

## Repository Layout

* `objects.yaml` — Source of truth for all 6 objects (mesh paths, collision types, USD approximations, and masses).
* `meshes/visual/<name>/` — Normalized visual `.obj`, `.mtl`, and `.jpg` textures (tracked with Git LFS).
* `meshes/collision/<name>_coll.obj` — Pre-computed collision meshes (box primitive, CoACD convex hulls, and thin-walled frustums).
* `meshes/usd/` — Generated locally (ignored by Git to keep the repo lightweight). Contains paired `render/<name>/<name>.usd` and `physics/<name>/<name>.usd` files.
* `scripts/tools/` — Utilities for collision generation, batch USD conversion, compound glass USD generation, alignment auditing, and physics testing.
* `source/rpl_demo_worth/` — Isaac Lab Python extension package.

---

## Quickstart for Teammates

### 1. Install Isaac Lab
Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (conda or `uv` recommended).

### 2. Clone the repository (with Git LFS)
Large mesh and texture files are stored in Git LFS. Clone outside the `IsaacLab` directory:

```bash
git lfs install
git clone https://github.com/gmanty23/RPL_demo_worth.git
cd RPL_demo_worth
```

*(If you already cloned without LFS, run `git lfs pull` inside the repo.)*

### 3. Install the extension in editable mode
Using the Python environment where Isaac Lab is installed:

```bash
conda activate env_isaaclab
python -m pip install -e source/rpl_demo_worth
```

### 4. Generate all USD assets locally
Because `.usd` files are excluded from Git, run the batch converter once (~2 minutes total):

```bash
# Optional: set ISAACLAB_PATH if your IsaacLab repo is not at ~/repos/IsaacLab
# export ISAACLAB_PATH=/path/to/your/IsaacLab

python scripts/tools/convert_all_to_usd.py
```

This automatically builds both `meshes/usd/render/<name>/<name>.usd` (visuals, no collision) and `meshes/usd/physics/<name>/<name>.usd` (collision + mass, including the 17-hull compound rigid bodies for `glass1` and `glass2`).

### 5. Verify assets & physics (optional)
```bash
# Check that Render and Physics USDs match in size, origin, and mass
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/check_usd_alignment.py

# Visual side-by-side comparison in Isaac Sim GUI
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_phase_e.py --mode compare

# Tabletop drop stability test (all 6 objects)
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_phase_e.py --mode drop

# Clearance & nesting test (box -> bowl, glass1 -> glass2)
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_phase_e.py --mode nest
```

---

## Asset Summary

| Object | Mass (kg) | Collision Mesh | USD Collision Approx | Notes |
| :--- | :---: | :--- | :--- | :--- |
| `bowl` | 0.25 | CoACD (12 hulls) | `convexDecomposition` | Open interior cavity |
| `box` | 0.03 | Box primitive | `boundingCube` | Fits inside `bowl` |
| `glass1` | 0.15 | Compound frustum (17 hulls) | `compoundConvexHull` | Nests cleanly inside `glass2` (zero jitter) |
| `glass2` | 0.15 | Compound frustum (17 hulls) | `compoundConvexHull` | Receives `glass1` (zero jitter) |
| `mug` | 0.30 | CoACD (12 hulls) | `convexDecomposition` | Open cavity & handle loop |
| `mug_rack` | 0.50 | CoACD (12 hulls) | `convexDecomposition` | Angled pegs preserved |

## Recommended PhysX Settings (for Task Configs)

Use these simulation and rigid-body settings in your Isaac Lab environment config for stable grasping and cup nesting:

* **Physics timestep:** `dt = 1.0 / 120.0` (120 Hz)
* **PhysX stabilization:** `enable_stabilization = True`
* **Solver iterations:** `solver_position_iteration_count = 16`, `solver_velocity_iteration_count = 2`
* **Collision offset:** `contact_offset = 0.002` (2 mm), `rest_offset = 0.0`
* **Damping / Sleep:** `linear_damping = 0.5`, `angular_damping = 0.5`, `sleep_threshold = 0.01`, `stabilization_threshold = 0.01`

---

## Running Isaac Lab Tasks

* **List available tasks:**
  ```bash
  python scripts/list_envs.py
  ```
* **Train a task:**
  ```bash
  python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
  ```
* **Run with dummy agents:**
  ```bash
  python scripts/zero_agent.py --task=<TASK_NAME>
  python scripts/random_agent.py --task=<TASK_NAME>
  ```

---

## Set up IDE (Optional)

Run VSCode Tasks by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task`, and running `setup_python_env`. When prompted, enter the absolute path to your Isaac Sim installation. This creates `.vscode/.python.env` for module indexing and autocomplete.

## Setup as Omniverse Extension (Optional)

1. Open `Window` -> `Extensions`, click the **Hamburger Icon** -> `Settings`.
2. Add the absolute paths to this repository's `source` directory and `IsaacLab/source` under `Extension Search Paths`, then click `Refresh`.
3. Search for your extension under the `Third Party` category and toggle it on.

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

## Troubleshooting

* **Pylance missing indexing:** Add `"<path-to-ext-repo>/source/rpl_demo_worth"` to `"python.analysis.extraPaths"` in `.vscode/settings.json`.
* **Pylance crash (out of memory):** Comment out unused Omniverse packages (`omni.anim.*`, `omni.kit.*`, `omni.graph.*`, `omni.services.*`) under `"python.analysis.extraPaths"` in `.vscode/settings.json`.
