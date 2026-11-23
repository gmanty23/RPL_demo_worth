# RPL Demo Worth — Isaac Lab Project & Custom Assets

Isaac Lab extension and custom scanned object assets (`bowl`, `box`, `glass1`, `glass2`, `mug`, `mug_rack`) normalized for simulation (**meters scale, Z-up, bottom-centered origin at `(0, 0, 0)`**).

Do not commit Isaac Sim or Isaac Lab to this repository. Install them locally using the pinned versions below.

---

## Hardware, Drivers & Pinned Versions

* **GPU:** NVIDIA RTX GPU with **16+ GB VRAM recommended** (6 GB laptops will OOM when rendering scenes/cameras).
* **NVIDIA Driver:** **580.x branch** (validated for Isaac Sim 5.1.0). **Avoid Driver 595.x**, which segfaults in `librtx.scenedb.plugin.so` whenever rendering is enabled.
* **Lab Workstation Baseline:** Ubuntu 26.04.1 LTS (Kernel `7.0.0-43-generic`), NVIDIA TITAN RTX (24 GB VRAM), Driver `580.178.04` (`nvidia-driver-580-open`, Secure Boot disabled), ~15 GB RAM (keep `--num_envs 1` during local dev).
* **Pinned Software:**
  * **Python:** `3.11` (`conda create -n env_isaaclab python=3.11 -y`)
  * **PyTorch:** `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`
  * **TensorDict:** `tensordict==0.9.1` (pinned for PyTorch 2.7.0 compatibility)
  * **Isaac Sim:** `isaacsim[all,extscache]==5.1.0`
  * **Isaac Lab:** `v2.3.2` (cloned outside this repo at `~/repos/IsaacLab`)
  * **Imitation Learning:** `robomimic==0.4.0` (via `./isaaclab.sh -i robomimic`)

---

## Repository Layout

* `objects.yaml` — Source of truth for all 6 objects (mesh paths, collision types, USD approximations, and masses).
* `meshes/visual/<name>/` — Normalized visual `.obj`, `.mtl`, and `.jpg` textures (tracked with Git LFS).
* `meshes/collision/<name>_coll.obj` — Pre-computed collision meshes (box primitive, CoACD convex hulls, and thin-walled frustums).
* `meshes/usd/` — Generated locally (ignored by Git to keep the repo lightweight). Contains `render/<name>/<name>.usd`, `physics/<name>/<name>.usd`, and combined `render_sim/<name>/<name>.usd` twins.
* `scripts/tools/` — Utilities for collision generation, batch USD conversion, compound glass USD generation, alignment auditing, and physics testing.
* `source/rpl_demo_worth/` — Isaac Lab Python extension package.

---

## Quickstart for Teammates

### 1. Clone This Repo (Git LFS Required)
Large `.obj` and `.jpg` texture assets are tracked with **Git LFS**. Without Git LFS installed, they will download as tiny text pointers and fail to load. Clone outside the `IsaacLab` directory:

```bash
sudo apt update && sudo apt install -y git-lfs cmake build-essential
git lfs install
git clone https://github.com/gmanty23/RPL_demo_worth.git
cd RPL_demo_worth
git lfs pull
```

### 2. Install Isaac Sim, Isaac Lab & Project Package
```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install uv
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10 \
  uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

cd ~/repos
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
./isaaclab.sh -i robomimic

# Install this external project in editable mode (adjust path to your clone)
python -m pip install -e <PATH_TO_REPO>/source/rpl_demo_worth
```

### 3. Generate All USD Assets Locally
Because `meshes/usd/` is excluded from Git, run the batch converter once (~2 minutes total):

```bash
cd <PATH_TO_REPO>
conda activate env_isaaclab

# Optional: set ISAACLAB_PATH if your IsaacLab repo is not at ~/repos/IsaacLab
# export ISAACLAB_PATH=/path/to/your/IsaacLab

python scripts/tools/convert_all_to_usd.py
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/build_render_twin_usds.py
```

This builds both `meshes/usd/render/<name>/<name>.usd` (visuals, no collision) and `meshes/usd/physics/<name>/<name>.usd` (collision + mass, including the 17-hull compound rigid bodies for `glass1` and `glass2`).

### 4. Verify Installation & Assets
> **First-launch note:** The first time you open the Isaac Sim GUI (or after changing NVIDIA drivers), it compiles its Vulkan/RTX shader cache and can appear frozen for **60–90 seconds**. Do not press `Ctrl+C`—subsequent launches take ~10–15 seconds.

```bash
# 1. Check project task registration (do not test with bare `python -c "import rpl_demo_worth"`)
python scripts/list_envs.py

# 2. Check that Render and Physics USDs match in size, origin, and mass
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/check_usd_alignment.py

# 3. Visual side-by-side comparison in Isaac Sim GUI (textured Render vs. Physics collision)
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_phase_e.py --mode compare

# 4. Tabletop drop stability test (all 6 objects)
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_phase_e.py --mode drop

# 5. Clearance & nesting test (box -> bowl, glass1 -> glass2)
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_phase_e.py --mode nest
```

---

## Asset Summary & PhysX Settings

| Object | Mass (kg) | Collision Mesh | USD Collision Approx | Notes |
| :--- | :---: | :--- | :--- | :--- |
| `bowl` | 0.25 | CoACD (12 hulls) | `convexDecomposition` | Open interior cavity |
| `box` | 0.03 | Box primitive | `boundingCube` | Fits inside `bowl` |
| `glass1` | 0.15 | Compound frustum (17 hulls) | `compoundConvexHull` | Nests cleanly inside `glass2` (zero jitter) |
| `glass2` | 0.15 | Compound frustum (17 hulls) | `compoundConvexHull` | Receives `glass1` (zero jitter) |
| `mug` | 0.30 | CoACD (12 hulls) | `convexDecomposition` | Open cavity & handle loop (`map_Bump` removed from `mug.mtl` to avoid UV shadow seams) |
| `mug_rack` | 0.50 | CoACD (12 hulls) | `convexDecomposition` | Angled pegs preserved |

**Recommended PhysX Settings (for Task Configs):**
* `dt = 1.0 / 120.0` (120 Hz), `enable_stabilization = True`
* `solver_position_iteration_count = 16`, `solver_velocity_iteration_count = 2`
* `contact_offset = 0.002` (2 mm), `rest_offset = 0.0`
* `linear_damping = 0.5`, `angular_damping = 0.5`, `sleep_threshold = 0.01`, `stabilization_threshold = 0.01`

---

## Registered Isaac Lab Tasks

Each task provides a **Physics** copy (`-v0`) and a **Render** twin (`-Render-v0`) that combines high-fidelity visual meshes with simplified physics colliders:

| Task | Physics Env ID | Render Twin Env ID |
| :--- | :--- | :--- |
| **Block in Bowl** | `RPL-BlockInBowl-v0` | `RPL-BlockInBowl-Render-v0` |
| **Mug on Rack** | `RPL-MugOnRack-v0` | `RPL-MugOnRack-Render-v0` |
| **Cup in Cup** | `RPL-CupInCup-v0` | `RPL-CupInCup-Render-v0` |

| Block in Bowl | Mug on Rack | Cup in Cup |
| :---: | :---: | :---: |
| ![Block in Bowl](docs/images/block_in_bowl.png) | ![Mug on Rack](docs/images/mug_on_rack.png) | ![Cup in Cup](docs/images/cup_in_cup.png) |

All tasks share a Franka Panda arm (relative Differential IK), `table_cam`, and `wrist_cam` defined in `source/rpl_demo_worth/rpl_demo_worth/tasks/manager_based/common/base_env_cfg.py`.

> **Camera flag required:** Because `table_cam` and `wrist_cam` are spawned in every scene, always pass `--enable_cameras` when launching a task.

```bash
# Verify physics, success/failure terminations, and camera snapshots (outputs/cam_check/)
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_block_in_bowl.py --task RPL-BlockInBowl-Render-v0
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_mug_on_rack.py --task RPL-MugOnRack-Render-v0
~/repos/IsaacLab/isaaclab.sh -p scripts/tools/test_cup_in_cup.py --task RPL-CupInCup-Render-v0

# Standard Isaac Lab entry points (pass --enable_cameras)
python scripts/zero_agent.py --task=RPL-BlockInBowl-Render-v0 --enable_cameras
python scripts/random_agent.py --task=RPL-BlockInBowl-Render-v0 --enable_cameras
```

---

## Known Issues & Fixes

### General (All Linux Machines)
1. **Timeouts downloading `isaacsim` wheels from `pypi.nvidia.com`:** Install with `uv` using `UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10`.
2. **`flatdict==4.0.1` build fails (`No module named 'pkg_resources'`):** Downgrade `setuptools<81`, install `flatdict` with `--no-build-isolation`, then run `pip install -e ~/repos/IsaacLab/source/isaaclab`.
3. **`egl_probe==1.0.2` build fails (CMake `< 3.5` policy error):** Run `CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install egl_probe==1.0.2`, then re-run `./isaaclab.sh -i robomimic`.
4. **RL dependencies upgrade PyTorch and break `tensordict`:** Re-pin PyTorch and `tensordict`:
   ```bash
   pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   pip install click==8.1.7 typing_extensions==4.12.2 psutil==5.9.8 packaging==23.0
   pip install --no-deps tensordict==0.9.1
   ```
5. **`replay_demos.py --headless` crashes on `Se3Keyboard`:** Guard `Se3Keyboard(...)` with `if not args_cli.headless:` (initializing `teleop_interface = None`) and guard `teleop_interface.reset()` with `if teleop_interface is not None:`.
6. **Git LFS HTTP/2 error on GitHub push (`locks/verify`):** Run `git config lfs.locksverify false` and authenticate via a GitHub PAT (`repo` scope) or SSH key.

### Ubuntu 26.04 / Kernel 7.0 & Driver 595 Specific
7. **RTX renderer / GUI segfault (`librtx.scenedb.plugin.so`):** Downgrade from Driver `595.x` to `nvidia-driver-580-open` (`580.178.04`). If Secure Boot blocks the DKMS module after reboot, disable Secure Boot in BIOS or enroll the MOK key.
8. **Missing `libxml2.so.2` (`omni.kit.tool.asset_importer` fails to load):** Install `libxml2=2.13.*` in Conda and **append** (do not prepend) `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`:
   ```bash
   conda install -c conda-forge "libxml2=2.13.*" -y
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib' > $CONDA_PREFIX/etc/conda/activate.d/libxml2.sh
   ```

### IDE & Formatting (Optional)
* **VSCode indexing:** Press `Ctrl+Shift+P` -> `Tasks: Run Task` -> `setup_python_env` (creates `.vscode/.python.env`). If Pylance crashes from OOM, comment out unused `omni.*` packages in `.vscode/settings.json`.
* **Pre-commit formatting:** `pip install pre-commit && pre-commit run --all-files`.
