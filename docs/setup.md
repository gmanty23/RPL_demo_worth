# Environment Setup & Pinned Versions

Do not commit Isaac Sim or Isaac Lab to this repository. Install them locally using the pinned versions below.

## Hardware & Driver Requirements
- **GPU:** NVIDIA RTX GPU with **16+ GB VRAM recommended** (6 GB laptops will OOM when rendering scenes/cameras).
- **NVIDIA Driver:** **580.x branch** (validated for Isaac Sim 5.1.0). **Avoid Driver 595.x**, which segfaults in `librtx.scenedb.plugin.so` whenever rendering is enabled.
- **Lab Workstation Baseline:**
  - **OS:** Ubuntu 26.04.1 LTS (Kernel `7.0.0-43-generic`)
  - **GPU:** NVIDIA TITAN RTX (24 GB VRAM)
  - **Driver:** `580.178.04` (`nvidia-driver-580-open`, Secure Boot disabled)
  - **System RAM:** ~15 GB (keep `--num_envs 1` during local development)

## Pinned Versions
- **Python:** 3.11 (`conda create -n env_isaaclab python=3.11 -y`)
- **PyTorch:** `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`
- **TensorDict:** `tensordict==0.9.1` (pinned for PyTorch 2.7.0 compatibility)
- **Isaac Sim:** `isaacsim[all,extscache]==5.1.0`
- **Isaac Lab:** `v2.3.2` (cloned outside this repo at `~/repos/IsaacLab`)
- **Imitation Learning:** `robomimic==0.4.0` (via `./isaaclab.sh -i robomimic`)

---

## 1. Clone This Repo (Git LFS Required)
This repository tracks `.usd`, `.usdc`, `.obj`, and image assets with **Git LFS**. Without Git LFS installed, asset files will download as tiny text pointers and fail to load in Isaac Sim.
```bash
sudo apt update && sudo apt install -y git-lfs cmake build-essential
git lfs install
git clone [https://github.com/gmanty23/RPL_demo_worth.git](https://github.com/gmanty23/RPL_demo_worth.git)
cd RPL_demo_worth
git lfs pull
```

## 2. Install Isaac Sim, Isaac Lab & Project Package
```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
pip install uv
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10 \
  uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com)

cd ~/repos
git clone --branch v2.3.2 [https://github.com/isaac-sim/IsaacLab.git](https://github.com/isaac-sim/IsaacLab.git)
cd IsaacLab
./isaaclab.sh --install
./isaaclab.sh -i robomimic

# Install this external project in editable mode (adjust path to your clone)
python -m pip install -e <PATH_TO_REPO>/source/rpl_demo_worth
```

---

## 3. Verify Your Installation
> **First-launch note:** The first time you open the Isaac Sim GUI (or after changing NVIDIA drivers), it compiles its Vulkan/RTX shader cache and can appear frozen for **60–90 seconds**. Do not press `Ctrl+C`—subsequent launches take ~10–15 seconds.

1. **Check project task registration** (do not test with bare `python -c "import rpl_demo_worth"`—`pxr` only loads after `AppLauncher` starts):
   ```bash
   cd <PATH_TO_REPO>
   python scripts/list_envs.py
   ```
2. **Check RTX GUI rendering:**
   ```bash
   cd ~/repos/IsaacLab
   ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
   ```
3. **Check `robomimic` + HDF5 demo replay:**
   ```bash
   cd ~/repos/IsaacLab
   ./isaaclab.sh -p scripts/tools/replay_demos.py \
     --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
     --dataset_file ./datasets/dataset.hdf5 \
     --headless
   ```
4. **Headless video recording** (for machines without a monitor):
   ```bash
   cd ~/repos/IsaacLab
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Cartpole-v0 --num_envs 1 --max_iterations 2 \
     --headless --enable_cameras --video --video_length 100
   ```

---

## 4. Known Issues & Fixes

### General (All Linux Machines)
1. **Timeouts downloading `isaacsim` wheels from `pypi.nvidia.com`:**
   - *Fix:* Install with `uv` using `UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10`.
2. **`flatdict==4.0.1` build fails (`ModuleNotFoundError: No module named 'pkg_resources'`):**
   - *Fix:* Downgrade `setuptools<81`, install `flatdict` with `--no-build-isolation`, then run `pip install -e ~/repos/IsaacLab/source/isaaclab`.
3. **`egl_probe==1.0.2` build fails (CMake `< 3.5` policy error):**
   - *Fix:* `CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install egl_probe==1.0.2`, then re-run `./isaaclab.sh -i robomimic`.
4. **RL dependencies upgrade PyTorch and break Isaac Sim / `rsl_rl` pins:**
   - *Symptoms:* Version conflicts or `ImportError: cannot import name 'is_structseq_instance' from 'torch.utils._pytree'` in `tensordict`.
   - *Fix:* Re-pin PyTorch and `tensordict`:
     ```bash
     pip install -U torch==2.7.0 torchvision==0.22.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
     pip install click==8.1.7 typing_extensions==4.12.2 psutil==5.9.8 packaging==23.0
     pip install --no-deps tensordict==0.9.1
     ```
5. **`replay_demos.py --headless` crashes on `Se3Keyboard`:**
   - *Fix:* Guard `Se3Keyboard(...)` creation with `if not args_cli.headless:` (initializing `teleop_interface = None`) and guard `teleop_interface.reset()` with `if teleop_interface is not None:`.
6. **Git LFS HTTP/2 error on GitHub push (`locks/verify`):**
   - *Fix:* Run `git config lfs.locksverify false` and authenticate via a GitHub Personal Access Token (`repo` scope) or SSH key.

### Ubuntu 26.04 / Kernel 7.0 & Driver 595 Specific *(skip on Ubuntu 22.04/24.04 if unaffected)*
7. **RTX renderer / GUI segfault (`librtx.scenedb.plugin.so`):**
   - *Cause:* NVIDIA Driver `595.99.02` is incompatible with Isaac Sim 5.1.0 whenever rendering is enabled (GUI or `--headless --enable_cameras`).
   - *Fix:* Downgrade to `nvidia-driver-580-open` (`580.178.04`). If Secure Boot blocks the DKMS module after reboot (stretched screen / `nvidia-smi` fails), disable Secure Boot in BIOS or enroll the MOK key.
8. **Missing `libxml2.so.2` (`omni.kit.tool.asset_importer` / URDF / MJCF fail to load):**
   - *Fix:* Install `libxml2=2.13.*` in Conda and **append** (do not prepend, to avoid `libtinfo.so.6` bash warnings) `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` on activation:
     ```bash
     conda install -c conda-forge "libxml2=2.13.*" -y
     mkdir -p $CONDA_PREFIX/etc/conda/activate.d
     echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib' >$CONDA_PREFIX/etc/conda/activate.d/libxml2.sh
     ```
