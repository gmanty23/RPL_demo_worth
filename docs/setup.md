# Environment Setup & Pinned Versions

Do not commit Isaac Sim or Isaac Lab to this repository. Install them locally using the pinned versions below.

## Lab Workstation Baseline
- **OS:** Ubuntu 26.04.1 LTS (Kernel 7.0.0-43-generic)
- **GPU:** NVIDIA TITAN RTX (24 GB VRAM)
- **NVIDIA Driver:** 595.99.02
- **System RAM:** ~15 GB (keep `--num_envs 1` during local development)

## Pinned Versions
- **Python:** 3.11 (`conda create -n env_isaaclab python=3.11 -y`)
- **PyTorch:** `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128`
- **Isaac Sim:** `isaacsim[all,extscache]==5.1.0`
- **Isaac Lab:** `v2.3.2` (cloned outside this repo at `~/repos/IsaacLab`)
- **Imitation Learning:** `robomimic==0.4.0` (via `./isaaclab.sh -i robomimic`)

## Install Commands
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
```

## Known Issues & Fixes Encountered
1. **Timeouts downloading `isaacsim` wheels from `pypi.nvidia.com`:**
   - *Fix:* Installed via `uv` with `UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10`.
2. **`flatdict==4.0.1` failed (`ModuleNotFoundError: No module named 'pkg_resources'`):**
   - *Fix:* Downgraded `setuptools<81`, built `flatdict` with `--no-build-isolation`, then ran `pip install -e ~/repos/IsaacLab/source/isaaclab`.
3. **`egl_probe==1.0.2` failed (CMake `< 3.5` policy error):**
   - *Fix:* `CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install egl_probe==1.0.2`, then re-ran `./isaaclab.sh -i robomimic`.
4. **RL dependencies upgraded PyTorch to `2.15.0` and broke Isaac Sim pins:**
   - *Fix:* Re-pinned `torch==2.7.0 torchvision==0.22.0` (`cu128`) and `click==8.1.7 typing_extensions==4.12.2 psutil==5.9.8 packaging==23.0`.
5. **Missing `libxml2.so.2` on Ubuntu 26.04 (`omni.kit.tool.asset_importer` / URDF / MJCF failed to load):**
   - *Fix:* `conda install -c conda-forge "libxml2=2.13.*" -y` and export `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
6. **GUI viewport segfault (`librtx.scenedb.plugin.so`) on TITAN RTX + Ubuntu 26.04:**
   - *Fix:* Run simulations with `--headless`. For `scripts/tools/replay_demos.py --headless`, guarded `Se3Keyboard` creation and `teleop_interface.reset()` with `if not args_cli.headless:` / `if teleop_interface is not None:`.
