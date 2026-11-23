import pathlib
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]
OBJECTS_YAML = REPO_ROOT / "objects.yaml"


def load_objects_cfg() -> dict:
    return yaml.safe_load(OBJECTS_YAML.read_text())["objects"]


def get_object_usd(name: str, use_render: bool = False) -> str:
    """Single config switch between physics-only USD and textured render-twin USD."""
    subdir = "render_sim" if use_render else "physics"
    path = (REPO_ROOT / "meshes" / "usd" / subdir / name / f"{name}.usd").resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing USD: {path}. Run convert_all_to_usd.py and build_render_twin_usds.py first."
        )
    return str(path)
