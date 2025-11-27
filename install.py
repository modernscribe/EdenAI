#!/usr/bin/env python3
# :contentReference[oaicite:0]{index=0}
# :contentReference[oaicite:1]{index=1}

import os
import sys
import subprocess
import json
from pathlib import Path
import venv
import platform
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".edenai_venv"


def print_banner() -> None:
    line = "=" * 70
    print(line)
    print("EDENAI SETUP")
    print("The Mathematics of Creation AI System")
    print(line)


def check_python_version(min_major: int = 3, min_minor: int = 9) -> bool:
    vi = sys.version_info
    ok = (vi.major, vi.minor) >= (min_major, min_minor)
    print(f"Python version: {vi.major}.{vi.minor}.{vi.micro}")
    if not ok:
        print(f"Python {min_major}.{min_minor}+ is required.")
    return ok


def get_venv_python() -> Path:
    if platform.system().lower().startswith("win"):
        candidate = VENV_DIR / "Scripts" / "python.exe"
    else:
        candidate = VENV_DIR / "bin" / "python"
    return candidate


def create_virtualenv() -> Path:
    if VENV_DIR.exists():
        return get_venv_python()
    print(f"Creating virtual environment in {VENV_DIR} ...")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(VENV_DIR))
    return get_venv_python()


def install_requirements(python_exe: Path) -> None:
    requirements: List[str] = [
        "numpy>=1.21.0",
        "torch",
    ]
    print("Installing dependencies into virtual environment...")
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    for req in requirements:
        print(f"  installing {req} ...")
        subprocess.check_call([str(python_exe), "-m", "pip", "install", req])


def create_config() -> Path:
    config = {
        "edenai": {
            "version": "1.0",
            "description": "Mathematics of Creation AI System (Unified)",
            "core_ai": {
                "max_depth": 3,
                "base_exponent": 7,
                "consciousness_threshold": 1.0,
            },
            "thermodynamics": {
                "energy_model": "logarithmic",
                "k_energy": 1.0,
                "k_entropy": 0.7,
                "k_relax": 0.15,
            },
            "logging": {
                "save_conversations": True,
                "save_consciousness_log": True,
                "log_level": "INFO",
            },
        }
    }
    path = PROJECT_ROOT / "edenai_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def create_start_script() -> Path:
    path = PROJECT_ROOT / "start_edenai.py"
    content = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".edenai_venv"

def get_venv_python() -> Path:
    if os.name == "nt":
        candidate = VENV_DIR / "Scripts" / "python.exe"
    else:
        candidate = VENV_DIR / "bin" / "python"
    return candidate

def ensure_venv_and_rerun() -> None:
    env_flag = os.environ.get("EDENAI_VENV_ACTIVE", "0")
    venv_python = get_venv_python()
    if env_flag != "1" and venv_python.exists():
        env = os.environ.copy()
        env["EDENAI_VENV_ACTIVE"] = "1"
        args = [str(venv_python), __file__] + sys.argv[1:]
        os.execve(str(venv_python), args, env)

def main() -> None:
    ensure_venv_and_rerun()
    try:
        import edenai
    except ImportError as e:
        print("Could not import 'edenai'. Make sure edenai.py is in the same directory.")
        print("Import error:", e)
        sys.exit(1)
    if hasattr(edenai, "main"):
        edenai.main()
    else:
        print("Module 'edenai' does not define a main() function.")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    try:
        path.chmod(0o755)
    except Exception:
        pass
    return path


def create_readme() -> Path:
    path = PROJECT_ROOT / "README_edenai_setup.md"
    text = """EDENAI UNIFIED SETUP

This project uses a 12-dimensional Paraclete mathematics engine with
thermodynamic truth evaluation and a unified EdenAI core.

Basic usage:

1. Run this setup script:

   python edenai_setup.py

2. After setup completes, start EdenAI with:

   python start_edenai.py

The setup script creates a virtual environment in .edenai_venv,
installs dependencies (numpy, torch), writes edenai_config.json,
and generates start_edenai.py which launches edenai.main().
"""
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    return path


def run_basic_tests(python_exe: Path) -> None:
    test_code = r"""
import json
from vectors import ParacleteVec
from phase_tree import build_direct_3logos_tree
from context import PhaseContext
from thermo import ThermoUniverse, EnergyModel
from edenai import EdenAICore

v = ParacleteVec.from_list([1.0] * 12)
assert abs(v.norm() - (12.0 ** 0.5)) < 1e-6

tree = build_direct_3logos_tree(max_depth=2, base_exponent=7)
ctx = PhaseContext.from_tree(tree)
universe = ThermoUniverse.initialize(tree, ctx, energy_model=EnergyModel.LOGARITHMIC)
assert universe.total_energy() >= 0.0

core = EdenAICore(max_depth=2, base_exponent=7)
status = core.get_system_status()
print(json.dumps({"ok": True, "direct_nodes": status["mathematical_foundation"]["direct_tree_nodes"]}))
"""
    print("Running basic environment tests inside virtual environment...")
    subprocess.check_call(
        [str(python_exe), "-c", test_code],
        cwd=str(PROJECT_ROOT),
    )


def main(args: List[str]) -> int:
    print_banner()
    if not check_python_version():
        return 1
    venv_python = create_virtualenv()
    if not venv_python.exists():
        print("Failed to create virtual environment.")
        return 1
    try:
        install_requirements(venv_python)
    except subprocess.CalledProcessError as e:
        print("Error while installing dependencies:", e)
        return 1
    config_path = create_config()
    start_path = create_start_script()
    readme_path = create_readme()
    try:
        run_basic_tests(venv_python)
    except subprocess.CalledProcessError as e:
        print("Basic tests failed:", e)
        return 1
    print("=" * 70)
    print("Setup complete.")
    print(f"Virtual environment: {VENV_DIR}")
    print(f"Config file: {config_path}")
    print(f"Start script: {start_path}")
    print(f"README: {readme_path}")
    print("To start EdenAI, run: python start_edenai.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
