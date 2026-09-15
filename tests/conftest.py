"""pytest configuration: set SUMO_HOME before any test module is collected.

This must run before any import of sumo_rl, which raises ImportError at module
level if SUMO_HOME is not set.  conftest.py is executed by pytest before test
collection begins, so os.environ mutations here are visible to all test modules.

Supported install methods (tried in order):
  1. pip install eclipse-sumo  -- sumo/ dir lives in site-packages
  2. conda install sumo        -- data at $CONDA_PREFIX/share/sumo
  3. apt / brew / native       -- standard FHS paths under /usr or /opt
  4. sumo binary on PATH       -- derive from binary location
"""
import os
import sys
import shutil
import sysconfig
from pathlib import Path


def _is_sumo_home(path: Path) -> bool:
    return path.is_dir() and (path / "data").is_dir() and (path / "tools").is_dir()


def _find_sumo_home() -> str | None:  # noqa: C901
    """Search for the SUMO home directory using multiple strategies."""

    # ------------------------------------------------------------------
    # Strategy 1: pip install eclipse-sumo
    #   The wheel places all SUMO data under <site-packages>/sumo/.
    #   We only accept site-packages directories that are absolute paths
    #   so we never accidentally match a project-local 'sumo/' folder.
    # ------------------------------------------------------------------
    site_dirs: list[Path] = []
    for scheme_key in ("purelib", "platlib"):
        p = Path(sysconfig.get_path(scheme_key))
        if p.is_absolute() and p not in site_dirs:
            site_dirs.append(p)
    for entry in sys.path:
        p = Path(entry)
        if p.is_absolute() and p.is_dir() and p not in site_dirs:
            site_dirs.append(p)

    for site_dir in site_dirs:
        candidate = site_dir / "sumo"
        if _is_sumo_home(candidate):
            return str(candidate)

    # ------------------------------------------------------------------
    # Strategy 1b: pip install eclipse-sumo into the BASE interpreter
    #   when running inside a venv with include-system-site-packages=false.
    #   Read pyvenv.cfg to find the base interpreter home, then look in
    #   its Lib/site-packages (Windows) or lib/pythonX.Y/site-packages (Unix).
    # ------------------------------------------------------------------
    pyvenv_cfg = Path(sys.prefix) / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        base_home: Path | None = None
        for line in pyvenv_cfg.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("home"):
                _, _, val = line.partition("=")
                base_home = Path(val.strip())
                break
        if base_home:
            # base_home is the Scripts/ or bin/ directory of the base Python.
            # The site-packages is typically one level up on Windows (Lib/site-packages)
            # and two levels up on Unix (lib/pythonX.Y/site-packages).
            base_root = base_home  # e.g. C:\Python314  or  C:\Users\...\pythoncore-3.14-64
            for sp_rel in (
                Path("Lib") / "site-packages",               # Windows
                Path("lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",  # Unix
                Path("lib") / "site-packages",               # some conda / pip layouts
            ):
                candidate = base_root / sp_rel / "sumo"
                if _is_sumo_home(candidate):
                    return str(candidate)

    #   SUMO data lives at $CONDA_PREFIX/share/sumo (all platforms).
    # ------------------------------------------------------------------
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "share" / "sumo"
        if _is_sumo_home(candidate):
            return str(candidate)

    # ------------------------------------------------------------------
    # Strategy 3: standard system/package-manager paths
    #   apt install sumo  → /usr/share/sumo
    #   brew install sumo → /opt/homebrew/share/sumo  (Apple Silicon)
    #                       /usr/local/share/sumo      (Intel Mac)
    #   Windows installer → C:/Program Files/Eclipse/Sumo
    # ------------------------------------------------------------------
    system_candidates: list[Path] = [
        # Linux apt / rpm
        Path("/usr/share/sumo"),
        Path("/usr/local/share/sumo"),
        # macOS Homebrew (Intel and Apple Silicon)
        Path("/usr/local/opt/sumo/share/sumo"),
        Path("/opt/homebrew/opt/sumo/share/sumo"),
        Path("/opt/homebrew/share/sumo"),
        # Windows native installer default locations
        Path("C:/Program Files/Eclipse/Sumo"),
        Path("C:/Program Files (x86)/Eclipse/Sumo"),
        # Interpreter prefix share dir (covers venv / conda envs where
        # sumo was installed directly into the environment root)
        Path(sys.prefix) / "share" / "sumo",
    ]
    for candidate in system_candidates:
        if _is_sumo_home(candidate):
            return str(candidate)

    # ------------------------------------------------------------------
    # Strategy 4: derive from sumo binary on PATH
    #   Handles any install where the binary is on PATH but the data
    #   directory is not in a standard location.
    #
    #   FHS layout (Linux/macOS): <prefix>/bin/sumo → <prefix>/share/sumo
    #   Windows installer layout: <SUMO_HOME>/bin/sumo.exe → <SUMO_HOME>
    # ------------------------------------------------------------------
    sumo_bin = shutil.which("sumo") or shutil.which("sumo.exe")
    if sumo_bin:
        bin_path = Path(sumo_bin).resolve()
        # Unix FHS: binary is <prefix>/bin/sumo, data at <prefix>/share/sumo
        fhs_candidate = bin_path.parent.parent / "share" / "sumo"
        if _is_sumo_home(fhs_candidate):
            return str(fhs_candidate)
        # Windows installer: binary is <SUMO_HOME>/bin/sumo.exe
        win_candidate = bin_path.parent.parent
        if _is_sumo_home(win_candidate):
            return str(win_candidate)

    return None


if "SUMO_HOME" not in os.environ:
    sumo_home = _find_sumo_home()
    if sumo_home:
        os.environ["SUMO_HOME"] = sumo_home
