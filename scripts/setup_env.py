import subprocess
import sys
from pathlib import Path

REQ_FILE = Path(__file__).resolve().parents[1] / "requirements.txt"


def install_package(requirement: str) -> None:
    """Install a single pip requirement if it's not already importable."""
    pkg_name = requirement.strip().split("==")[0].split(">=")[0]
    if not pkg_name or pkg_name.startswith("#"):
        return

    try:
        __import__(pkg_name)
        print(f"[OK] {pkg_name} already installed")
    except ImportError:
        print(f"[INSTALL] {requirement}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])


def main() -> None:
    if not REQ_FILE.exists():
        print(f"requirements.txt not found at {REQ_FILE}")
        return

    with REQ_FILE.open() as f:
        requirements = [line.strip() for line in f if line.strip()]

    for req in requirements:
        try:
            install_package(req)
        except subprocess.CalledProcessError as e:
            # Fail gracefully but continue with other packages
            print(f"[WARN] Failed to install {req}: {e}")


if __name__ == "__main__":
    main()
