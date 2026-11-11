"""
Cross-platform launcher for Multi-Robot Task Allocation in Webots
Supports:
- Experiment selection (Experiments 1, 2, 3, 4)
- GUI interactive mode or headless fast batch mode
- Standard paper timeline or accelerated demo timeline
"""

import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

def find_webots_executable():
    """Find the Webots executable on the current system."""
    which_path = shutil.which("webots")
    if which_path:
        return which_path

    if sys.platform.startswith("linux"):
        candidate_paths = [
            Path("/snap/bin/webots"),
            Path("/usr/local/bin/webots"),
            Path("/usr/bin/webots"),
        ]
        for p in candidate_paths:
            if p.exists() and os.access(p, os.X_OK):
                return str(p)

    elif sys.platform == "darwin":
        app_path = Path("/Applications/Webots.app")
        if app_path.exists():
            return str(app_path)

    elif sys.platform == "win32":
        default_win_path = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Webots" / "msys64" / "mingw64" / "bin" / "webots.exe"
        if default_win_path.exists():
            return str(default_win_path)

    return None

def launch_webots(world_file, experiment=1, quick=False, headless=False, nu=0.0):
    """Launch Webots with selected experiment and options."""
    world_path = Path(world_file).resolve()
    if not world_path.exists():
        print(f"❌ ERROR: World file not found: {world_path}")
        sys.exit(1)

    webots_exec = find_webots_executable()
    if not webots_exec:
        print("❌ ERROR: Webots not found on this system.")
        sys.exit(1)

    # Set environment variables for the supervisor and robot controllers
    env = os.environ.copy()
    env["EXPERIMENT"] = str(experiment)
    env["EXP_QUICK"] = "1" if quick else "0"
    env["HEADLESS"] = "1" if headless else "0"
    env["NU"] = str(nu)

    exp_names = {
        1: "Experiment 1: Temporary Surge in Area 1 (w1=5.0)",
        2: "Experiment 2: Permanent Growth Increase in Area 1 (w1=5.0)",
        3: "Experiment 3: 15 Robot Failures (Attrition & Recovery)",
        4: f"Experiment 4: Payoff Parameter nu={nu}"
    }

    print("=" * 65)
    print("LAUNCHING MULTI-ROBOT TASK ALLOCATION IN WEBOTS")
    print(f"  Configuration: {exp_names.get(experiment, 'Experiment ' + str(experiment))}")
    print(f"  Mode:          {'Headless Fast Batch' if headless else 'GUI Interactive Window'}")
    print(f"  Timeline:      {'Quick Demo (100s)' if quick else 'Paper Scale (1000s–2000s)'}")
    print(f"  Webots Binary: {webots_exec}")
    print(f"  World:         {world_path.name}")
    print("=" * 65)

    try:
        cmd = [webots_exec]
        if headless:
            cmd.extend(["--mode=fast", "--no-rendering", "--batch", "--stdout", "--stderr"])
        cmd.append(str(world_path))

        print("\nExecuting simulation... Press Ctrl+C to stop.\n")
        subprocess.run(cmd, env=env, check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation finished with exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Robot Task Allocation Webots Simulation Runner")
    parser.add_argument("-e", "--experiment", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Experiment to run: 1 (temporary surge), 2 (permanent increase), 3 (15 robot failures), 4 (nu parameter)")
    parser.add_argument("-q", "--quick", action="store_true", default=False,
                        help="Use accelerated demo timeline (100s) for live demonstration")
    parser.add_argument("-b", "--headless", action="store_true", default=False,
                        help="Run headless in fast mode without 3D window (auto-saves data & plots)")
    parser.add_argument("--nu", type=float, default=0.0,
                        help="Payoff parameter nu for Experiment 4 (0, 40, or 800)")

    args = parser.parse_args()

    world_file = "worlds/trash_collection.wbt"
    if not Path(world_file).exists():
        print("❌ ERROR: Not in project directory. Run from project root.")
        sys.exit(1)

    launch_webots(world_file, experiment=args.experiment, quick=args.quick,
                  headless=args.headless, nu=args.nu)