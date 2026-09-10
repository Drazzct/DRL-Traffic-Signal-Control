"""
Test script to verify that the modules can be imported without syntax errors.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import traffic_drl
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
        return True
    except Exception as e:
        print(f"  ✗ {module_name}: {e}")
        return False

def main():
    print("Testing imports...")
    modules = [
        "traffic_drl.config",
        "traffic_drl.run_id",
        "traffic_drl.environment.make_env",
        "traffic_drl.environment.wrappers",
        "traffic_drl.train.train_dqn",
        "traffic_drl.evaluation.evaluate_benchmark",
        "traffic_drl.contracts",
    ]

    all_passed = True
    for module in modules:
        if not test_import(module):
            all_passed = False

    if all_passed:
        print("\nAll imports succeeded.")
    else:
        print("\nSome imports failed.")

    # Also check that the notebook can be imported as a module? Not necessary.
    # Instead, let's check the notebook file exists and is valid JSON.
    notebook_path = Path(__file__).parent.parent / "notebooks" / "demo_orchestration.ipynb"
    if notebook_path.exists():
        import json
        try:
            with open(notebook_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Notebook is valid JSON with {len(data.get('cells', []))} cells.")
        except Exception as e:
            print(f"✗ Notebook is invalid JSON: {e}")
            all_passed = False
    else:
        print("✗ Notebook not found.")
        all_passed = False

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)