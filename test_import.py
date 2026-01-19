import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
try:
    from simulation_run_v1.models.model_v10 import build_index_projection
    print("Found build_index_projection!")
except ImportError as e:
    print(f"ImportError: {e}")
except NameError as e:
    print(f"NameError: {e}")
except Exception as e:
    print(f"Error: {e}")
