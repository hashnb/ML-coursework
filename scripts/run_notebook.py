"""Execute a project notebook with the current interpreter in a fresh kernel."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def run_notebook(path):
    path = ROOT / path
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(notebook, km=manager, timeout=600,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
        nbformat.write(notebook, path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    print(f"Executed {path.name} from a fresh kernel.")


if __name__ == "__main__":
    run_notebook(sys.argv[1] if len(sys.argv) > 1 else "notebooks/02_model_comparison.ipynb")
