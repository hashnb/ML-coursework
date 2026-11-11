"""Run the exploration notebook from a fresh kernel using this interpreter."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/01_data_exploration.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=180,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
        nbformat.write(notebook, path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    print("Exploration finished. Notebook, figures and tables saved.")


if __name__ == "__main__":
    main()
