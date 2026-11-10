"""Execute the saved notebook with this Python environment and a fresh kernel."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from download_data import ROOT


def main():
    path = ROOT / "01_data_and_eda.ipynb"
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
    client.execute()
    nbformat.write(notebook, path)
    print("Notebook executed successfully from a fresh kernel.")


if __name__ == "__main__":
    main()
