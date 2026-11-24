"""Esegue il notebook completo e salva soltanto un'esecuzione senza errori."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'tutorial_qaoa_sat_grover.ipynb'
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook, timeout=180, kernel_name='python3',
    resources={'metadata': {'path': str(ROOT)}},
).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f'Notebook eseguito: {sum(c.cell_type == "code" for c in notebook.cells)} celle di codice.')
