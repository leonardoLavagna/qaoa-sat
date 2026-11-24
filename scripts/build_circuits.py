"""Render Qiskit diagrams once, outside Streamlit's rerun threads.

The app simulates with NumPy. The notebook and tests execute Qiskit circuits.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qiskit.circuit import Parameter
from quantum_lab import SQUARE, DIAGONAL, qiskit_qaoa, square_oracle

assets=ROOT/'assets'
assets.mkdir(exist_ok=True)
for name,edges in [('square',SQUARE),('diagonal',DIAGONAL)]:
    for p in (1,2,3):
        bs=[Parameter(f'b{j+1}') for j in range(p)]
        gs=[Parameter(f'g{j+1}') for j in range(p)]
        circuit=qiskit_qaoa(bs,gs,edges)
        text='b_j = beta_j, g_j = gamma_j (radians)\n\n'+str(circuit.draw(output='text',fold=100))
        (assets/f'qaoa_{name}_p{p}.txt').write_text(text)
(assets/'square_oracle.txt').write_text(str(square_oracle().draw(output='text',fold=100)))
print('Generati 7 diagrammi Qiskit.')
