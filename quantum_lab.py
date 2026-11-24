"""Simulazione esatta di piccole istanze; stringhe x1...xn da sinistra a destra.

La base NumPy è big-endian. In Qiskit x1 corrisponde al qubit n-1,
così le stringhe stampate mantengono lo stesso significato.
"""
from itertools import combinations, product
import numpy as np

SQUARE = ((0, 1), (1, 2), (2, 3), (3, 0))
DIAGONAL = SQUARE + ((0, 2),)


def assignments(n=4):
    return np.array(list(product((0, 1), repeat=n)), dtype=int)


def labels(n=4):
    return [format(i, f"0{n}b") for i in range(2**n)]


def costs(edges=SQUARE, n=4):
    bits = assignments(n)
    return sum((bits[:, i] ^ bits[:, j] for i, j in edges), np.zeros(2**n, dtype=int))


def uniform(n=4):
    return np.full(2**n, 1 / np.sqrt(2**n), dtype=complex)


def x_mixer(state, beta):
    """Applica exp(-i beta sum_j X_j) senza costruire una matrice densa."""
    state = np.asarray(state, dtype=complex).copy()
    n = int(np.log2(len(state)))
    indices = np.arange(len(state))
    for j in range(n):
        state = np.cos(beta) * state - 1j * np.sin(beta) * state[indices ^ (1 << j)]
    return state


def projector_mixer(state, beta):
    """exp(-i beta |s><s|)."""
    return state + (np.exp(-1j * beta) - 1) * np.mean(state)


def alternating(diagonal, betas, gammas, mixer="x"):
    if len(betas) != len(gammas):
        raise ValueError("Each layer requires one beta and one gamma.")
    if mixer not in ("x", "projector"):
        raise ValueError("Unknown mixer.")
    state = uniform(int(np.log2(len(diagonal))))
    mix = x_mixer if mixer == "x" else projector_mixer
    for beta, gamma in zip(betas, gammas):
        state *= np.exp(-1j * gamma * np.asarray(diagonal))
        state = mix(state, beta)
    return state


def qaoa(betas, gammas, edges=SQUARE):
    return alternating(costs(edges), betas, gammas)


def expectation(state, diagonal):
    return float(np.dot(np.abs(state)**2, diagonal))


def grover_trace(marked, iterations):
    state = uniform(int(np.log2(len(marked))))
    trace = [("Iniziale", state.copy())]
    for r in range(1, iterations + 1):
        state = state * np.where(marked, -1, 1)
        trace.append((f"Oracolo {r}", state.copy()))
        state = 2 * np.mean(state) - state
        trace.append((f"Diffusione {r}", state.copy()))
    return trace


def grover_probability(marked_count, size, iterations):
    theta = np.arcsin(np.sqrt(marked_count / size))
    return float(np.sin((2 * iterations + 1) * theta)**2)


def sample(state, shots=1024, seed=7):
    p = np.abs(state)**2
    return np.random.default_rng(seed).multinomial(shots, p / p.sum())


def all_edges_cnf(edges=SQUARE):
    """CNF 2-SAT che impone di tagliare tutti gli archi (soglia K=m)."""
    return [clause for i, j in edges for clause in ((i+1, j+1), (-i-1, -j-1))]


def threshold_cnf(edges=SQUARE, threshold=4, n=4):
    """Impone y_e <-> x_i XOR x_j e sum(y_e)>=K.

    La cardinalità usa sottoinsiemi: una codifica leggibile per grafi piccoli,
    ma potenzialmente esponenziale. L'appendice A presenta un contatore
    sequenziale di dimensione polinomiale.
    """
    clauses = []
    m = len(edges)
    for k, (i, j) in enumerate(edges):
        a, b, y = i+1, j+1, n+k+1
        clauses += [(a, b, -y), (-a, -b, -y), (a, -b, y), (-a, b, y)]
    if threshold > m:
        clauses.append(())  # Una clausola vuota è sempre falsa.
    elif threshold > 0:
        clauses += list(combinations(range(n+1, n+m+1), m-threshold+1))
    return clauses, n+m


def satisfies(bits, clauses):
    return all(any(bool(bits[abs(lit)-1]) == (lit > 0) for lit in clause) for clause in clauses)


def dimacs(clauses, nvars):
    return f"p cnf {nvars} {len(clauses)}\n" + "\n".join(" ".join(map(str, c)) + " 0" for c in clauses) + "\n"


def optimize_qaoa(p=1, edges=SQUARE, seed=7, starts=4):
    """Ottimizza la media esatta; le valutazioni non usano shots finiti."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    c = costs(edges)
    best = None
    histories = []
    for start in range(starts):
        history = []
        x0 = rng.uniform(0, np.pi, 2*p)
        def objective(theta):
            value = expectation(qaoa(theta[:p], theta[p:], edges), c)
            history.append(value)
            return -value
        result = minimize(objective, x0, method="BFGS", options={"maxiter":200, "gtol":1e-8})
        histories.append(history)
        if best is None or result.fun < best.fun:
            best = result
    return best, histories


def qiskit_qaoa(betas, gammas, edges=SQUARE, n=4):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for beta, gamma in zip(betas, gammas):
        # Conserva la fase costante per confrontare direttamente le ampiezze.
        qc.global_phase -= gamma * len(edges) / 2
        for i, j in edges:
            qc.rzz(-gamma, n-1-i, n-1-j)
        qc.rx(2*beta, range(n))
    return qc


def square_oracle():
    """11 qubit: 4 dati, 4 archi, 2 AND intermedi e 1 risultato.

    Le ancille partono da zero. Il predicato è calcolato dagli archi,
    senza inserire esplicitamente gli assegnamenti soddisfacenti.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    x = QuantumRegister(4, "x")
    e = QuantumRegister(4, "e")
    t = QuantumRegister(2, "t")
    q = QuantumRegister(1, "q")
    comp = QuantumCircuit(x, e, t, q)
    for k, (i, j) in enumerate(SQUARE):
        comp.cx(x[3-i], e[k])
        comp.cx(x[3-j], e[k])
    comp.ccx(e[0], e[1], t[0])
    comp.ccx(e[2], e[3], t[1])
    comp.ccx(t[0], t[1], q[0])
    oracle = comp.copy()
    oracle.z(q[0])
    oracle.compose(comp.inverse(), inplace=True)
    return oracle
