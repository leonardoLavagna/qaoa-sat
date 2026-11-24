"""Controlli indipendenti delle identità usate nella dispensa e nel notebook."""
import unittest
from functools import reduce
from itertools import product

import numpy as np

from quantum_lab import (
    SQUARE, DIAGONAL, assignments, costs, qaoa, uniform,
    grover_trace, grover_probability, expectation, sample,
)


I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])


def pauli(word):
    return reduce(np.kron, [dict(I=I, X=X, Y=Y, Z=Z)[letter] for letter in word])


class LessonMathTests(unittest.TestCase):
    def test_full_heisenberg_conjugations(self):
        for gamma in (-1.17, .2, .73, np.pi/4):
            uc = np.diag(np.exp(-1j*gamma*costs()))
            a, b = np.cos(gamma), np.sin(gamma)
            mixed = (a*a*pauli('ZYII')-a*b*pauli('ZXZI')
                     -a*b*pauli('IXII')-b*b*pauli('IYZI'))
            yy = (a*a*pauli('YYII')-a*b*(pauli('YXZI')+pauli('XYIZ'))
                  +b*b*pauli('XXZZ'))
            np.testing.assert_allclose(uc.conj().T@pauli('ZYII')@uc,mixed,atol=1e-14)
            np.testing.assert_allclose(uc.conj().T@pauli('YYII')@uc,yy,atol=1e-14)
            self.assertAlmostEqual(np.vdot(uniform(),mixed@uniform()).real,-a*b)
            self.assertAlmostEqual(abs(np.vdot(uniform(),yy@uniform())),0)

    def test_amplitudes_hamming_table_and_variance(self):
        state = qaoa([np.pi/8],[np.pi/4])
        table = {
            0: [(1,0,0),(0,4,0),(0,4,2),(0,4,0),(1,0,0)],
            1: [(0,1,0),(1,2,1),(0,6,0),(1,2,1),(0,1,0)],
            3: [(0,1,0),(0,4,0),(2,2,2),(0,4,0),(0,1,0)],
            5: [(0,0,1),(0,4,0),(2,4,0),(0,4,0),(0,0,1)],
        }
        expected = {0:1j/8,1:-1/8,3:-1/4-1j/8,5:-1/2+1j/8}
        for index, rows in table.items():
            distance = np.sum(assignments()!=assignments()[index],axis=1)
            actual = [[np.sum((distance==d)&(costs()==c)) for c in (0,2,4)] for d in range(5)]
            np.testing.assert_array_equal(actual,rows)
            self.assertAlmostEqual(abs(state[index]-expected[index]),0)
        self.assertAlmostEqual(expectation(state,costs()**2),41/4)
        self.assertAlmostEqual(expectation(state,(costs()-3)**2),5/4)

    def test_predicate_projector_and_topology_example(self):
        c, cd = costs(SQUARE), costs(DIAGONAL)
        pf = np.diag((c>=4).astype(float))
        hc = np.diag(c)
        polynomial = hc@(hc-2*np.eye(16))/8
        expansion = (pauli('IIII')-pauli('ZZII')-pauli('IZZI')-pauli('IIZZ')
                     -pauli('ZIIZ')+pauli('ZIZI')+pauli('IZIZ')+pauli('ZZZZ'))/8
        np.testing.assert_allclose(pf,polynomial)
        np.testing.assert_allclose(pf,expansion)
        np.testing.assert_array_equal(c>=4,cd>=4)
        for r in range(7):
            psi = grover_trace(c>=4,r)[-1][1]
            np.testing.assert_allclose(psi,grover_trace(cd>=4,r)[-1][1])
            p = grover_probability(2,16,r)
            self.assertAlmostEqual(expectation(psi,c),12/7+16*p/7)
        self.assertAlmostEqual(expectation(grover_trace(c>=4,2)[-1][1],cd),125/32)
        ps = np.abs(qaoa([np.pi/8],[np.pi/4]))**2
        pd = np.abs(qaoa([np.pi/8],[np.pi/4],DIAGONAL))**2
        self.assertFalse(np.allclose(ps,pd))
        self.assertAlmostEqual(pd[cd==4].sum(),11/32)

    def test_transcript_observables_and_symmetry(self):
        c = costs()
        # Una stessa media del costo non fissa la probabilità dell'ottimo.
        distributions = (np.array([0,1/2,1/2]),np.array([1/32,7/16,17/32]))
        for p in distributions:
            self.assertAlmostEqual(p@np.array([0,2,4]),3)
        for edges in (SQUARE,DIAGONAL):
            psi = qaoa([.23,-.17],[.34,.83],edges)
            np.testing.assert_allclose(psi,psi[::-1],atol=1e-14)
            self.assertAlmostEqual(expectation(psi,np.arange(16)),7.5)
        psi = grover_trace(c>=4,2)[-1][1]
        observed = sample(psi,256,19)
        expanded = np.repeat(np.arange(16),observed)
        self.assertAlmostEqual(observed@c/256,np.mean(c[expanded]))
        self.assertAlmostEqual(observed@(c>=4)/256,np.mean(c[expanded]>=4))

    def test_cardinality_appendix_gate_clauses(self):
        # Le equivalenze di porta dell'appendice devono valere su tutti gli input.
        for a,b,z in product((False,True),repeat=3):
            and_cnf = ((not z or a) and (not z or b) and (z or not a or not b))
            or_cnf = ((z or not a) and (z or not b) and (not z or a or b))
            self.assertEqual(and_cnf,z==(a and b))
            self.assertEqual(or_cnf,z==(a or b))
        for y in product((0,1),repeat=5):
            for k in range(1,6):
                r = {(j,0):True for j in range(6)}
                r.update({(0,t):False for t in range(1,k+1)})
                for j in range(1,6):
                    for t in range(1,k+1):
                        r[j,t] = r[j-1,t] or (r[j-1,t-1] and bool(y[j-1]))
                        self.assertEqual(r[j,t],sum(y[:j])>=t)
                self.assertEqual(r[5,k],sum(y)>=k)


if __name__ == '__main__':
    unittest.main()
