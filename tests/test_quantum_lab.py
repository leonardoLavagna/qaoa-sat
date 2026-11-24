import unittest
from itertools import product
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from quantum_lab import *


class QuantumLabTests(unittest.TestCase):
    def test_classical_spectrum_and_solutions(self):
        np.testing.assert_array_equal(np.unique(costs(),return_counts=True)[1],[2,12,2])
        self.assertEqual([s for s,c in zip(labels(),costs()) if c==4],["0101","1010"])
        self.assertEqual(costs(DIAGONAL).max(),4)

    def test_analytic_qaoa_and_reference_distribution(self):
        rng=np.random.default_rng(13)
        for beta,gamma in rng.uniform(-4,4,(20,2)):
            state=qaoa([beta],[gamma])
            self.assertAlmostEqual(expectation(state,costs()),2+np.sin(4*beta)*np.sin(2*gamma),places=12)
        p=np.abs(qaoa([np.pi/8],[np.pi/4]))**2
        expected=np.array([1,1,1,5,1,17,5,1,1,5,17,1,5,1,1,1])/64
        np.testing.assert_allclose(p,expected,atol=1e-14)

    def test_qiskit_independent_circuit_and_endianness(self):
        for edges in (SQUARE,DIAGONAL,((0,1),(1,3))):
            for betas,gammas in [([.17],[.73]),([.31,-.67],[.41,.29])]:
                expected=Statevector.from_instruction(qiskit_qaoa(betas,gammas,edges)).data
                np.testing.assert_allclose(qaoa(betas,gammas,edges),expected,atol=1e-13)

    def test_threshold_cnf(self):
        for edges in (SQUARE,DIAGONAL):
            for k in range(len(edges)+2):
                clauses,nvars=threshold_cnf(edges,k)
                for x,c in zip(assignments(),costs(edges)):
                    y=[x[i]^x[j] for i,j in edges]
                    self.assertEqual(satisfies(list(x)+y,clauses),c>=k)
            for x,c in zip(assignments(),costs(edges)):
                self.assertEqual(satisfies(x,all_edges_cnf(edges)),c==len(edges))
        # Exhaust the auxiliary bit as well: XOR clauses must force its value.
        clauses,_=threshold_cnf(((0,1),),0,n=2)
        for a,b,y in product((0,1),repeat=3):
            self.assertEqual(satisfies([a,b,y],clauses),y==(a^b))

    def test_oracle_all_basis_states_and_clean_workspace(self):
        oracle=square_oracle()
        for x in range(16):
            initial=np.zeros(2**11,complex);initial[x]=1
            actual=Statevector(initial).evolve(oracle).data
            expected=initial*(-1 if costs()[x]==4 else 1)
            np.testing.assert_allclose(actual,expected,atol=1e-14)
        initial=np.zeros(2**11,complex);initial[:16]=uniform()
        actual=Statevector(initial).evolve(oracle).data
        expected=np.zeros(2**11,complex);expected[:16]=uniform()*np.where(costs()==4,-1,1)
        np.testing.assert_allclose(actual,expected,atol=1e-14)

    def test_grover_reference_and_degenerate_cases(self):
        marked=costs()==4
        for r,expected in enumerate([1/8,25/32,121/128,169/512]):
            actual=grover_trace(marked,r)[-1][1]
            self.assertAlmostEqual(float((np.abs(actual)**2)[marked].sum()),expected,places=13)
        for m in (np.zeros(16,bool),np.ones(16,bool),costs()>=2):
            for r in range(6):
                state=grover_trace(m,r)[-1][1]
                self.assertAlmostEqual(np.linalg.norm(state),1,places=13)
                self.assertAlmostEqual(float((np.abs(state)**2)[m].sum()),grover_probability(m.sum(),16,r),places=12)

    def test_grover_operator_identity(self):
        s=uniform(); projector=np.outer(s,s.conj())
        f=np.diag((costs()==4).astype(float))
        d=2*projector-np.eye(16);o=np.eye(16)-2*f
        layer=expm(-1j*np.pi*projector)@expm(-1j*np.pi*f)
        np.testing.assert_allclose(layer,-d@o,atol=1e-14)
        for r in (1,2,3):
            actual=alternating(costs()==4,[np.pi]*r,[np.pi]*r,"projector")
            np.testing.assert_allclose(actual,(-1)**r*grover_trace(costs()==4,r)[-1][1],atol=1e-13)
        # The standard mixer at pi is identity for four qubits.
        np.testing.assert_allclose(x_mixer(s,np.pi),s,atol=1e-14)

    def test_optimizer_and_sampling(self):
        result,_=optimize_qaoa()
        self.assertAlmostEqual(-result.fun,3,places=7)
        state=qaoa([np.pi/8],[np.pi/4]);counts=sample(state,1024,7)
        self.assertEqual(counts.sum(),1024)
        np.testing.assert_array_equal(counts,sample(state,1024,7))


if __name__=="__main__":unittest.main()
