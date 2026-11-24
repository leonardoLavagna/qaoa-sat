import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=str(Path(__file__).resolve().parents[1]/"app.py")


class AppTests(unittest.TestCase):
    def test_pages_graphs_and_thresholds(self):
        at=AppTest.from_file(APP,default_timeout=30).run()
        self.assertFalse(at.exception)
        pages=list(at.sidebar.radio[0].options)
        for graph in ("square","diagonal"):
            at.sidebar.selectbox[0].set_value(graph).run()
            for page in pages:
                at.sidebar.radio[0].set_value(page).run()
                self.assertFalse(at.exception,msg=f"{graph}: {page}")
        at.sidebar.radio[0].set_value(pages[3]).run()
        for k in (0,6):
            at.sidebar.slider[0].set_value(k).run()
            self.assertFalse(at.exception)

    def test_optimization_and_teacher_controls(self):
        at=AppTest.from_file(APP,default_timeout=30).run()
        at.sidebar.radio[0].set_value("2 · QAOA").run()
        at.button[1].click().run()
        self.assertFalse(at.exception)
        at.select_slider[0].set_value(2).run()
        at.button[1].click().run()
        self.assertFalse(at.exception)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        at.sidebar.toggle[0].set_value(True).run()
        self.assertFalse(at.exception)
        at.sidebar.radio[0].set_value("5 · Operators and measurements").run()
        self.assertEqual(at.metric[0].value,"1.0000000000")

    def test_same_counts_two_observables(self):
        at=AppTest.from_file(APP,default_timeout=30).run()
        at.sidebar.radio[0].set_value("5 · Operators and measurements").run()
        comparison=at.dataframe[0].value
        transcript=at.dataframe[1].value
        self.assertEqual(transcript['n_x'].sum(),1024)
        self.assertAlmostEqual(comparison.iloc[0,1],31/8)
        self.assertAlmostEqual(comparison.iloc[1,1],121/128)
        self.assertAlmostEqual(comparison.iloc[0,2],(transcript['n_x']*transcript['C(x)']).sum()/1024)
        self.assertAlmostEqual(comparison.iloc[1,2],(transcript['n_x']*transcript['f_K(x)']).sum()/1024)
        # Limiti del predicato: il costo medio resta un osservabile distinto.
        at.sidebar.slider[0].set_value(0).run()
        self.assertFalse(at.exception)
        self.assertAlmostEqual(at.dataframe[0].value.iloc[1,2],1)
        at.sidebar.slider[0].set_value(5).run()
        self.assertFalse(at.exception)
        self.assertAlmostEqual(at.dataframe[0].value.iloc[1,2],0)

    def test_english_walkthrough_and_operator_choices(self):
        at=AppTest.from_file(APP,default_timeout=30).run()
        self.assertEqual(at.sidebar.selectbox[0].options,["Square C₄","Square + diagonal"])
        at.select_slider[0].set_value('0011').run()
        self.assertEqual(at.metric[0].value,'2')
        at.select_slider[0].set_value('0101').run()
        self.assertEqual(at.metric[0].value,'4')
        at.sidebar.radio[0].set_value('2 · QAOA').run()
        at.button[0].click().run()
        self.assertEqual(at.metric[0].value,'3.0000')
        self.assertEqual(at.metric[1].value,'53.125%')
        next(s for s in at.slider if s.label=='β1 / π').set_value(0.).run()
        self.assertEqual(at.metric[0].value,'2.0000')
        self.assertEqual(at.metric[1].value,'12.500%')
        at.button[0].click().run()
        at.sidebar.radio[0].set_value('3 · From MaxCut to SAT').run()
        at.sidebar.selectbox[0].set_value('diagonal').run()
        at.sidebar.slider[0].set_value(5).run()
        self.assertEqual(at.metric[2].value,'UNSAT')
        at.sidebar.slider[0].set_value(4).run()
        self.assertEqual(at.metric[1].value,'2')
        at.sidebar.selectbox[0].set_value('square').run()
        at.sidebar.radio[0].set_value('4 · Oracle and Grover').run()
        for r,percent in [(0,'12.500%'),(1,'78.125%'),(2,'94.531%'),(3,'33.008%')]:
            next(s for s in at.slider if s.label=='Complete Grover iterations').set_value(r).run()
            self.assertEqual(at.metric[2].value,percent)
        at.sidebar.radio[0].set_value('5 · Operators and measurements').run()
        self.assertEqual(at.metric[0].value,'1.0000000000')
        next(s for s in at.selectbox if s.label=='Mixer').set_value('x').run()
        self.assertAlmostEqual(at.dataframe[0].value.iloc[1,1],1/8)
        next(s for s in at.selectbox if s.label=='Mixer').set_value('projector').run()
        self.assertAlmostEqual(at.dataframe[0].value.iloc[1,1],121/128)
        # La scelta usa identificatori interni: tradurre le etichette non cambia il costo.
        next(s for s in at.selectbox if s.label=='Cost operator').set_value('cut').run()
        self.assertAlmostEqual(at.dataframe[0].value.iloc[1,1],1/8)
        next(s for s in at.selectbox if s.label=='Cost operator').set_value('predicate').run()
        self.assertAlmostEqual(at.dataframe[0].value.iloc[1,1],121/128)
        self.assertEqual(list(at.dataframe[0].value.columns),
                         ['Observable','Exact expectation','Estimate from the same counts'])
        self.assertFalse(at.exception)


if __name__=="__main__":unittest.main()
