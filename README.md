# Laboratorio QAOA, SAT e Grover

Materiali per una lezione alla lavagna di 150 minuti, rivolta a studenti magistrali che conoscono circuiti quantistici, Python e Grover. MaxCut sul quadrato collega QAOA, decisione a soglia, SAT, oracoli reversibili e operatori alternati. Dispensa, notebook e guida all'app sono in italiano; l'interfaccia Streamlit è interamente in inglese. Formule, convenzioni e risultati di riferimento sono comuni a tutti i materiali.

## Materiali

- `output/pdf/guida_docente.pdf`: dispensa tecnica per una lezione universitaria alla lavagna, con derivazioni complete QAOA, riduzione a SAT, oracolo reversibile e conclusione sui risultati delle misure. Grover è assunto noto.
- `materials/guida_docente.tex` e `materials/sections/`: sorgenti LaTeX modificabili della dispensa.
- `output/pdf/guida_app.pdf`: guida di tre pagine, con orientamento per gli studenti e dimostrazione interattiva di circa 20 minuti per il docente. Riporta le etichette inglesi dei controlli e i risultati attesi a ogni passaggio.
- `materials/guida_app.tex`: sorgente LaTeX della guida all'app.
- `tutorial_qaoa_sat_grover.ipynb`: notebook autonomo, con 25 celle di codice eseguite, esercizi e soluzioni. Non richiede `quantum_lab.py` per funzionare.
- `app.py`: laboratorio Streamlit in inglese, in cinque tappe, con soluzioni degli esercizi visualizzabili a richiesta.
- `quantum_lab.py`: simulazione esatta NumPy, formule CNF e circuiti Qiskit.
- `assets/`: diagrammi Qiskit precalcolati, usati dall'app senza invocare il runtime nativo nei thread Streamlit.

## Avvio locale

Python 3.11 o successivo; ambiente verificato con Python 3.13 e Streamlit 1.67. Dalla cartella del progetto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Per questo checkout l'ambiente `.venv` è già predisposto. Basta eseguire:

```bash
.venv/bin/python -m streamlit run app.py
```

Il server indica l'indirizzo locale da aprire. Dopo l'installazione il laboratorio funziona senza servizi cloud o credenziali. Per chiuderlo, Ctrl+C nel terminale.

## Percorso nell'app

1. Colorare i quattro vertici e confrontare il taglio con l'enumerazione esatta.
2. Modificare gli angoli QAOA, ottimizzare la media, confrontare probabilità esatte e frequenze delle misure.
3. Variare la soglia K, leggere la CNF ed esportarla in DIMACS. La diagonale fornisce un controesempio alla condizione «tagliare tutti gli archi».
4. Collegare la rete reversibile all'oracolo e applicare brevemente Grover all'istanza.
5. Cambiare costo e mixer, verificare l'identità con Grover e stimare due osservabili dagli stessi conteggi.

La soglia selezionata vale per SAT e Grover; QAOA standard ottimizza sempre il costo C(x). La simulazione usa quattro qubit dati. Il circuito reversibile esplicito è fornito per C₄, K=4 e impiega sette ancille pulite. Per le altre soglie la simulazione usa il predicato diagonale esatto, senza una compilazione reversibile generica.

La ricerca numerica degli angoli usa valori attesi esatti, sia nell'app sia nel notebook. Il campionamento con un numero finito di misure è un esperimento separato: permette di confrontare lo stimatore con il suo valore atteso, ma non alimenta l'ottimizzatore. La dispensa distingue questa simulazione dal ciclo variazionale su un dispositivo.

## Corrispondenza fra i materiali

| Argomento | Dispensa | Notebook | App |
|---|---|---|---|
| Motivazione, MaxCut e Hamiltoniano | Sezioni 1–3 | Sezioni 1–2 | Tappe 1–2 |
| Derivazione di F₁ e ampiezze | Sezioni 4–5 | Sezione 2 | Tappa 2, approfondimento sui calcoli |
| Decisione, CNF e cardinalità | Sezione 6 e appendice A | Sezione 3 | Tappa 3 |
| Rete reversibile e inversione del calcolo | Sezione 7 | Sezione 4 | Tappa 4, pannello sull'oracolo |
| Applicazione di Grover | Sezione 8 | Sezione 5 | Tappa 4 |
| Identità degli operatori | Sezione 9 | Sezione 6 | Tappa 5 |
| Risultati delle misure, osservabili e topologia | Sezione 10 | Sezione 7 | Tappa 5 |

La scansione comune comprende cinque minuti di pausa e cinque di richiamo di Grover. I contatori CNF, l'adiabaticità e il calcolo con matrici densità sono approfondimenti. Nel notebook le verifiche al punto di riferimento rimangono distinte dagli esperimenti con angoli modificabili.

## Convenzioni

- Stringhe in ordine `x1 x2 x3 x4`, da sinistra a destra; x1 corrisponde a q3 in Qiskit.
- `H_C = Σ(I-Z_i Z_j)/2`, `U_C = exp(-i γ H_C)`, `H_M = ΣX_i`.
- Porte `RZZ(-γ)` e `RX(2β)`; la fase globale del costo è inclusa.
- Si massimizza il taglio medio; SciPy minimizza il suo negativo.
- C₄ a p=1: β=π/8, γ=π/4 danno media 3 e probabilità ottimo 17/32.
- Grover a r=0,1,2,3: probabilità 1/8, 25/32, 121/128, 169/512.
- L'identità con Grover richiede il proiettore del predicato come costo e il mixer `|s><s|`, con β=γ=π. È un caso degli operatori alternati generalizzati, non del QAOA con mixer `ΣX`.

Il quadrato e la formula 2-SAT sono facili classicamente. Le percentuali non sono un confronto dei costi totali degli algoritmi. La codifica di cardinalità per sottoinsiemi è limitata ai piccoli grafi del laboratorio e non è una riduzione scalabile.

## Verifica e rigenerazione

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/build_notebook.py
python scripts/execute_notebook.py
python scripts/build_circuits.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf materials/guida_docente.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf materials/guida_docente.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf materials/guida_app.tex
```

Il generatore ricrea il notebook senza output; lo script successivo lo esegue e salva i risultati solo se tutte le celle terminano correttamente. Le due compilazioni della dispensa aggiornano indice e riferimenti. I 17 test confrontano NumPy con Qiskit e verificano coniugazioni di Pauli, ampiezze, oracolo su tutti gli input, ancille, CNF, proiettori, confronto fra topologie e interazioni Streamlit. Il percorso descritto nella guida è verificato anche attraverso i controlli inglesi dell'app; le etichette tradotte sono separate dagli identificatori che selezionano grafi e operatori. `requirements-lock.txt` registra le versioni dell'ambiente di verifica.

I disegni dell'app sono simbolici: gli angoli selezionati alimentano la simulazione NumPy. La generazione dei diagrammi avviene separatamente per evitare un arresto nativo di Qiskit 2.6.2 osservato su macOS durante la ricreazione dei circuiti nei thread di Streamlit. I circuiti eseguibili restano disponibili nel notebook.

## Pubblicazione Streamlit

Per Streamlit Community Cloud usare `app.py` come file di avvio e `requirements.txt` come dipendenze, mantenendo la struttura del progetto. Includere `quantum_lab.py`, `assets/`, `.streamlit/`, il notebook e i due PDF in `output/pdf/`: questi ultimi sono scaricabili dall'app e indicati esplicitamente come materiali in italiano. Il progetto è preparato localmente; nessuna pubblicazione o modifica del repository remoto è inclusa.

L'app di riferimento è [leonardoLavagna/qaoa](https://github.com/leonardoLavagna/qaoa), collegata a [quantum-optimization-app.streamlit.app](https://quantum-optimization-app.streamlit.app/). Questa implementazione è un laboratorio autonomo incentrato sulla lezione; il vecchio codice non è stato copiato. Il materiale di partenza è `tutorial_3_2.ipynb` e `notes/notes.tex` fornito dal docente.
