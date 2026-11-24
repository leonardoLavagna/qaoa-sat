"""Run with: streamlit run app.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from quantum_lab import (
    SQUARE, DIAGONAL, assignments, labels, costs, uniform, qaoa,
    alternating, expectation, grover_trace, grover_probability, sample,
    all_edges_cnf, threshold_cnf, satisfies, dimacs, optimize_qaoa,
)

ROOT=Path(__file__).parent

st.set_page_config(page_title="QAOA · SAT · Grover", page_icon="◈", layout="wide")
st.html("""<style>
.block-container {max-width:1180px;padding-top:2.4rem;padding-bottom:3rem}
h1 {letter-spacing:-.045em;font-weight:750!important}
h2 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:#EEF2F3;padding:16px;border-radius:12px}
[data-testid="stSidebar"] {border-right:1px solid #dce4e5}
.eyebrow {font-size:.78rem;letter-spacing:.16em;color:#176B87;font-weight:700}
.lead {font-size:1.1rem;color:#526371;max-width:780px;line-height:1.65}
</style>""")
STEPS = ["1 · MaxCut", "2 · QAOA", "3 · From MaxCut to SAT", "4 · Oracle and Grover", "5 · Operators and measurements"]
GRAPH_LABELS = {"square": "Square C₄", "diagonal": "Square + diagonal"}
with st.sidebar:
    st.markdown("## Quantum optimization")
    st.caption("MaxCut on a square · QAOA, SAT and Grover")
    page = st.radio("Explore", STEPS)
    st.divider()
    graph = st.selectbox("Graph", list(GRAPH_LABELS), format_func=GRAPH_LABELS.get)
    edges = SQUARE if graph == "square" else DIAGONAL
    threshold = st.slider("Threshold K: C(x) ≥ K", 0, len(edges)+1, 4)
    st.caption("K defines the SAT instance and the oracle. Standard QAOA always optimizes the full cut objective C(x).")
    teacher = st.toggle("Show exercise solutions", value=False)
    st.divider()
    st.caption("Ideal simulation on 4 data qubits. Each shot repeats state preparation and measurement. Sampling includes finite-shot fluctuations, but no hardware noise.")

bits = assignments()
names = labels()
c = costs(edges)
marked = c >= threshold
optimal = c == c.max()


def heading(k, title, intro):
    st.html(f'<div class="eyebrow">QAOA, SAT AND GROVER · {k} / 5</div>')
    st.title(title)
    st.markdown(f'<div class="lead">{intro}</div>', unsafe_allow_html=True)
    st.write("")


def exercise(question, answer):
    st.info("Exercise. " + question)
    if teacher:
        with st.expander("Discussion and solution", expanded=True):
            st.markdown(answer)


def style(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=15, r=15, t=30, b=25),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#172B3A"), legend=dict(orientation="h", y=1.16))
    return fig


def graph_plot(x):
    pos = [(0,1), (1,1), (1,0), (0,0)]
    fig = go.Figure()
    for i,j in edges:
        cut = x[i] != x[j]
        fig.add_trace(go.Scatter(x=[pos[i][0],pos[j][0]],y=[pos[i][1],pos[j][1]],
            mode="lines",line=dict(color="#D57936" if cut else "#A9B7BE",width=6 if cut else 3),
            showlegend=False,hoverinfo="text",text=f"Edge {i+1}–{j+1}: {'cut' if cut else 'not cut'}"))
    fig.add_trace(go.Scatter(x=[p[0] for p in pos], y=[p[1] for p in pos],mode="markers+text",
        marker=dict(size=47,color=["#176B87" if b==0 else "#D57936" for b in x],line=dict(width=3,color="white")),
        text=[f"x{i+1}={b}" for i,b in enumerate(x)],textposition="top center",showlegend=False,
        hoverinfo="text"))
    fig.update_xaxes(visible=False,range=[-.4,1.4])
    fig.update_yaxes(visible=False,range=[-.3,1.4],scaleanchor="x")
    return style(fig,330)


def distribution(state, sampled=None, highlight=None, highlight_label="optimal assignments for the selected graph"):
    p = np.abs(state)**2
    highlight = optimal if highlight is None else highlight
    fig = go.Figure(go.Bar(x=names,y=p,name="Exact probability",
        marker_color=["#D57936" if o else "#176B87" for o in highlight],
        hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    if sampled is not None:
        fig.add_trace(go.Scatter(x=names,y=sampled/sampled.sum(),name="Observed frequency",
            mode="markers",marker=dict(symbol="diamond",color="#172B3A",size=8)))
    fig.update_layout(yaxis_title="Probability",xaxis_title="Bit string x₁x₂x₃x₄",bargap=.25)
    fig.update_xaxes(type="category",categoryorder="array",categoryarray=names,
                     tickmode="array",tickvals=names,ticktext=names,tickangle=-45)
    ymax = max(float(p.max()),float((sampled/sampled.sum()).max()) if sampled is not None else 0.)
    fig.update_yaxes(range=[0,min(1.05,max(.3,ymax*1.2))])
    st.plotly_chart(style(fig),width="stretch")
    st.caption(f"Orange: {highlight_label}. Bit strings are ordered as x₁x₂x₃x₄.")


def metrics(state):
    a,b,d = st.columns(3)
    p = np.abs(state)**2
    a.metric("Expected cut value ⟨C⟩", f"{expectation(state,c):.4f}")
    b.metric("Probability of an optimal cut", f"{100*p[optimal].sum():.3f}%")
    d.metric(f"Success probability: C ≥ {threshold}",f"{100*p[marked].sum():.3f}%")


def cnf_text(clauses, ndata=4):
    def literal(lit):
        v = abs(lit)
        name = f"x{v}" if v <= ndata else f"y{v-ndata}"
        return ("¬" if lit < 0 else "") + name
    return " ∧\n".join("(" + " ∨ ".join(literal(lit) for lit in clause) + ")" if clause else "⊥" for clause in clauses) or "⊤"


if page == STEPS[0]:
    heading(1,"MaxCut on a square","MaxCut asks us to divide the vertices of a graph into two groups so that as many edges as possible connect different groups. One bit per vertex specifies a candidate assignment. We will use this small example to follow the same problem through several quantum and classical representations.")
    left,right = st.columns([1,1.1],gap="large")
    with left:
        chosen = st.select_slider("Assignment x₁x₂x₃x₄",options=names,value="0101")
        x = np.array(list(map(int,chosen)))
        st.plotly_chart(graph_plot(x),width="stretch")
    with right:
        st.latex(r"C(x)=\sum_{(i,j)\in E}(x_i\oplus x_j)")
        a,b = st.columns(2)
        a.metric("Cut edges",int(c[int(chosen,2)]))
        b.metric("Optimum by enumeration",int(c.max()))
        rows=[{"Edge":f"{i+1}–{j+1}","XOR":int(x[i]^x[j])} for i,j in edges]
        st.dataframe(rows,hide_index=True,width="stretch")
    exercise("On the square with K=4, compare 0101 and 1010. Do they represent different cuts? How many solutions does Grover count?",
             "They represent the same unordered partition, but they are distinct computational-basis states. Grover therefore has M=2 marked states out of N=16.")
    with st.expander("All assignments and the threshold predicate"):
        table=pd.DataFrame({"x":names,"C(x)":c,f"C(x) ≥ {threshold}":marked})
        st.dataframe(table,hide_index=True,width="stretch")
        st.download_button("Download assignment table",table.to_csv(index=False),"maxcut.csv","text/csv")
    st.markdown("A cycle has an even number of cut edges: each cut edge switches groups, and a full traversal returns to the starting group. On the square, the possible cut values are 0, 2 and 4, attained by 2, 12 and 2 assignments respectively.")
    st.caption("General MaxCut is NP-hard, but C₄ is bipartite and easy to solve classically. This example makes each construction explicit; it does not demonstrate a quantum advantage.")

elif page == STEPS[1]:
    heading(2,"QAOA: expectation values and measurements","QAOA prepares a family of states by varying the angles of two alternating operators. The cost operator assigns phases according to the cut value; the mixer allows amplitudes to interfere. A classical optimizer searches for angles that improve the expected cut value.")
    st.markdown(r"Using $x_i\oplus x_j=x_i+x_j-2x_ix_j$ and $\widehat x_i=(I-Z_i)/2$ gives the cost Hamiltonian. We choose the mixer separately: it connects bit strings that differ in one bit.")
    st.latex(r"H_C=\sum_{(i,j)\in E}\frac{I-Z_iZ_j}{2},\qquad H_M=\sum_i X_i")
    st.latex(r"U_C(\gamma)=e^{-i\gamma H_C},\qquad U_M(\beta)=e^{-i\beta H_M}")
    st.latex(r"|\psi_p\rangle=U_M(\beta_p)U_C(\gamma_p)\cdots U_M(\beta_1)U_C(\gamma_1)|+\rangle^{\otimes4}")
    st.latex(r"F_{G,p}(\boldsymbol\beta,\boldsymbol\gamma)=\langle\psi_p|H_C|\psi_p\rangle")
    p = st.select_slider("Layers p",[1,2,3],value=1)
    def preset():
        for j in range(p):
            st.session_state[f"beta{j}"] = .125 if j==0 else 0.
            st.session_state[f"gamma{j}"] = .25 if j==0 else 0.
        st.session_state.pop("optimization",None)
    a,b = st.columns(2)
    a.button("Set reference angles",on_click=preset,width="stretch")
    if b.button("Optimize exact expectation · 4 starts",type="primary",width="stretch"):
        with st.spinner("Optimizing the circuit angles…"):
            result,histories=optimize_qaoa(p,edges)
        for j in range(p):
            st.session_state[f"beta{j}"] = float((result.x[j]/np.pi+1)%2-1)
            st.session_state[f"gamma{j}"] = float((result.x[p+j]/np.pi+1)%2-1)
        st.session_state["optimization"]=(graph,p,histories,float(-result.fun))
    st.caption("The reference sets β₁=π/8 and γ₁=π/4, with any later layers set to zero. The optimizer uses exact simulator expectations. Shot-based estimates below illustrate sampling separately; they do not drive this optimization.")
    betas,gammas=[],[]
    for j in range(p):
        a,b=st.columns(2)
        bk,gk=f"beta{j}",f"gamma{j}"
        if bk not in st.session_state: st.session_state[bk]=.125 if j==0 else 0.
        if gk not in st.session_state: st.session_state[gk]=.25 if j==0 else 0.
        betas.append(a.slider(f"β{j+1} / π",-1.,1.,step=.005,key=bk,format="%.3f")*np.pi)
        gammas.append(b.slider(f"γ{j+1} / π",-1.,1.,step=.005,key=gk,format="%.3f")*np.pi)
    state=qaoa(betas,gammas,edges)
    metrics(state)
    a,b=st.columns([2,1])
    with b:
        shots=st.select_slider("Shots S",[64,256,1024,4096,16384],value=1024)
        seed=st.number_input("Sampling seed",0,100000,7)
        counts=sample(state,shots,seed)
        estimate=float(counts@c/shots)
        st.metric("Estimated cut value",f"{estimate:.4f}")
        variance=float(np.abs(state)**2 @ ((c-expectation(state,c))**2))
        st.caption(f"Theoretical standard error: {np.sqrt(variance/shots):.4f}. Change the seed to generate another reproducible sample.")
    with a: distribution(state,counts)
    with st.expander("From measurement counts to the estimated cut value"):
        st.latex(r"\widehat F_{G,p}=\frac1S\sum_x n_xC_G(x),\qquad\operatorname{Var}(\widehat F_{G,p})=\frac{\operatorname{Var}(C_G)}S")
        st.markdown("Each measurement returns a bit string. These counts estimate F at the current angles; they do not reconstruct the entire function. Every string provides the contribution of every edge, since all cost terms are diagonal in the computational basis.")
        st.dataframe(pd.DataFrame({"x":names,"n_x":counts,"C(x)":c,"n_x C(x) / S":counts*c/shots}),hide_index=True,width="stretch")
    if graph=="square" and p==1:
        st.latex(r"F_1(\beta,\gamma)=2+\sin(4\beta)\sin(2\gamma)\leq3")
        st.caption("The analytical maximum is 3, attained at β=π/8 and γ=π/4. Here we can choose the angles without numerical optimization. An individual measurement still has cut value 0, 2 or 4.")
        with st.expander("Deriving the expectation and amplitudes"):
            st.markdown("Symmetry reduces the calculation to one edge: F₁=2−2⟨Z₁Z₂⟩. Conjugating the observable by the mixer gives ZZ, ZY, YZ and YY terms. After conjugation by the cost unitary, only the mixed terms contribute to the expectation in the uniform state. Sections 4 and 5 of the lecture notes (in Italian) give the full derivation.")
            st.latex(r"\langle Z_1Z_2\rangle=-\frac12\sin(4\beta)\sin(2\gamma)")
            st.markdown("At β=π/8 and γ=π/4, the four symmetry classes have the amplitudes below. This reference table does not change with the sliders.")
            st.table(pd.DataFrame({"Representative":["0000","0001","0011","0101"],"Multiplicity":[2,8,4,2],"Cut value":[0,2,2,4],"Amplitude":["i/8","−1/8","−1/4−i/8","−1/2+i/8"],"Probability per string":["1/64","1/64","5/64","17/64"]}))
            st.markdown("An expectation of 3 does not by itself fix the probability of an optimum: in general F=2+2(P₄−P₀). The value 17/32 follows from this circuit's distribution at the reference angles.")
        with st.expander("Explore the parameter landscape"):
            bs=np.linspace(0,np.pi/2,65); gs=np.linspace(0,np.pi,65)
            z=2+np.sin(4*bs[:,None])*np.sin(2*gs[None,:])
            fig=go.Figure(go.Heatmap(x=gs/np.pi,y=bs/np.pi,z=z,colorscale="Teal",colorbar_title="⟨C⟩"))
            fig.update_layout(xaxis_title="γ / π",yaxis_title="β / π")
            st.plotly_chart(style(fig),width="stretch")
    record=st.session_state.get("optimization")
    if record and record[:2]==(graph,p):
        with st.expander("Latest optimization · objective evaluations"):
            fig=go.Figure()
            for j,h in enumerate(record[2]):fig.add_trace(go.Scatter(y=h,name=f"Start {j+1}"))
            fig.update_layout(xaxis_title="Objective evaluation",yaxis_title="⟨C⟩")
            st.plotly_chart(style(fig),width="stretch")
            st.caption("These curves include all objective evaluations, including those used to estimate gradients. The numerical search does not certify a global optimum.")
    with st.expander("Circuit and Qiskit conventions"):
        circuit_name=f"qaoa_{'square' if edges==SQUARE else 'diagonal'}_p{p}.txt"
        st.code((ROOT/"assets"/circuit_name).read_text(),language=None)
        st.markdown("We use `RZZ(-γ)` for each edge and `RX(2β)` for each qubit. The circuit includes the global phase from the constant cost term. We map x₁ to q₃, so Qiskit's printed strings read x₁x₂x₃x₄. The diagram uses symbolic angles; the sliders set their numerical values in the simulation.")
    exercise("On the square, choose one layer and the reference angles π/8 and π/4. Why does ⟨C⟩=3 not mean that you will measure a cut of value 3?",
             "C can only be 0, 2 or 4. At these angles P(C=0)=1/32, P(C=2)=7/16 and P(C=4)=17/32. The expectation is 3, while the probability of an optimal cut is 53.125%.")

elif page == STEPS[2]:
    heading(3,"From the MaxCut decision problem to SAT","Fix a threshold K and ask whether a cut of value at least K exists. We can express this question as a Boolean formula. A satisfying assignment will provide a candidate that meets the threshold.")
    st.latex(r"f_K(x)=\mathbf{1}[C(x)\geq K]")
    st.markdown("SAT asks whether an assignment makes a Boolean formula true. A formula in conjunctive normal form (CNF) is an AND of clauses; each clause is an OR of variables or their negations. If we introduce auxiliary variables a, the encoding must satisfy the following equivalence for every x:")
    st.latex(r"\exists a:\Phi_{G,K}(x,a)=1\quad\Longleftrightarrow\quad C_G(x)\geq K")
    a,b,d=st.columns(3)
    a.metric("Threshold K",threshold);b.metric("Satisfying assignments",int(marked.sum()));d.metric("Result by enumeration","SAT" if marked.any() else "UNSAT")
    if threshold==len(edges):
        clauses=all_edges_cnf(edges);nvars=4
        st.markdown("When **K equals the number of edges**, every edge must be cut:")
        st.latex(r"x_i\oplus x_j=1\iff(x_i\lor x_j)\land(\neg x_i\lor\neg x_j)")
        st.code(cnf_text(clauses),language=None)
        st.caption("This is a 2-SAT formula: it tests whether the graph is bipartite. It is a special case of the MaxCut decision problem.")
    else:
        clauses,nvars=threshold_cnf(edges,threshold)
        st.markdown("For a general threshold, introduce one variable yₑ for each edge:")
        st.latex(r"y_e\leftrightarrow(x_i\oplus x_j),\qquad\sum_e y_e\geq K")
        with st.expander("Full CNF with auxiliary variables"):
            st.code(cnf_text(clauses),language=None)
            st.caption("For these small graphs we encode the cardinality constraint using subsets of edges. This encoding can grow exponentially; scalable alternatives include sequential counters. The y values are computed from x, so our Grover search uses only the 4 data bits.")
    st.download_button("Download DIMACS formula",dimacs(clauses,nvars),"maxcut.cnf","text/plain")
    st.dataframe(pd.DataFrame({"x":names,"C(x)":c,"f_K(x)":marked.astype(int)}),hide_index=True,width="stretch")
    exercise("Add the diagonal and compare K=5 with K=4. If the formula requiring every edge to be cut is UNSAT, does MaxCut have no solution?",
             "No. The diagonal creates triangles, so K=5 is impossible. The optimum is 4, attained by 0101 and 1010. MaxCut still has an optimal solution; only the threshold 5 is unattainable.")
    with st.expander("Decision, optimization and Max-2-SAT"):
        st.markdown("Answering a single threshold query does not generally determine the maximum. For an unweighted graph with m edges, an exact decision solver can find the optimal value by binary search over K from 0 to m. On the square, K=4 is already an attainable upper bound.")
        st.markdown("The two-clause-per-edge formula also gives an optimization problem. A cut edge satisfies two clauses; an uncut edge satisfies one. The total is m+C(x). Maximizing this count is Max-2-SAT, distinct from 2-SAT, which asks whether all clauses can be satisfied.")

elif page == STEPS[3]:
    heading(4,"The oracle and Grover search","The threshold predicate becomes a phase oracle. Assuming familiarity with Grover's algorithm, we apply it to this instance and focus on the reversible construction and the interpretation of its output.")
    st.latex(r"O_f|x\rangle=(-1)^{f_K(x)}|x\rangle")
    st.latex(r"D=2|s\rangle\langle s|-I,\qquad G=DO_f")
    st.caption("Here the SAT table gives M by enumeration. In a general search M may be unknown, requiring a suitable strategy for choosing the iteration count. One unsuccessful measurement does not certify UNSAT.")
    if not marked.any():st.warning("No assignment meets this threshold. The simulated success probability is zero at every iteration.")
    if marked.all():st.info("Every assignment meets this threshold. Success is certain even before applying Grover.")
    iterations=st.slider("Complete Grover iterations",0,8,2)
    trace=grover_trace(marked,iterations)
    stage=st.select_slider("Inspect a step",options=list(range(len(trace))),value=len(trace)-1,
        format_func=lambda i:"Initial state" if i==0 else f"{'Oracle' if i%2 else 'Diffusion'} {(i+1)//2}")
    state=trace[stage][1]
    metrics(state)
    a,b=st.columns(2)
    with a:
        fig=go.Figure(go.Bar(x=names,y=state.real,marker_color=["#D57936" if m else "#176B87" for m in marked]))
        fig.update_layout(yaxis_title="Real amplitude (including sign)",xaxis_title="Bit string x₁x₂x₃x₄")
        fig.update_xaxes(type="category",categoryorder="array",categoryarray=names,
                         tickmode="array",tickvals=names,ticktext=names,tickangle=-45)
        st.plotly_chart(style(fig),width="stretch")
    with b:distribution(state,highlight=marked,highlight_label=f"assignments satisfying C(x) ≥ {threshold}")
    st.latex(r"P_r=\sin^2((2r+1)\theta),\qquad\sin^2\theta=M/N")
    probabilities=[grover_probability(int(marked.sum()),16,r) for r in range(9)]
    fig=go.Figure(go.Scatter(x=list(range(9)),y=probabilities,mode="lines+markers",line_color="#176B87"))
    fig.update_layout(xaxis_title="Complete iterations r",yaxis_title="P(C ≥ K)")
    fig.update_yaxes(range=[0,1.05]);st.plotly_chart(style(fig,250),width="stretch")
    with st.expander("Inside the oracle: compute → phase → uncompute"):
        st.markdown("For C₄ with K=4, compute four edge XORs into four ancillas, then two intermediate ANDs and a final AND. Apply Z to the result and reverse the computation. This circuit uses **4 data qubits and 7 ancillas**, with 16 CNOTs, 6 Toffolis and one Z, including uncomputation. It is not minimal; its purpose is to derive the oracle from the edges without using a list of solutions.")
        st.code("e12 = x1 XOR x2; e23 = x2 XOR x3\ne34 = x3 XOR x4; e41 = x4 XOR x1\nt1 = e12 AND e23; t2 = e34 AND e41\nq = t1 AND t2\nZ(q)\nuncompute(q, t2, t1, e41, e34, e23, e12)",language=None)
        st.latex(r"\begin{aligned}|x\rangle|0^7\rangle&\xrightarrow{W}|x\rangle|g(x)\rangle|f_4(x)\rangle\\&\xrightarrow{Z_q}(-1)^{f_4(x)}|x\rangle|g(x)\rangle|f_4(x)\rangle\\&\xrightarrow{W^\dagger}(-1)^{f_4(x)}|x\rangle|0^7\rangle\end{aligned}")
        st.markdown("Uncomputation restores the ancillas to zero for every input while preserving the acquired phase. Input-dependent information left in the work registers would change interference on the data register. The diffusion step assumes these registers have been reset coherently.")
        if graph=="square" and threshold==4:
            st.code((ROOT/"assets"/"square_oracle.txt").read_text(),language=None)
        else:
            st.caption("The construction above is specifically for C₄, K=4. For the current selection, the simulator applies the exact diagonal predicate. A general reversible implementation would also require counting cut edges and comparing with K.")
    exercise("On the square with K=4, compare r=0,1,2,3. Why does the third iteration reduce the success probability?",
             "The probabilities are 1/8, 25/32, 121/128 and 169/512. Grover performs a rotation: after two iterations the state is close to the marked subspace; the third overshoots the first peak. The oracle alone does not change computational-basis probabilities.")

else:
    heading(5,"Alternating operators and measurement outcomes","Return to the alternating structure of QAOA. To recover Grover, choose the solution projector as the cost operator and the uniform-state projector as the mixer. We can then separate the circuit identity from the classical use of measurement data.")
    st.markdown(r"For a projector, $P^2=P$, so $e^{-i\alpha P}=I+(e^{-i\alpha}-1)P$. Setting the angle to π gives a reflection:")
    st.latex(r"P_f=\sum_{x:f_K(x)=1}|x\rangle\langle x|,\quad P_s=|s\rangle\langle s|")
    st.latex(r"e^{-i\pi P_f}=O_f,\qquad e^{-i\pi P_s}=-D")
    st.latex(r"(e^{-i\pi P_s}e^{-i\pi P_f})^p=(-1)^pG^p")
    a,b=st.columns(2)
    cost_choice=a.selectbox("Cost operator",["predicate","cut"],format_func={"predicate":"Predicate projector P_f","cut":"Cut Hamiltonian H_C"}.get)
    mixer_choice=b.selectbox("Mixer",["projector","x"],format_func={"projector":"Uniform-state projector P_s","x":"Standard ΣX"}.get)
    a,b,d=st.columns(3)
    beta=a.slider("β / π · shared by all layers",0.,2.,1.,.05)
    gamma=b.slider("γ / π · shared by all layers",0.,2.,1.,.05)
    p=d.slider("Layers p / reference Grover iterations",1,6,2)
    state=alternating(marked.astype(float) if cost_choice=="predicate" else c,
        [np.pi*beta]*p,[np.pi*gamma]*p,mixer_choice)
    target=grover_trace(marked,p)[-1][1]
    fidelity=float(abs(np.vdot(target,state))**2)
    overlap=np.vdot(target,state)
    phase=overlap/abs(overlap) if abs(overlap)>1e-14 else 1
    distance=float(np.linalg.norm(state-phase*target))
    a,b=st.columns(2)
    a.metric("State fidelity with Grover",f"{fidelity:.10f}")
    b.metric("Distance after global-phase alignment",f"{distance:.2e}")
    metrics(state)
    distribution(state,highlight=marked,highlight_label=f"assignments satisfying C(x) ≥ {threshold}")
    st.markdown("The identity above requires **both projector operators and β=γ=π**. Fidelity compares the prepared states; the algebraic identity is stronger and holds between operators, up to a global phase. It does not identify Grover with standard MaxCut QAOA using the ΣX mixer.")
    exercise("Replace the projector mixer with ΣX while keeping β=π. What happens?",
             "For four qubits exp(−iπΣX)=I. The standard mixer does not perform diffusion, so these angles do not amplify probabilities. Recovering Grover requires the specified generalized family of alternating operators.")
    st.subheader("One sample, two observables")
    st.markdown("Both algorithms return bit strings when measured. The same counts can estimate the expected cut value or the acceptance probability: assign C(x) or fₖ(x) to each string. The sample below comes from the circuit currently selected on this page.")
    st.latex(r"\widehat F=\frac1S\sum_x n_xC_G(x),\qquad\widehat P_{\rm succ}=\frac1S\sum_x n_xf_{G,K}(x)")
    a,b=st.columns(2)
    transcript_shots=a.select_slider("Comparison shots",[64,256,1024,4096,16384],value=1024,key="transcript_shots")
    transcript_seed=b.number_input("Comparison seed",0,100000,11,key="transcript_seed")
    observed=sample(state,transcript_shots,transcript_seed)
    comparison=pd.DataFrame({"Observable":["Cut value C(x)",f"Predicate f_K(x), K={threshold}"],
        "Exact expectation":[expectation(state,c),expectation(state,marked)],
        "Estimate from the same counts":[float(observed@c/transcript_shots),float(observed@marked/transcript_shots)]})
    st.dataframe(comparison,hide_index=True,width="stretch")
    with st.expander("Measurement counts and the two estimates"):
        transcript_table=pd.DataFrame({"x":names,"n_x":observed,"C(x)":c,"f_K(x)":marked.astype(int),
            "n_x C(x) / S":observed*c/transcript_shots,"n_x f_K(x) / S":observed*marked/transcript_shots})
        st.dataframe(transcript_table,hide_index=True,width="stretch")
        st.download_button("Download counts and observables",transcript_table.to_csv(index=False),"transcript.csv","text/csv")
        st.caption("Counts aggregate the measurement record and are sufficient for both averages. Change the seed to observe sampling fluctuations.")
    st.markdown("In a variational QAOA loop, estimates of F guide the next choice of angles; the final circuit is then sampled for candidates. Standard Grover uses fixed reflection phases and verifies the measured candidate. F is a theoretical expectation: a sample estimates its value at one parameter setting, not the full function of the angles.")
    st.latex(r"F_p^{(f,s)}(\vartheta)=\langle\psi_p^{(f,s)}(\vartheta)|P_f|\psi_p^{(f,s)}(\vartheta)\rangle")
    st.latex(r"F_p^{(f,s)}(\pi\mathbf1,\pi\mathbf1)=\sin^2\!\left((2p+1)\arcsin\sqrt{M/N}\right)")
    st.markdown("The family containing Grover therefore also has an objective function. Since P_f is the solution projector, this objective is a success probability at any angles. At angles π it takes the Grover form above, depending only on N, M and p: ideal Grover search from the uniform state treats all marked strings alike, and all unmarked strings alike. The graph and threshold still determine which strings are solutions. Measurement does not remove this dependence.")
    with st.expander("Two graphs with the same predicate at K=4"):
        cs,cd=costs(SQUARE),costs(DIAGONAL)
        qs=qaoa([np.pi/8],[np.pi/4],SQUARE)
        qd=qaoa([np.pi/8],[np.pi/4],DIAGONAL)
        gs=grover_trace(cs>=4,2)[-1][1]
        gd=grover_trace(cd>=4,2)[-1][1]
        st.markdown("This comparison fixes K=4, QAOA at p=1 with β=π/8 and γ=π/4, and Grover at r=2, independently of the controls above. After adding edge (1,3), assignments with x₁≠x₃ have cut value 3; all other cut values stay unchanged. Only 0101 and 1010 meet the threshold.")
        st.table(pd.DataFrame({"Graph":["Square","Square + diagonal"],
            "QAOA P(optimum)":[expectation(qs,cs==cs.max()),expectation(qd,cd==cd.max())],
            "Grover P(success)":[expectation(gs,cs>=4),expectation(gd,cd>=4)],
            "Expected cut after Grover":[expectation(gs,cs),expectation(gd,cd)]}))
        st.markdown("The predicates are identical, so the ideal Grover distributions are identical. Standard QAOA distinguishes the full cut objectives. Even the same Grover state has different expected cut values when we change the observable C. Having only the same number M of solutions would guarantee equal total success probabilities, but not necessarily equal probabilities for each labelled string.")
    with st.expander("Reference comparison · square, K=4"):
        st.table(pd.DataFrame({"Method":["Uniform state","QAOA p=1, β=π/8 γ=π/4","Grover r=1","Grover r=2"],
            "Probability of an optimum":["12.500%","53.125%","78.125%","94.53125%"]}))
        st.caption("These probabilities are not a resource benchmark. Grover uses a threshold predicate and an oracle that must be implemented; standard QAOA uses the full cut objective. Oracle-query complexity is not the total gate count.")
    st.markdown("References: [QAOA, Farhi et al.](https://arxiv.org/abs/1411.4028) · [Alternating operators, Hadfield et al.](https://arxiv.org/abs/1709.03489) · [Grover](https://arxiv.org/abs/quant-ph/9605043)")

with st.sidebar:
    root=Path(__file__).parent
    st.caption("Companion materials are in Italian.")
    for filename,label in [("output/pdf/guida_docente.pdf","Lecture notes (Italian)"),("tutorial_qaoa_sat_grover.ipynb","Tutorial notebook (Italian)"),("output/pdf/guida_app.pdf","App walkthrough (Italian)")]:
        file=root/filename
        if file.exists():st.download_button(label,file.read_bytes(),file.name)
