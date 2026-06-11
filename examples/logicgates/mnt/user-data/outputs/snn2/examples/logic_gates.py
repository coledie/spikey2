"""
Learning logic gates with reward-modulated STDP, culminating in dendritic XOR
(Gidon et al. 2020). Prints the curriculum result table + validation checks and
writes the figure.

Run:  python examples/logic_gates.py
"""
from snn2.logic_gates import validate, make_figure


def main():
    # Full curriculum + assertions: starts-not-knowing, OR learned, AND ceiling,
    # monotonic-XOR wall, dendritic-XOR solved (~1.0), exact [0,1,1,0] truth table.
    validate(n_seeds=8)

    # 4-panel figure: learning curves, learned truth tables, the dendritic XOR
    # firing window, and the point-vs-dendrite contrast on XOR.
    path = make_figure("logic_gates.png", n_seeds=8)
    print("\nfigure written to", path)


if __name__ == "__main__":
    main()
