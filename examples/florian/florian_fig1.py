"""
Reproduce Florian (2007) Figure 1 (MSTDPET) and validate it against the
paper's reference equations.

    python examples/florian_fig1.py
"""
from snn2 import florian

# 1) validate: module curve must equal an independent transcription of the
#    paper equations, plus the qualitative facts visible in the figure.
florian.validate()

# 2) render the 8-panel figure.
path = florian.make_figure("florian_fig1.png")
print("wrote", path)
