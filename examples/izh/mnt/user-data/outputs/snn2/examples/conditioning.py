"""
Instrumental (GO/NO-GO) conditioning, and the proof that batched grouping is
correct (independent, position-invariant, learns different contingencies in one
batch).

    python examples/conditioning.py
"""
from snn2 import conditioning

conditioning.validate()                              # 4 grouping/learning checks
print("wrote", conditioning.make_figure("grouping_conditioning.png"))
