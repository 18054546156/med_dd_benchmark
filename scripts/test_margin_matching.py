"""Exercise both controls through the original model and factor-two decoder."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapted" / "ncfm"))
from models.convnet import ConvNet
from condenser.decode import decode_zoom
from NCFM.margin_matching import margin_loss

torch.manual_seed(4)
model = ConvNet(9).eval().requires_grad_(False)
canvas = torch.rand(10,3,32,32,requires_grad=True)
decoded, _ = decode_zoom((32,32),canvas,torch.zeros(10,dtype=torch.long),2)
targets = {"rivals":torch.tensor([[[1,2]]*9]),"pair":torch.randn(1,9,3,2),
           "overall":torch.randn(1,9,3)}
for mode in ("overall", "rival"):
    loss = margin_loss(model,decoded,0,0,targets,mode)
    grad, = torch.autograd.grad(loss,canvas,retain_graph=True)
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.abs().sum() > 0
    print(mode, "loss", float(loss), "canvas_gradient", float(grad.norm()))
assert all(p.grad is None for p in model.parameters())
