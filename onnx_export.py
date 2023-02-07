from model import get_model

import torch

ssdlite = get_model("cuda")
model = ssdlite.eval()
dummy_input = torch.rand((1, 3, 320, 320), device="cuda")

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
    opset_version=11
)