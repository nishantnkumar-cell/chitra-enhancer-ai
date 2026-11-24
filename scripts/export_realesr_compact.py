"""Export Real-ESRGAN "compact" weights (SRVGGNetCompact, BSD-3-Clause) to ONNX.

    uv run --with torch --with onnx --with onnxconverter-common --with onnxscript \
        scripts/export_realesr_compact.py realesr-general-x4v3.pth out_dir

Writes <name>_fp32.onnx and <name>_fp16.onnx. Input: RGB 0..1, NCHW, any size
that is a multiple of 1; output is 4x larger. Weights are loaded with
`weights_only=True`, so nothing in the file can run code.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, nf=64, nconv=32, up=4):
        super().__init__()
        self.up = up
        layers = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        for _ in range(nconv):
            layers += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        layers += [nn.Conv2d(nf, 3 * up * up, 3, 1, 1)]
        self.body = nn.ModuleList(layers)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = F.pixel_shuffle(out, self.up)
        return out + F.interpolate(x, scale_factor=self.up, mode="nearest")


def main(weights: str, outdir: str) -> None:
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    nconv = (len([k for k in sd if k.endswith(".weight") and k.startswith("body.")]) - 2) // 2
    net = SRVGGNetCompact(nconv=nconv)
    net.load_state_dict(sd)
    net.eval()
    name = Path(weights).stem
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    f32 = out / f"{name}_fp32.onnx"
    torch.onnx.export(
        net, torch.zeros(1, 3, 64, 64), str(f32),
        input_names=["input"], output_names=["output"], opset_version=17,
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h2", 3: "w2"}},
        dynamo=False,
    )
    import onnx
    from onnxconverter_common import float16

    m = float16.convert_float_to_float16(onnx.load(str(f32)), keep_io_types=False)
    onnx.save(m, str(out / f"{name}_fp16.onnx"))
    print("wrote", f32, "and the fp16 twin")


if __name__ == "__main__":
    main(*sys.argv[1:3])
