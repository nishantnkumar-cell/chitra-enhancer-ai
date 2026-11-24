"""Export FastDVDnet (multi-frame video denoiser, MIT licence) to ONNX.

    uv run --with torch --with onnx --with onnxconverter-common --with onnxscript \
        scripts/export_fastdvdnet.py model.pth out_dir

Weights: https://github.com/m-tassano/fastdvdnet (model.pth).
Network definition below follows that repository (MIT, (c) Matias Tassano).
Inputs : frames  [1, 15, H, W]  five RGB frames (t-2..t+2) stacked, 0..1
         sigma   [1, 1, H, W]   noise level map, sigma/255 (e.g. 0.04 ~ sigma 10)
Output : [1, 3, H, W]           the denoised middle frame. H and W: multiples of 4.
Weights are loaded with weights_only=True so the file cannot run code.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn


class cv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.convblock(x)


class InputCv(nn.Module):
    def __init__(self, n, o):
        super().__init__()
        m = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(n * 4, n * m, 3, padding=1, groups=n, bias=False),
            nn.BatchNorm2d(n * m), nn.ReLU(True),
            nn.Conv2d(n * m, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.convblock(x)


class Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(True), cv(o, o))

    def forward(self, x):
        return self.convblock(x)


class Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.convblock = nn.Sequential(
            cv(i, i), nn.Conv2d(i, o * 4, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.convblock(x)


class Out(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(i, i, 3, padding=1, bias=False), nn.BatchNorm2d(i), nn.ReLU(True),
            nn.Conv2d(i, o, 3, padding=1, bias=False))

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = InputCv(3, 32)
        self.downc0 = Down(32, 64)
        self.downc1 = Down(64, 128)
        self.upc2 = Up(128, 64)
        self.upc1 = Up(64, 32)
        self.outc = Out(32, 3)

    def forward(self, a, b, c, nm):
        x0 = self.inc(torch.cat((a, nm, b, nm, c, nm), dim=1))
        x1 = self.downc0(x0)
        x2 = self.upc2(self.downc1(x1))
        x1 = self.upc1(x1 + x2)
        return b - self.outc(x0 + x1)


class FastDVDnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = DenBlock()
        self.temp2 = DenBlock()

    def forward(self, x, nm):
        f = [x[:, 3 * i:3 * i + 3] for i in range(5)]
        a = self.temp1(f[0], f[1], f[2], nm)
        b = self.temp1(f[1], f[2], f[3], nm)
        c = self.temp1(f[2], f[3], f[4], nm)
        return self.temp2(a, b, c, nm)


def main():
    weights, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    net = FastDVDnet()
    net.load_state_dict(sd)
    net.eval()
    x = torch.rand(1, 15, 64, 64)
    nm = torch.full((1, 1, 64, 64), 0.04)
    fp32 = out / "fastdvdnet_fp32.onnx"
    torch.onnx.export(
        net, (x, nm), str(fp32), input_names=["frames", "sigma"], output_names=["out"],
        dynamic_axes={k: {2: "h", 3: "w"} for k in ("frames", "sigma", "out")},
        opset_version=17, dynamo=False)
    from onnxconverter_common import float16
    import onnx
    m16 = float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=False)
    onnx.save(m16, str(out / "fastdvdnet_fp16.onnx"))
    print("wrote", fp32, "and fp16")


if __name__ == "__main__":
    main()
