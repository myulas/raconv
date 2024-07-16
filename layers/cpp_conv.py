import gc
import time

import torchmetrics
import torch.nn as nn
import torch.nn.functional as F


class CppConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        patch_size = kernel_size * kernel_size * in_channels
        self.weight = nn.Parameter(weight.detach().reshape(out_channels, patch_size))
        self.bias = nn.Parameter(bias.detach().reshape(out_channels))

        self.exec_time = torchmetrics.MeanMetric()

        from layers.bindings import module

        self.cpp_convolution = module.convolution

    def __repr__(self):
        msg = (
            f"CppConv(Cin={self.in_channels}, Cout={self.out_channels}, "
            f"K={self.kernel_size}, S={self.stride}, P={self.padding})"
        )

        return msg

    def forward(self, fmap):
        N, _, Hin, Win = fmap.shape
        fmap = F.unfold(
            fmap,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).permute(0, 2, 1)

        gc.disable()
        tick = time.perf_counter()

        output = self.cpp_convolution(fmap, self.weight, self.bias)

        self.exec_time.update(time.perf_counter() - tick)
        gc.enable()

        Hout = int((Hin + 2 * self.padding - self.kernel_size) / self.stride + 1)
        Wout = int((Win + 2 * self.padding - self.kernel_size) / self.stride + 1)
        output = output.permute(0, 2, 1).reshape(N, self.out_channels, Hout, Wout)

        return output
