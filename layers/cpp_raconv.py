import gc
import time

import torchmetrics
import torch.nn as nn
import torch.nn.functional as F


class CppRAConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        weight,
        bias,
        summary,
        scale_factor,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        patch_size = kernel_size * kernel_size * in_channels
        self.weight = nn.Parameter(weight.detach().reshape(out_channels, patch_size))
        self.bias = nn.Parameter(bias.detach().reshape(out_channels))

        self.summary = summary
        self.scale_factor = scale_factor

        self.exec_time = torchmetrics.MeanMetric()
        self.num_unique_patches = torchmetrics.MeanMetric()
        self.num_patches = 0
        self.memory_size = torchmetrics.MeanMetric()

        from layers.bindings import module

        match self.summary:
            case "avg":
                self.raconv_func = module.raconvolution_avg
                self.summary_func = lambda fmap: fmap.mean(dim=2).mul_(self.scale_factor).int()
            case "max":
                self.raconv_func = module.raconvolution_max
                self.summary_func = (
                    lambda fmap: fmap.max(dim=2).values.mul_(self.scale_factor).int()
                )
            case default:
                raise Exception("The summary type is invalid.")

    def __repr__(self):
        msg = (
            f"CppRAConv(Cin={self.in_channels}, Cout={self.out_channels}, "
            f"K={self.kernel_size}, S={self.stride}, P={self.padding}, "
            f"summary={self.summary.upper()}, scale_factor={self.scale_factor:.2E})"
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

        memory_size, min_summary = self._get_memory_info(fmap)

        gc.disable()
        tick = time.perf_counter()

        output = self.raconv_func(
            fmap, self.weight, self.bias, self.scale_factor, memory_size, min_summary
        )

        self.exec_time.update(time.perf_counter() - tick)
        gc.enable()

        Hout = int((Hin + 2 * self.padding - self.kernel_size) / self.stride + 1)
        Wout = int((Win + 2 * self.padding - self.kernel_size) / self.stride + 1)
        self.num_patches = Hout * Wout
        output = output.permute(0, 2, 1).reshape(N, self.out_channels, Hout, Wout)

        return output

    def _get_memory_info(self, fmap):
        summaries = self.summary_func(fmap)
        batch_min_summary = batch_max_summary = batch_memory_size = 0

        for i in range(fmap.size(0)):
            self.num_unique_patches.update(summaries[i].unique().numel())

            min_summary = summaries[i].min().item()
            max_summary = summaries[i].max().item()
            self.memory_size.update(max_summary - min_summary + 1)

            if min_summary < batch_min_summary:
                batch_min_summary = min_summary
            if max_summary > batch_max_summary:
                batch_max_summary = max_summary

            batch_memory_size = batch_max_summary - batch_min_summary + 1

        return batch_memory_size, batch_min_summary
