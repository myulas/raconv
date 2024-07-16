from cupy import unique
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAConv(nn.Module):
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
        self.bias = nn.Parameter(bias.detach().reshape(out_channels, 1))

        self.summary = summary
        self.scale_factor = scale_factor

        match self.summary:
            case "avg":
                self.summary_func = lambda fmap: fmap.mean(dim=1).mul_(self.scale_factor).int()
            case "max":
                self.summary_func = (
                    lambda fmap: fmap.max(dim=1).values.mul_(self.scale_factor).int()
                )
            case default:
                raise Exception("The summary type is invalid.")

    def __repr__(self):
        msg = (
            f"RAConv(Cin={self.in_channels}, Cout={self.out_channels}, "
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
        )
        summaries = self.summary_func(fmap)
        output = torch.empty(size=(N, self.out_channels, fmap.shape[2])).type_as(fmap)

        for i in range(N):
            _, ui, di = unique(summaries[i], return_index=True, return_inverse=True)
            ui = torch.as_tensor(ui).type_as(summaries)
            di = torch.as_tensor(di).type_as(summaries)
            output[i] = self.bias.addmm(self.weight, fmap[i].index_select(1, ui)).index_select(
                1, di
            )

        Hout = int((Hin + 2 * self.padding - self.kernel_size) / self.stride + 1)
        Wout = int((Win + 2 * self.padding - self.kernel_size) / self.stride + 1)
        output = output.reshape(N, self.out_channels, Hout, Wout)

        return output
