import torch


@torch.inference_mode()
def _test_function(
    num_trials,
    batch_size,
    input_height,
    input_width,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    summary,
    scale_factor,
):
    import time

    import torchmetrics
    import torch.nn as nn

    from layers.cpp_raconv import CppRAConv
    from layers.raconv import RAConv

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    layer_cpp = CppRAConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight=conv.weight,
        bias=conv.bias,
        summary=summary,
        scale_factor=scale_factor,
    )

    layer_python = RAConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight=conv.weight,
        bias=conv.bias,
        summary=summary,
        scale_factor=scale_factor,
    )

    time_python = torchmetrics.MeanMetric()
    results = torch.zeros(num_trials, dtype=torch.bool)
    for i in range(num_trials):
        fmap = torch.randn(size=(batch_size, in_channels, input_height, input_width))

        tick = time.perf_counter()
        output_python = layer_python(fmap)
        time_python.update(time.perf_counter() - tick)

        output_cpp = layer_cpp(fmap)

        results[i] = torch.allclose(output_python, output_cpp, atol=1e-6)

    print(f"Results are the same: {results.all().item()}")
    print(f"Python: {round(time_python.compute().item() * 1e3, 3)}ms")
    print(f"C++   : {round(layer_cpp.exec_time.compute().item() * 1e3, 3)}ms")


if __name__ == "__main__":
    from utils.methods import set_seed

    set_seed(2024)

    for summary in ["avg", "max"]:
        _test_function(
            num_trials=100,
            batch_size=6,
            input_height=300,
            input_width=200,
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            summary=summary,
            scale_factor=1e4,
        )
