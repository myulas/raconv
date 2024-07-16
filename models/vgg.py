import torch.nn as nn

from layers.cpp_conv import CppConv
from layers.cpp_raconv import CppRAConv
from layers.raconv import RAConv

# fmt: off
_config_dict = {
    11: [
        64, 'M',
        128, 'M',
        256, 256, 'M',
        512, 512, 'M',
        512, 512, 'M'
    ],
    13: [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 'M',
        512, 512, 'M',
        512, 512, 'M'
    ],
    16: [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'M',
        512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    19: [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 256, 'M',
        512, 512, 512, 512, 'M',
        512, 512, 512, 512, 'M'
    ]
}
# fmt: on


class VGG(nn.Module):
    def __init__(self, version, batch_norm, num_classes):
        super().__init__()
        self.version = version
        self.batch_norm = batch_norm
        self.num_classes = num_classes
        self.features = self._get_features(_config_dict[version])
        self.classifier = self._get_classifier()

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x

    def _get_features(self, config):
        layers = []
        in_channels = 3

        for x in config:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels,
                        out_channels=x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(True),
                ]
                in_channels = x

                if not self.batch_norm:
                    layers.pop(-2)

        return nn.Sequential(*layers)

    def _get_classifier(self):
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes),
        )

    def replace_with_cpp_conv(self, conv_no):
        curr_conv_no = 0
        for idx, layer in enumerate(self.features):
            if isinstance(layer, (nn.Conv2d, CppConv, CppRAConv, RAConv)):
                curr_conv_no += 1
                if curr_conv_no == conv_no:
                    if isinstance(layer, nn.Conv2d):
                        kernel_size = layer.kernel_size[0]
                        stride = layer.stride[0]
                        padding = layer.padding[0]
                    else:
                        kernel_size = layer.kernel_size
                        stride = layer.stride
                        padding = layer.padding

                    self.features[idx] = CppConv(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        weight=layer.weight,
                        bias=layer.bias,
                    )

                    return

    def replace_with_raconv(self, is_cpp, conv_no, summary, scale_factor):
        curr_conv_no = 0
        for idx, layer in enumerate(self.features):
            if isinstance(layer, (nn.Conv2d, CppConv, CppRAConv, RAConv)):
                curr_conv_no += 1
                if curr_conv_no == conv_no:
                    if isinstance(layer, nn.Conv2d):
                        kernel_size = layer.kernel_size[0]
                        stride = layer.stride[0]
                        padding = layer.padding[0]
                    else:
                        kernel_size = layer.kernel_size
                        stride = layer.stride
                        padding = layer.padding

                    if is_cpp:
                        self.features[idx] = CppRAConv(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            weight=layer.weight,
                            bias=layer.bias,
                            summary=summary,
                            scale_factor=scale_factor,
                        )
                    else:
                        self.features[idx] = RAConv(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            weight=layer.weight,
                            bias=layer.bias,
                            summary=summary,
                            scale_factor=scale_factor,
                        )

                    return

    def get_conv_layer(self, conv_no):
        curr_conv_no = 0
        for idx, layer in enumerate(self.features):
            if isinstance(layer, (nn.Conv2d, CppConv, CppRAConv, RAConv)):
                curr_conv_no += 1
                if curr_conv_no == conv_no:
                    return layer
        return None
