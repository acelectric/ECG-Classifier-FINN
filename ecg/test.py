import brevitas.nn as qnn
import brevitas.quant_tensor as QuantTensor
import torch.nn as nn


class BrevitasIEEE(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            qnn.QuantConv1d(1, 64, kernel_size=5),
            nn.QuantReLU(inplace=True),
            qnn.QuantConv1d(64, 64, kernel_size=5),
            nn.QuantReLU(inplace=True),
            nn.MaxPool1d(2),
            qnn.QuantConv1d(64, 128, kernel_size=3),
            nn.QuantReLU(inplace=True),
            qnn.QuantConv1d(128, 128, kernel_size=3),
            nn.QuantReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            qnn.QuantLinear(128 * 28, 256),
            qnn.QuantLinear(256, 128),
            qnn.QuantLinear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 128 * 28)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x
