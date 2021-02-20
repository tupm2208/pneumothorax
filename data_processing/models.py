from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

class EfnUnet(nn.Module):
    def __init__(self):
        super().__init__()
        main_net = EfficientNet.from_pretrained("efficientnet-b2")

        self._conv_stem = main_net._conv_stem
        self._conv_head = main_net._conv_head
        self._bn0 = main_net._bn0
        self._bn1 = main_net._bn1
        self._blocks = main_net._blocks
        self._global_params = main_net._global_params
        self._swish = MemoryEfficientSwish()
    
    def forward(self, inputs):

        
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

if __name__ == '__main__':
    net = EfnUnet()
    data = torch.rand((1, 3, 512, 512))
    print(net(data).shape)