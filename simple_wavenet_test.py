import time
from wavenet import *
import torch

dtype = torch.FloatTensor
ltype = torch.LongTensor



model = WaveNetModel(layers=8,
                     blocks=4,
                     dilation_channels=16,
                     residual_channels=16,
                     skip_channels=16,
                     output_length=8,
                     dtype=dtype,
                     is_verbose=False)

x = torch.randn(1,256,10000)
out = model(x)
print(out.shape)