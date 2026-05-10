#!/usr/bin/env python

import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(1000, 1000).cuda()
y = x @ x
print("Computation successful:", y.shape)
