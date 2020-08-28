# -*- coding: utf-8 -*-

import torch
import numpy as np


x = torch.rand(5,1, requires_grad = True)

x_ = x.clone()

x_ = torch.mm(x_, x_.T)

loss = torch.sum(x_)

loss.backward()
