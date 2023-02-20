# -*- coding: utf-8 -*-

"""
@File    : test.py
@Description:
@Author  : zqgCcoder
@Time    : 2023/2/17 16:32
"""
import torch
import torch.nn as nn

x = torch.rand((2, 3, 8))
print('pre:')
print(x)

print(x.view(2, 3, 2, 2, 2))

print('reshape2:')
print(x.view(2, 6, 2, 2))
