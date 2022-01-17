#!/usr/bin/env python
"""mapper_var.py"""

import sys
import numpy as np

KEY = '1'


for line in sys.stdin:
    
    prices_line = line.strip()
    prices_list = list(map(float, prices_line.split()))

    c_size = len(prices_list)
    variance = np.var(prices_list)
    mean = np.mean(prices_list)
    
    sys.stdout.write('%s\t%s,%s,%s\n'%(KEY, c_size, mean, variance))
