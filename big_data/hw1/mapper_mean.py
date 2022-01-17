#!/usr/bin/env python
"""mapper_mean.py"""

import sys
import numpy as np

KEY = '1'

# input comes from STDIN (standard input)
for line in sys.stdin:
    # Clean and cast the input line
    prices_line = line.strip()
    prices_list = list(map(float, prices_line.split()))
    c_size = len(prices_list)
    mean = np.mean(prices_list)
    
    sys.stdout.write('%s\t%s,%s\n'%(KEY, c_size, mean))


