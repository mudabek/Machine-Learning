#!/usr/bin/env python
"""reducer_mean.py"""

import sys

def calculate_mean(c_j, c_k, m_j, m_k):
    return (c_j * m_j + c_k * m_k) / (c_j + c_k)

KEY = '1'
cur_size = 0
cur_mean = 0.0

for line in sys.stdin:
    line = line.split('\t')
    key = line[0]
    value = line[1].strip().split(',')
    
    if key == KEY:

        chunk_size = int(value[0])
        chunk_mean = float(value[1])

        cur_mean = calculate_mean(cur_size, chunk_size, cur_mean, chunk_mean)
        cur_size = cur_size + chunk_size

sys.stdout.write('%s\n'%cur_mean)

