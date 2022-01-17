#!/usr/bin/env python
"""reducer_var.py"""

import sys


def calculate_variance(c_j, c_k, m_j, m_k, v_j, v_k):
    var_term = (c_j * v_j + c_k * v_k) / (c_j + c_k)
    mean_term = (c_j * c_k) * (((m_j - m_k) / (c_j + c_k)) ** 2)
    return var_term + mean_term


def calculate_mean(c_j, c_k, m_j, m_k):
    return (c_j * m_j + c_k * m_k) / (c_j + c_k)


KEY = '1'
cur_size = 0
cur_mean = 0.0
cur_variance = 0.0

for line in sys.stdin:
    
    line = line.split('\t')
    key = line[0]
    value = line[1].strip().split(',')
    
    if key == KEY:

        chunk_size = int(value[0])
        chunk_mean = float(value[1])
        chunk_variance = float(value[2])

        cur_variance = calculate_variance(cur_size, chunk_size, cur_mean, chunk_mean, cur_variance, chunk_variance)
        cur_mean = calculate_mean(cur_size, chunk_size, cur_mean, chunk_mean)
        cur_size = cur_size + chunk_size

sys.stdout.write('%s\n'%cur_variance)