'''
Created on Oct 10, 2015

@author: Cristian
'''

import numpy as np

'''If the signals are evenly sampled, then we can pre-compute lookup tables
to speed up robustness. A lookup table is computed for each signal and each
primitive. If we used to trick with the negation, then we need to compute only
two lookup tables per signal, see paper for details.
The table size is n*T^3, where T is the number of samples per trace and n is the
dimension of the signal.
'''

def trace_robustness_lkt_1d(trace):
    '''Computes the lookup table for an one-dimensional trace.'''
    T = len(trace)
    # 1. Create t_i, t3 matrix
    slk = np.full((T, T), np.inf, dtype=np.float32)
    slk[0] = trace
    for i in range(1, T):
        n = T-i
        slk[i, :n] = np.minimum(slk[i-1, :n], slk[i-1, 1:n+1])
    # 2. Create 3-dimensional array t0, delta1, t3
    lk = np.full((T, T, T), -np.inf, dtype=np.float32)
    for t3 in range(0, T):
        n = T-t3
        lk[:n, 0, t3] = slk[t3, :n]
        for j in range(1, n):
            m = n-j
            lk[:m, j, t3] = np.maximum(lk[:m, j-1, t3], lk[1:m+1, j-1, t3])
    return lk

def trace_robustness_lkt_nd(trace):
    '''Computes the lookup table for a n-dimensional trace'''
    return np.array([trace_robustness_lkt_1d(tr) for tr in trace],
                    dtype=np.float32)

def traces_robustness_lkt(traces):
    '''Computes the lookup table for all traces.'''
    traces.traces_lkt_max_min = np.array(
        [trace_robustness_lkt_nd(tr) for tr in traces.signals],
        dtype=np.float32)
    traces.traces_lkt_min_max = np.array(
        [-trace_robustness_lkt_nd(-tr) for tr in traces.signals],
        dtype=np.float32)

def robustness_lookup(t0, t1, t3, mu, sindex, pred_less, ev_al, traces):
    '''Robustness lookup for primitives
    F_{[t_0, t_1]} (G_{[0, t_3]} (s_{sindex} \leq mu)), if ev_al is true,
    G_{[t_0, t_1]} (F_{[0, t_3]} (s_{sindex} \leq mu)), otherwise.
    If pred_less is False then the predicate is assumed to be negated, i.e. gt.
    '''
    t0, t1, t3 = np.floor(t0), np.ceil(t1), np.ceil(t3)
    sgn = 1 if pred_less else -1
    if ev_al:
        return sgn*(traces.traces_lkt_max_min[:, sindex, t0, t1-t0, t3] - mu)
    return sgn*(traces.traces_lkt_min_max[:, sindex, t0, t1-t0, t3] - mu)

if __name__ == '__main__':
    pass