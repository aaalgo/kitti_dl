#!/usr/bin/env python3

import sys 
import math
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
from scipy.stats import norm
import subprocess as sp
from tqdm import tqdm
from kitti import *

#Z: 1.71389 0.379994
#H: 1.52715 0.136921
#W: 1.62636 0.102967
#L: 3.88716 0.430096
#T: -0.0743265 1.71175


Z = []
H = []
W = []
L = []
T = []

sp.check_call('mkdir -p stats', shell=True)

with open('train.txt', 'r') as f:
    tasks = [int(l.strip()) for l in f]
for pk in tqdm(tasks):
    sample = Sample(pk, LOAD_LABEL2)
    #[z, x, y, h, w, l, obj.rot, _])
    boxes = sample.get_voxelnet_boxes(["Car"])
    for box in boxes:
        x, y, z, h, w, l, t, _ = box
        Z.append(z)
        H.append(h)
        W.append(w)
        L.append(l)
        T.append(t)
        pass
    pass

def hist_plot (X, label):
    # 对X值画直方图并估计正太分布, 限制在limits范围内
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    mu, sigma = norm.fit(X)
    n, bins, _ = ax.hist(X, 100, density=True)
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y)
    ax.set_xlabel(label)
    print('%s: %g %g' % (label, mu, sigma))
    fig.savefig('stats/%s.png' % label)
    pass

hist_plot(Z, 'Z')
hist_plot(H, 'H')
hist_plot(W, 'W')
hist_plot(L, 'L')
hist_plot(T, 'T')

