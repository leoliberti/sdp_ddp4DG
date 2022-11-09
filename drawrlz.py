#!/usr/bin/env python

## draw the 2D/3D realization in input using matplotlib

import sys
import numpy as np
import math
import types
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read input file
lines = [line.rstrip('\n').split()[1:] for line in open(sys.argv[1])]

# number of vertices
n = len(lines)

# number of dimensions
K = min([len(line) for line in lines])

# turn into float array
x = np.array([[float(lines[i][j]) for j in range(K)] for i in range(n)])

# draw
if K == 2:
    plt.scatter(x[:,0], x[:,1])
    plt.plot(x[:,0], x[:,1])
elif K == 3:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:,0], x[:,1], x[:,2])
    ax.plot(x[:,0], x[:,1], x[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
plt.show()
