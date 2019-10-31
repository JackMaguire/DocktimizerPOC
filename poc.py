import math
from math import sin

factor = 25

def score_x( x ):
    value = sin( factor * x ) * x
    if value > 0:
        value = value / 5
    return value * 100#to be on the same scale as score3

def score_xy( x, y ):
    value = sin( factor * x ) * sin( factor * y ) * (x+y)
    if value > 0:
        value = value / 5
    return value * 100#to be on the same scale as score3

def score_xyz( x, y, z ):
    value = sin( factor * x ) * sin( factor * y ) * sin( factor * z ) * (x+y+z)
    if value > 0:
        value = value / 5
    return value * 100 #to be on the same scale as score3

num = 0
while num <= 1:
    print( num, score_x( num ) )
    num += 0.01


# Visualize
## https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

xspace = np.linspace(0, 1, 100)
yspace = np.linspace(0, 1, 100)
X, Y = np.meshgrid(xspace, yspace)

Z = np.copy( X )
for i in range( 0, len( Z ) ):
    for j in range( 0, len( Z[ i ] ) ):
        Z[i][j] = score_xy( X[i][j], Y[i][j] )

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot_surface(X,Y,Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Score')
#p.show()
plt.savefig('test.pdf')

def run_single_monte_carlo
