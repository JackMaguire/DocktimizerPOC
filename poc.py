import math
from math import sin
import time

import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

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

# Visualize
## https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
'''
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
'''

def estimate_lowest_score():
    lowest_score = 0.0
    for x in [i * 0.01 for i in range(80, 100)]:
        for y in [i * 0.01 for i in range(80, 100)]:
            score = score_xy( x, y )
            lowest_score = min( lowest_score, score )
    return lowest_score;

print( "Lowest score: ", estimate_lowest_score() )

def run_single_monte_carlo():
    start = time.time()
    x = 0.5
    y = 0.5
    score = score_xy( x, y )
    best_score = score
    current_score = score

    neg_temp = -0.8 # Based on DockingLowRes.cc

    for  _ in range( 0, 500 ):
        trial_x = np.random.normal( x, 0.1 )
        trial_y = np.random.normal( y, 0.1 )

        if trial_x > 1:
            trial_x = 1
        if trial_x < 0:
            trial_x = 0

        if trial_y > 1:
            trial_y = 1
        if trial_y < 0:
            trial_y = 0

        trial_score = score_xy( trial_x, trial_y )
        if trial_score < current_score:
            x = trial_x
            y = trial_y
            current_score = trial_score
            if trial_score < best_score:
                best_score = trial_score
        else:
            score_delta = trial_score - current_score
            boltz_factor = score_delta / neg_temp
            probability = math.exp( min( 40.0, max( -40.0, boltz_factor ) ) )
            #print( score_delta, probability )
            if probability < 1:
                if np.random.uniform() < probability:
                    x = trial_x
                    y = trial_y
                    current_score = trial_score
    end = time.time()
    return best_score, (end - start)

best_score, runtime = run_single_monte_carlo()
print( best_score, runtime )
