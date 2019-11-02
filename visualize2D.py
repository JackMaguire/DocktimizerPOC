import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
from math import sin
import time

import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.compat.v1 as tf1

factor = 25
seconds_per_score = 1.0

def schedule( epoch, lr ):
    if lr < 0.0001:
        return lr * 2
    return lr * 0.9
lrs = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
callbacks=[lrs]


def score_xy( x, y ):
    value = sin( factor * x ) * sin( factor * y ) * (x+y)
    if value > 0:
        value = value / 5
    return value * 100#to be on the same scale as score3


# Visualize
## https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
def draw_target():
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
    plt.savefig('target.png')

draw_target()

def estimate_lowest_score_2D():
    lowest_score = 0.0
    for x in [i * 0.01 for i in range(80, 100)]:
        for y in [i * 0.01 for i in range(80, 100)]:
            score = score_xy( x, y )
            lowest_score = min( lowest_score, score )
    return lowest_score;

#print( "Lowest 2D score: ", estimate_lowest_score_2D() )


def create_model():
    input = Input(shape=(2,), name="in1", dtype="float32" )
    dense1 = Dense( units=100, activation='relu' )( input )
    dense2 = Dense( units=100, activation='relu' )( dense1 )
    dense3 = Dense( units=100, activation='relu' )( dense2 )
    dense4 = Dense( units=100, activation='relu' )( dense3 )
    dense5 = Dense( units=100, activation='relu' )( dense4 )
    output = Dense( 1 )( dense5 )

    model = Model(inputs=input, outputs=output )
    metrics_to_output=[ 'accuracy' ]
    model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
    #model.summary()
    return model

def train_model_on_data( inputs, outputs ):
    model = create_model()
    ins = np.asarray( inputs )
    outs = np.asarray( outputs )
    model.fit( x=ins, y=outs, batch_size=100, epochs=25, shuffle=True, validation_split=0.0, callbacks=callbacks)
    return model

def generate_image_from_model( model, num ):
    xspace = np.linspace(0, 1, 100)
    yspace = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xspace, yspace)

    Z = np.copy( X )
    for i in range( 0, len( Z ) ):
        for j in range( 0, len( Z[ i ] ) ):
            input = np.asarray( [[ X[i][j], Y[i][j] ],] )
            Z[i][j] = model.predict( input )[0]

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.plot_surface(X,Y,Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Score')
    #p.show()
    plt.savefig("model_" + str(num).zfill( 5 ) + ".png")

def generate_ML_pics( num_pics, samples_per_frame ):
    inputs = []
    outputs = []
    for i in range( 0, num_pics ):
        for j in range( 0, samples_per_frame ):
            x = np.random.uniform()
            y = np.random.uniform()
            score = score_xy( x, y )
            inputs.append( [ x, y ] )
            outputs.append( [ score ] )
        model = train_model_on_data( inputs, outputs )
        generate_image_from_model( model, i )

generate_ML_pics( 1, 10000 )
