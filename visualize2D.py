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
    input = Input(shape=(6,), name="in1", dtype="float32" )
    dense1 = Dense( units=100, activation='relu' )( input )
    dense2 = Dense( units=100, activation='relu' )( dense1 )
    dense3 = Dense( units=100, activation='relu' )( dense2 )
    output = Dense( 1 )( dense3 )

    model = Model(inputs=input, outputs=output )
    metrics_to_output=[ 'accuracy' ]
    model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
    #model.summary()
    return model


def run_docktimizer():
    start = time.time()
    inputs = []
    outputs = []
    time_spent = 0
    n_init_loop = 10000
    best_score = 0

    #stage 1
    for _ in range( 0, n_init_loop ):
        x = np.random.uniform()
        y = np.random.uniform()
        z = np.random.uniform()
        a = np.random.uniform()
        b = np.random.uniform()
        c = np.random.uniform()
        score = score_6D( x, y, z, a, b, c )
        best_score = min( best_score, score )
        inputs.append( [ x, y, z, a, b, c ] )
        outputs.append( [ score ] )
    time_spent += n_init_loop * seconds_per_score

    tf.compat.v1.disable_eager_execution() #needs to be done to call tf.gradients

    #Stage 2
    n_train_loop = 1000
    #n_train_loop = 10
    samples_per_loop = 250
    #samples_per_loop = 10
    for loop in range( 0, n_train_loop ):
        print( "" )
        print( "XXX", time_spent, best_score )

        #2a generate seeds for sampling
        seeds = []
        for i in range( 0, samples_per_loop ):
            values_list = [[ np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform() ],]
            values = np.asarray( values_list )
            score = [score_6D( values[0][0], values[0][1], values[0][2], values[0][3], values[0][4], values[0][5] )]
            inputs.append( values_list[0] )
            outputs.append( score )
            seeds.append( values )
            

        #2b train model
        K.clear_session()
        model = create_model( len( inputs ) )
        #global graph
        #graph = tf.get_default_graph()
        ins = np.asarray( inputs )
        outs = np.asarray( outputs )
        #TODO 100 epochs, early stopping
        model.fit( x=ins, y=outs, batch_size=100, epochs=25, shuffle=True, validation_split=0.0, callbacks=callbacks)
        
        #2c generate next samples
        data2b = []
        for i in range( 0, samples_per_loop ):
            #printProgressBar( i, samples_per_loop, "2b" )
            data2b.append( generate_minimized_data( model, seeds[ i ] ) )

        min_this_round = 0
        #2d run next samples (can be combined with 2b)
        for input in data2b:
            score = [score_6D( input[0][0], input[0][1], input[0][2], input[0][3], input[0][4], input[0][5] )]
            min_this_round = min( min_this_round, score[0] )
            '''
            print( inputs[ 0 ] )
            print( input )
            print( input[0].tolist() )
            exit( 0 )
            '''
            inputs.append( input[0].tolist() )
            #print( score )
            outputs.append( score )
            best_score = min( best_score, score[0] )
        time_spent += len( data2b ) * seconds_per_score * 2
        print( "XXY", loop, min_this_round )
    end = time.time()
    time_spent += (end - start)
    return time_spent, best_score

#time, score = run_docktimizer()
#print( time, score )
