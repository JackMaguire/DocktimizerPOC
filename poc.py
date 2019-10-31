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

def score_6D( x, y, z, a, b, c ):
    value = sin( factor * x ) * sin( factor * y ) * sin( factor * z ) *sin( factor * a ) * sin( factor * b ) * sin( factor * c ) * (x+y+z+a+b+c)
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

#https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def estimate_lowest_score_2D():
    lowest_score = 0.0
    for x in [i * 0.01 for i in range(80, 100)]:
        for y in [i * 0.01 for i in range(80, 100)]:
            score = score_xy( x, y )
            lowest_score = min( lowest_score, score )
    return lowest_score;

def estimate_lowest_score_3D():
    lowest_score = 0.0
    for x in [i * 0.001 for i in range(800, 1000)]:
        for y in [i * 0.001 for i in range(800, 1000)]:
            for z in [i * 0.001 for i in range(800, 1000)]:
                score = score_xyz( x, y, z )
                lowest_score = min( lowest_score, score )
    return lowest_score;

def estimate_lowest_score_6D():
    lowest_score = 0.0
    for x in [i * 0.01 for i in range(80, 100)]:
        for y in [i * 0.01 for i in range(80, 100)]:
            for z in [i * 0.01 for i in range(80, 100)]:
                for a in [i * 0.01 for i in range(80, 100)]:
                    for b in [i * 0.01 for i in range(80, 100)]:
                        for c in [i * 0.01 for i in range(80, 100)]:
                            score = score_6D( x, y, z, a, b, c )
                            lowest_score = min( lowest_score, score )
    return lowest_score;


#print( "Lowest 2D score: ", estimate_lowest_score_2D() )
#print( "Lowest 3D score: ", estimate_lowest_score_3D() )
#print( "Lowest 6D score: ", estimate_lowest_score_6D() )

def run_single_monte_carlo_2D():
    start = time.time()
    x = 0.5
    y = 0.5
    score = score_xy( x, y )
    best_score = score
    current_score = score

    neg_temp = -0.8 # Based on DockingLowRes.cc

    for  _ in range( 0, 500 ):
        trial_x = np.random.normal( x, 0.05 )
        trial_y = np.random.normal( y, 0.05 )

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
                #print( x, y )
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

#best_score, runtime = run_single_monte_carlo_2D()
#print( "2D", best_score, runtime )

def run_single_monte_carlo_3D():
    start = time.time()
    x = 0.5
    y = 0.5
    z = 0.5
    score = score_xyz( x, y, z )
    best_score = score
    current_score = score

    neg_temp = -0.8 # Based on DockingLowRes.cc

    for  _ in range( 0, 500 ):
        trial_x = np.random.normal( x, 0.1 )
        trial_y = np.random.normal( y, 0.1 )
        trial_z = np.random.normal( z, 0.1 )

        if trial_x > 1:
            trial_x = 1
        if trial_x < 0:
            trial_x = 0

        if trial_y > 1:
            trial_y = 1
        if trial_y < 0:
            trial_y = 0

        if trial_z > 1:
            trial_z = 1
        if trial_z < 0:
            trial_z = 0

        trial_score = score_xyz( trial_x, trial_y, trial_z )
        if trial_score < current_score:
            x = trial_x
            y = trial_y
            z = trial_z
            current_score = trial_score
            if trial_score < best_score:
                best_score = trial_score
                #print( x, y )
        else:
            score_delta = trial_score - current_score
            boltz_factor = score_delta / neg_temp
            probability = math.exp( min( 40.0, max( -40.0, boltz_factor ) ) )
            #print( score_delta, probability )
            if probability < 1:
                if np.random.uniform() < probability:
                    x = trial_x
                    y = trial_y
                    z = trial_z
                    current_score = trial_score
    end = time.time()
    return best_score, (end - start)

#best_score, runtime = run_single_monte_carlo_3D()
#print( "3D", best_score, runtime )

'''
def run3DMC():
    print( "Starting 3D run" )
    best_score = 0
    time = 0
    for _ in range( 0, 1000 ):
        score, runtime = run_single_monte_carlo_3D()
        time += runtime
        if score < best_score:
            best_score = score
        print( time, best_score )

run3DMC()
'''

def run_single_monte_carlo_6D():
    start = time.time()
    x = 0.5
    y = 0.5
    z = 0.5
    a = 0.5
    b = 0.5
    c = 0.5
    score = score_6D( x, y, z, a, b, c )
    best_score = score
    current_score = score

    neg_temp = -0.8 # Based on DockingLowRes.cc

    for  _ in range( 0, 500 ):
        trial_x = np.random.normal( x, 0.1 )
        trial_y = np.random.normal( y, 0.1 )
        trial_z = np.random.normal( z, 0.1 )
        trial_a = np.random.normal( a, 0.1 )
        trial_b = np.random.normal( b, 0.1 )
        trial_c = np.random.normal( c, 0.1 )

        if trial_x > 1:
            trial_x = 1
        if trial_x < 0:
            trial_x = 0

        if trial_y > 1:
            trial_y = 1
        if trial_y < 0:
            trial_y = 0

        if trial_z > 1:
            trial_z = 1
        if trial_z < 0:
            trial_z = 0

        if trial_a > 1:
            trial_a = 1
        if trial_a < 0:
            trial_a = 0

        if trial_b > 1:
            trial_b = 1
        if trial_b < 0:
            trial_b = 0

        if trial_c > 1:
            trial_c = 1
        if trial_c < 0:
            trial_c = 0

        trial_score = score_6D( trial_x, trial_y, trial_z, trial_a, trial_b, trial_c )
        if trial_score < current_score:
            x = trial_x
            y = trial_y
            z = trial_z
            a = trial_a
            b = trial_b
            c = trial_c
            current_score = trial_score
            if trial_score < best_score:
                best_score = trial_score
                #print( x, y )
        else:
            score_delta = trial_score - current_score
            boltz_factor = score_delta / neg_temp
            probability = math.exp( min( 40.0, max( -40.0, boltz_factor ) ) )
            #print( score_delta, probability )
            if probability < 1:
                if np.random.uniform() < probability:
                    x = trial_x
                    y = trial_y
                    z = trial_z
                    a = trial_a
                    b = trial_b
                    c = trial_c
                    current_score = trial_score
    end = time.time()
    #return best_score, (end - start) + (seconds_per_score * 500)
    return best_score, (seconds_per_score * 500)

def run6DMC():
    print( "Starting 6D run" )
    best_score = 0
    time = 0
    for _ in range( 0, 1000 ):
        score, runtime = run_single_monte_carlo_6D()
        time += runtime
        if score < best_score:
            best_score = score
        print( time, best_score )

#run6DMC()

def create_model( num_elements ):
    dense_size = int( 100 + math.sqrt( num_elements / 100 ) )

    input = Input(shape=(6,), name="in1", dtype="float32" )
    '''
    dense1 = Dense( units=dense_size, activation='relu' )( input )
    dense2 = Dense( units=100, activation='relu' )( dense1 )
    dense3 = Dense( units=(dense_size/2), activation='relu' )( dense2 )
    '''
    dense1 = Dense( units=100, activation='relu' )( input )
    dense2 = Dense( units=100, activation='relu' )( dense1 )
    dense3 = Dense( units=100, activation='relu' )( dense2 )
    output = Dense( 1 )( dense3 )

    model = Model(inputs=input, outputs=output )
    metrics_to_output=[ 'accuracy' ]
    model.compile( loss='mean_squared_error', optimizer='adam', metrics=metrics_to_output )
    #model.summary()
    return model

def generate_minimized_data( model ):
    best_score = 1000
    values = 0
    for i in range( 0, 10 ):
        ivalues_list = [[ np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform() ],]
        ivalues = np.asarray( ivalues_list )
        pred = model.predict( ivalues )
        if pred[0] < best_score or i == 0:
            best_score = pred[0]
            values = ivalues

    #tf.compat.v1.disable_eager_execution() #needs to be done to call tf.gradients
    #K.clear_session()
        
    pred = model.predict( values )

    m_input = model.input
    m_output = model.output
    m_loss = model.output
    m_grad = tf.gradients(m_loss, m_input)[0]

    #print( "values before anything" )
    #print( values )
    #print( "pred before anything" )
    #print( pred )
    sess = tf1.keras.backend.get_session()
    coeff = 0.1
    for _ in range( 0, 25 ):
        values -= coeff * sess.run( m_grad, {m_input:values})
        coeff /= 2.0
        for i in range( 0, 6 ):
            values[0][i] = min( 1.0, values[0][i] )
            values[0][i] = max( 0.0, values[0][i] )
        #print( values )
    #print( "pred at the end" )
    #print( model.predict( values ) )

    return values

def run_docktimizer():
    start = time.time()
    inputs = []
    outputs = []
    time_spent = 0
    n_init_loop = 100000
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
        print( "XXX", loop, best_score )
        #2a train model
        K.clear_session()
        model = create_model( len( inputs ) )
        #global graph
        #graph = tf.get_default_graph()
        ins = np.asarray( inputs )
        outs = np.asarray( outputs )
        #TODO 100 epochs, early stopping
        model.fit( x=ins, y=outs, batch_size=100, epochs=25, shuffle=True, validation_split=0.0, callbacks=callbacks)


        #2b generate next samples
        data2b = []
        for i in range( 0, samples_per_loop ):
            printProgressBar( i, samples_per_loop, "2b" )
            data2b.append( generate_minimized_data( model ) )
        
        min_this_round = 0
        #2c run next samples (can be combined with 2b)
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
        print( "XXY", loop, min_this_round )
    end = time.time()
    time_spent += (end - start)
    return time_spent, best_score

time, score = run_docktimizer()
print( time, score )
