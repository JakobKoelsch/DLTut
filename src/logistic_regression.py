# after http://deeplearning.net/tutorial/logreg.htm

import theano
import theano.tensor as T
import numpy as np

n_in = 5
n_out = 3


# initialize W with 0 as a matrix of shape (n_in, n_out)
W = theano.shared(
    value = np.zeros(
        (n_in, n_out),
        dtype=theano.config.floatX
    ),
    name='W',
    borrow=True
)

#initialize biases b as a vector of (n_out) 0s

b = theano.shared(
    value=np.zeros(
        (n_out,),
        dtype=theano.config.floatX
    ),
    name='b',
    borrow=True
)

p_y_given_x = T.nnet.softmax(T.dot(input, W)+ b)

y_pred = T.argmax(p_y_given_x, axis=1)