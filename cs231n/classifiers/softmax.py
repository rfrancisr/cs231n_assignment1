import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in xrange(num_train):
    score = X[i].dot(W)
    log_const = -1.0*np.max(score)
    score += log_const
    exp_score = np.exp(score)
    numerator = 0
    denominator = np.sum(exp_score)
    for j in xrange(num_class):
      if j == y[i]:
        numerator = exp_score[j]
        dW[:, j] += - np.dot(X[i].T, (1 - numerator/denominator))
        continue
      dW[:, j] += np.dot(X[i].T, exp_score[j]/denominator)
    prob = numerator/denominator
    loss_i = -np.log(prob)
    loss += loss_i
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  score = np.dot(X, W)
  score -= np.max(score, axis=1, keepdims=True)
  exp_score = np.exp(score)
  numerator = exp_score[np.arange(num_train), y]
  numerator = numerator[:, np.newaxis]
  denominator = np.sum(exp_score, axis=1, keepdims=True)
  loss_i = -np.log(numerator / denominator)
  loss = np.sum(loss_i)
  loss /= num_train
  loss += reg * np.sum(W * W)

  dscore = exp_score/denominator
  dscore[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, dscore)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

