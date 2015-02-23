# encoding=utf-8
#author: Bocharov Ivan

import numpy
import theano
import theano.tensor as T
import logging

rng = numpy.random


class TheanoLogisticRegression(object):

    def __init__(self, features_count):
        self.weights = theano.shared(rng.randn(features_count), name="w")
        self.b = theano.shared(0., name="b")
        self.x = T.matrix("x")
        self.y = T.vector("y")
        self.construct_expression_graph()
        self.compile()

    def construct_expression_graph(self):
        p_1 = 1 / (1 + T.exp(-T.dot(self.x, self.weights) - self.b))
        self.prediction = p_1 > 0.5
        self.xent = -self.y * T.log(p_1) - (1-self.y) * T.log(1-p_1)
        self.cost = self.xent.mean() + 0.01 * (self.weights ** 2).sum()
        self.weights_grad, self.b_grad = T.grad(self.cost, [self.weights, self.b])

        self.example = T.vector("v")
        self.scalar_dot = T.dot(self.example, self.weights) + self.b
        self.factor_function = theano.function(inputs=[self.example], outputs=self.scalar_dot)

    def compile(self):
            # Compile
        self.train_function = theano.function(
                  inputs=[self.x, self.y],
                  outputs=[self.prediction, self.xent],
                  updates=((self.w, self.w - 0.1 * self.weights_grad), (self.b, self.b - 0.1 * self.b_grad)))
        self.predict_function = theano.function(inputs=[self.x], outputs=self.prediction)

    def train(self, training_steps, factors, answers):
        for i in xrange(training_steps):
            pred, err = self.train(factors, answers)
            logging.debug("Error on step {} equals: {}".format(i, err))


if __name__ == '__main__':
    pass