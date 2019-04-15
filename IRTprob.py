#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:12:06 2019

@author: lxu
"""

import numpy as np
import sklearn as sklearn
import itertools

class IRTprob(object):
    def __init__(self, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=300, batch_size=1000):



        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)


        self.beta2 = None
        self.alpha = None

        self.beta2_inc = None
        self.alpha_inc = None

        self.logloss_train = []
        self.logloss_test = []
        self.auc_train = []
        self.auc_test = []
        self.baseline_auc_test= []

        self.classification_train = []
        self.classification_test = []


    def fit(self, train_vec, test_vec,  num_user, num_problem):
        pairs_train = train_vec.shape[0]
        pairs_test = test_vec.shape[0]

        # average scores
        self.mean_inv = np.mean(train_vec['correct'])
        self.epoch = 0
        # initialization

        self.beta2 = 0.1 * np.random.randn(num_problem)
        self.alpha = 0.1 * np.random.randn(num_user)


        self.beta2_inc = np.zeros(num_problem)
        self.alpha_inc = np.zeros(num_user)

        while self.epoch < self.maxepoch:
            self.epoch += 1
            # shuffle training tuples
            shuffled_order = np.arange(pairs_train)
            np.random.shuffle(shuffled_order)

            # batch update
            for batch in range(self.num_batches):
                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # index used in this batch

                # compute gradient of obj
                batch_UserID = np.array(train_vec.loc[shuffled_order[batch_idx], 'user_id'], dtype='int32')
                batch_ProbID = np.array(train_vec.loc[shuffled_order[batch_idx], 'problem_id'], dtype='int32')
                x = self.alpha[batch_UserID] + self.beta2[batch_ProbID]

                expnx = np.exp(-x)
                linkx = np.divide(1, (1+expnx))
                y = train_vec.loc[shuffled_order[batch_idx], 'correct'].values
                gradlogloss = -  np.multiply(y,1-linkx) + np.multiply(1-y, linkx)

                # compute the gradient
                dw_beta2 = np.zeros(num_problem)
                dw_alpha = np.zeros(num_user)

                beta2_grad = 2 * gradlogloss + self._lambda * self.beta2[batch_ProbID]
                alpha_grad = 2 * gradlogloss + self._lambda * self.alpha[batch_UserID]

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_beta2[batch_ProbID[i]]  += beta2_grad[i]
                    dw_alpha[batch_UserID[i]]  += alpha_grad[i]

                # Update with momentum
                self.beta2_inc = self.momentum * self.beta2_inc + self.epsilon * dw_beta2 / self.batch_size
                self.alpha_inc = self.momentum * self.alpha_inc + self.epsilon * dw_alpha / self.batch_size


                # gradien descent
                self.beta2 = self.beta2 - self.beta2_inc
                self.alpha = self.alpha - self.alpha_inc



                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    batch_UserID = np.array(train_vec.loc[:, 'user_id'], dtype='int32')
                    batch_ProbID = np.array(train_vec.loc[:, 'problem_id'], dtype='int32')

                    x = self.alpha[batch_UserID] + self.beta2[batch_ProbID]
                    expnx = np.exp(-x)
                    linkx = np.divide(1, (1+expnx))
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)

                    y = train_vec.loc[:, 'correct'].values
                    logloss =  np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx))

                    obj = logloss + 0.5 * self._lambda * ( np.linalg.norm(self.beta2) ** 2 + \
                         np.linalg.norm(self.alpha) ** 2)

                    self.logloss_train.append((obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    batch_UserID = np.array(test_vec.loc[:, 'user_id'], dtype='int32')
                    batch_ProbID = np.array(test_vec.loc[:, 'problem_id'], dtype='int32')
                    x = self.alpha[batch_UserID] + self.beta2[batch_ProbID]
                    expnx = np.exp(-x)
                    linkx = np.divide(1, (1+expnx))
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)
                    y = test_vec.loc[:, 'correct'].values
                    logloss =  np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx))
                    auc = sklearn.metrics.roc_auc_score(np.array(y, dtype=bool), linkx)
                    self.auc_test.append(auc)
                    self.logloss_test.append((logloss) / (pairs_test))
                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training logloss: %f, Test logloss %f, Test AUC %f' \
                              % (self.logloss_train[-1], self.logloss_test[-1], self.auc_test[-1]))



    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

