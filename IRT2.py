#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:12:06 2019

@author: lxu
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools

class IRT2(object):
    def __init__(self, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=300, batch_size=1000,
                    problem=True, MF=True, num_feat=20, user=True,  global_bias=True,
                    problem_dyn_embedding=False):

        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
        self.MF = MF
        self.num_feat = num_feat
        self.problem = problem
        self.user = user
        self.problem_dyn_embedding = problem_dyn_embedding
        self.global_bias = global_bias

        self.beta_prob = None
        self.beta_user = None
        self.beta_global = None

        self.gamma = None  # success
        self.rho = None  # failure

        self.w_prob = None  # problem feature vectors
        self.w_user = None  # user feature vectors

        self.h_user = None # dynamic, user learning ability
        self.h_prob = None # dynamic, skill teaching ability

        self.beta_prob_inc = None
        self.beta_user_inc = None
        self.beta_global_inc = None

        self.gamma_inc = None
        self.rho_inc = None

        self.w_prob_inc = None  # problem feature vectors
        self.w_user_inc = None  # user feature vectors


        self.h_prob_inc = None
        self.h_user_inc  = None

        self.logloss_train = []
        self.logloss_test = []
        self.auc_train = []
        self.auc_test = []
        self.acc_train = []
        self.acc_test= []
        self.baseline_auc_test= []

        self.classification_train = []
        self.classification_test = []


    def fit(self, train_vec, test_vec,  num_user, num_prob):
        pairs_train = train_vec.shape[0]
        pairs_test = test_vec.shape[0]

        # average scores
        self.epoch = 0

        # initialization
        self.beta_prob = 0.1 * np.random.randn(num_prob)
        self.beta_user = 0.1 * np.random.randn(num_user)
        self.beta_global = 0.1 * np.random.randn(1)
        self.w_prob = 0.1 * np.random.rand(num_prob, self.num_feat)  # problem latent matrix
        self.w_user = 0.1 * np.random.randn(num_user, self.num_feat)  # user latent matrix

        self.beta_prob_inc = np.zeros(num_prob)
        self.beta_user_inc = np.zeros(num_user)
        self.beta_global_inc = np.zeros(1)
        self.w_prob_inc = np.zeros((num_prob, self.num_feat))
        self.w_user_inc = np.zeros((num_user, self.num_feat))

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
                x = self.CalcPrediction(train_vec.loc[shuffled_order[batch_idx], :])
                expnx = np.exp(-x)
                linkx = np.divide(1, (1+expnx))
                y = train_vec.loc[shuffled_order[batch_idx], 'correct'].values
                gradlogloss = -  np.multiply(y, 1-linkx) + np.multiply(1-y, linkx)

                if self.global_bias:
                    dw_beta_global = np.zeros(1)
                    dw_beta_global = dw_beta_global + 2 * np.sum(gradlogloss)
                    self.beta_global_inc = self.momentum * self.beta_global_inc + self.epsilon * dw_beta_global / self.batch_size
                    self.beta_global = self.beta_global - self.beta_global_inc

                if self.user:
                    dw_beta_user = np.zeros(num_user)
                    batch_userid = np.array(train_vec.loc[shuffled_order[batch_idx], 'user_id'], dtype='int32')
                    beta_user_grad = 2 * gradlogloss + self._lambda * self.beta_user[batch_userid]
                    for i in range(self.batch_size):
                        dw_beta_user[batch_userid[i]]  += beta_user_grad[i]
                    self.beta_user_inc = self.momentum * self.beta_user_inc + self.epsilon * dw_beta_user / self.batch_size
                    self.beta_user = self.beta_user - self.beta_user_inc

                if self.problem:
                    dw_beta_prob = np.zeros(num_prob)
                    batch_probid = np.array(train_vec.loc[shuffled_order[batch_idx], 'problem_id'], dtype='int32')
                    beta_prob_grad = 2 * gradlogloss + self._lambda * self.beta_prob[batch_probid]
                    for i in range(self.batch_size):
                        dw_beta_prob[batch_probid[i]]  += beta_prob_grad[i]
                    self.beta_prob_inc = self.momentum * self.beta_prob_inc + self.epsilon * dw_beta_prob / self.batch_size
                    self.beta_prob = self.beta_prob - self.beta_prob_inc

                if self.MF:
                    Ix_user = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_prob[batch_probid, :]) \
                       + self._lambda * self.w_user[batch_userid, :]
                    Ix_prob = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_user[batch_userid, :]) \
                       + self._lambda * (self.w_prob[batch_probid, :])    # np.newaxis :increase the dimension
                    dw_prob = np.zeros((num_prob, self.num_feat))
                    dw_user = np.zeros((num_user, self.num_feat))
                    for i in range(self.batch_size):
                        dw_prob[batch_probid[i], :] = dw_prob[batch_probid[i], :] + Ix_prob[i, :]
                        dw_user[batch_userid[i], :] = dw_user[batch_userid[i], :] + Ix_user[i, :]
                    self.w_user_inc = self.momentum * self.w_user_inc + self.epsilon * dw_user / self.batch_size
                    self.w_prob_inc = self.momentum * self.w_prob_inc + self.epsilon * dw_prob / self.batch_size
                    self.w_prob = self.w_prob - self.w_prob_inc
                    self.w_user = self.w_user - self.w_user_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    logloss, auc, acc = self.calc_loss(train_vec, train_vec.loc[:, 'correct'].values)
                    self.auc_train.append(auc)
                    self.acc_train.append(acc)
                    obj = logloss
                    self.logloss_train.append((obj / train_vec.shape[0]))

                # Compute validation error
                if batch == self.num_batches - 1:
                    logloss, auc, acc = self.calc_loss(test_vec, test_vec.loc[:, 'correct'].values)
                    self.auc_test.append(auc)
                    self.acc_test.append(acc)
                    self.logloss_test.append(logloss/pairs_test)
                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training logloss: %f, Train ACC %f, Train AUC %f, Test logloss %f, Test ACC %f, Test AUC %f' \
                              % (self.logloss_train[-1], self.acc_train[-1], self.auc_train[-1], self.logloss_test[-1], self.acc_test[-1], self.auc_test[-1]) )



    def calc_loss(self, X, y):
        x = self.CalcPrediction(X)
        expnx = np.exp(-x)
        linkx = np.divide(1, (1+expnx))
        logx = np.log(linkx)
        lognx = np.log(1-linkx)
        logloss = np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx))
        auc = roc_auc_score(np.array(y, dtype=bool), linkx)
        pred = linkx
        pred[linkx<0.5] = 0
        pred[linkx>=0.5] = 1
        acc = accuracy_score(np.array(y, dtype=bool), np.array(pred, dtype=bool))
        return logloss, auc, acc



    def CalcPrediction(self, data):
        x = np.zeros(data.shape[0])
        if self.global_bias:
            x = x + self.beta_global
        if self.user or self.MF:
            batch_userid = np.array(data.loc[:, 'user_id'], dtype='int32')
            if self.user:
                x = x + self.beta_user[batch_userid]
        if self.problem or self.MF:
            batch_probid = np.array(data.loc[:, 'problem_id'], dtype='int32')
            if self.problem:
                x = x + self.beta_prob[batch_probid]
        if self.MF:
            x = x + np.sum(np.multiply(self.w_user[batch_userid, :], self.w_prob[batch_probid, :]), axis=1)
        if self.problem_dyn_embedding:
            print('Not implemented yet!')

        return x


    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)
            self.MF = parameters.get('MF', False)
            self.num_feat = parameters.get('num_feat', 5)
            self.problem = parameters.get('problem', False)
            self.user = parameters.get('user', True)
            self.problem_dyn_embedding = parameters.get('problem_dyn_embedding', False)

