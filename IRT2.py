#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:12:06 2019

@author: lxu
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class IRT2(object):
    def __init__(self, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=300, batch_size=1000,
                    problem=True, MF_prob=True, num_feat=20, user=True,  global_bias=True,
                    problem_dyn_embedding=False, patience = 5, skill=False, MF_skill=False, PFA=False):

        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
        self.MF_prob = MF_prob
        self.num_feat = num_feat
        self.problem = problem
        self.user = user
        self.problem_dyn_embedding = problem_dyn_embedding
        self.global_bias = global_bias
        self.patience = patience
        self.skill = skill
        self.MF_skill = MF_skill
        self.PFA = PFA

        self.beta_prob = None
        self.beta_user = None
        self.beta_global = None
        self.beta_skill = None

        self.gamma = None  # success
        self.rho = None  # failure

        # TODO: use different feature vectors
        self.w_prob = None  # problem feature vectors
        self.w_user = None  # user feature vectors
        self.w_skill = None # skill feature vectors

        self.h_user = None # dynamic, user learning ability
        self.h_prob = None # dynamic, skill teaching ability

        self.beta_prob_inc = None
        self.beta_user_inc = None
        self.beta_global_inc = None
        self.beta_skill_inc = None

        self.gamma_inc = None
        self.rho_inc = None

        self.w_prob_inc = None  # problem feature vectors
        self.w_user_inc = None  # user feature vectors
        self.w_skill_inc = None

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

    def fit(self, train_vec, test_vec,  num_user, num_prob, num_skill):
        pairs_train = train_vec.shape[0]
        pairs_test = test_vec.shape[0]
        if self.problem_dyn_embedding:
            train_order = train_vec['hist'].values
            test_order = test_vec['hist'].values


        # average scores
        self.epoch = 0

        # initialization
        self.beta_prob = 0.1 * np.random.randn(num_prob)
        self.beta_user = 0.1 * np.random.randn(num_user)
        self.beta_skill = 0.1 * np.random.randn(num_skill)
        self.beta_global = 0.1 * np.random.randn(1)
        self.w_prob = 0.1 * np.random.rand(num_prob, self.num_feat)  # problem latent matrix
        self.w_user = 0.1 * np.random.randn(num_user, self.num_feat)  # user latent matrix
        self.w_skill = 0.1 * np.random.rand(num_skill, self.num_feat)
        self.h_prob = 0.01 * np.random.randn(num_prob)
        self.h_user = 0.01 * np.random.randn(num_user)
        self.gamma = 0.1 * np.random.randn(num_skill)
        self.rho = 0.1 *np.random.randn(num_skill)

        self.beta_prob_inc = np.zeros(num_prob)
        self.beta_user_inc = np.zeros(num_user)
        self.beta_skill_inc = np.zeros(num_skill)
        self.beta_global_inc = np.zeros(1)
        self.w_prob_inc = np.zeros((num_prob, self.num_feat))
        self.w_user_inc = np.zeros((num_user, self.num_feat))
        self.w_skill_inc = np.zeros((num_skill, self.num_feat))
        self.h_prob_inc = 0 * np.random.randn(num_prob)
        self.h_user_inc = 0 * np.random.randn(num_user)
        self.gamma_inc = 0 * np.random.randn(num_skill)
        self.rho_inc = 0 *np.random.randn(num_skill)

        self.dyn_flag = False
        best_metric = []
        patience_cnt = 0
        stop_flag = False
        while self.epoch < self.maxepoch:
            self.epoch += 1
            # shuffle training tuples
            shuffled_order = np.arange(pairs_train)
            np.random.shuffle(shuffled_order)
            if stop_flag:
                break
            # batch update
            for batch in range(self.num_batches):
                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # index used in this batch

                # compute gradient of obj
                if self.dyn_flag:
                    x = self.calc_prediction(train_vec.loc[shuffled_order[batch_idx], :],
                                             train_order[shuffled_order[batch_idx]])
                else:
                    x = self.calc_prediction(train_vec.loc[shuffled_order[batch_idx], :],
                                             None)
                expnx = np.exp(-x)
                linkx = np.divide(1, (1+expnx))
                y = train_vec.loc[shuffled_order[batch_idx], 'correct'].values
                gradlogloss = -  np.multiply(y, 1-linkx) + np.multiply(1-y, linkx)
                if 1:
                    if self.global_bias:
                        dw_beta_global = np.zeros(1)
                        dw_beta_global = dw_beta_global + 2 * np.sum(gradlogloss)
                        self.beta_global_inc = self.momentum * self.beta_global_inc + self.epsilon * dw_beta_global / self.batch_size
                        self.beta_global = self.beta_global - self.beta_global_inc

                    if self.user:
                        dw_beta_user = np.zeros(num_user)
                        batch_user_id = np.array(train_vec.loc[shuffled_order[batch_idx], 'user_id'], dtype='int32')
                        beta_user_grad = 2 * gradlogloss + self._lambda * self.beta_user[batch_user_id]
                        for i in range(self.batch_size):
                            dw_beta_user[batch_user_id[i]] += beta_user_grad[i]
                        self.beta_user_inc = self.momentum * self.beta_user_inc + self.epsilon * dw_beta_user / self.batch_size
                        self.beta_user = self.beta_user - self.beta_user_inc

                    if self.problem:
                        dw_beta_prob = np.zeros(num_prob)
                        batch_prob_id = np.array(train_vec.loc[shuffled_order[batch_idx], 'problem_id'], dtype='int32')
                        beta_prob_grad = 2 * gradlogloss + self._lambda * self.beta_prob[batch_prob_id]
                        for i in range(self.batch_size):
                            dw_beta_prob[batch_prob_id[i]]  += beta_prob_grad[i]
                        self.beta_prob_inc = self.momentum * self.beta_prob_inc + self.epsilon * dw_beta_prob / self.batch_size
                        self.beta_prob = self.beta_prob - self.beta_prob_inc

                    if self.skill:
                        dw_beta_skill = np.zeros(num_skill)
                        batch_skill_id = np.array(train_vec.loc[shuffled_order[batch_idx], 'skill_id'], dtype='int32')
                        beta_skill_grad = 2 * gradlogloss + self._lambda * self.beta_skill[batch_skill_id]
                        for i in range(self.batch_size):
                            dw_beta_skill[batch_skill_id[i]]  += beta_skill_grad[i]
                        self.beta_skill_inc = self.momentum * self.beta_skill_inc + self.epsilon * dw_beta_skill / self.batch_size
                        self.beta_skill = self.beta_skill - self.beta_skill_inc

                    if self.MF_skill:
                        Ix_user = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_skill[batch_skill_id, :]) \
                           + self._lambda * self.w_user[batch_user_id, :]
                        Ix_skill = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_user[batch_user_id, :]) \
                           + self._lambda * (self.w_skill[batch_skill_id, :])    # np.newaxis :increase the dimension
                        dw_skill = np.zeros((num_skill, self.num_feat))
                        dw_user = np.zeros((num_user, self.num_feat))
                        for i in range(self.batch_size):
                            dw_skill[batch_skill_id[i], :] = dw_skill[batch_skill_id[i], :] + Ix_skill[i, :]
                            dw_user[batch_user_id[i], :] = dw_user[batch_user_id[i], :] + Ix_user[i, :]
                        self.w_user_inc = self.momentum * self.w_user_inc + self.epsilon * dw_user / self.batch_size
                        self.w_skill_inc = self.momentum * self.w_skill_inc + self.epsilon * dw_skill / self.batch_size
                        self.w_skill = self.w_skill - self.w_skill_inc
                        self.w_user = self.w_user - self.w_user_inc

                    if self.MF_prob:
                        Ix_user = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_prob[batch_prob_id, :]) \
                           + self._lambda * self.w_user[batch_user_id, :]
                        Ix_prob = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_user[batch_user_id, :]) \
                           + self._lambda * (self.w_prob[batch_prob_id, :])    # np.newaxis :increase the dimension
                        dw_prob = np.zeros((num_prob, self.num_feat))
                        dw_user = np.zeros((num_user, self.num_feat))
                        for i in range(self.batch_size):
                            dw_prob[batch_prob_id[i], :] = dw_prob[batch_prob_id[i], :] + Ix_prob[i, :]
                            dw_user[batch_user_id[i], :] = dw_user[batch_user_id[i], :] + Ix_user[i, :]
                        self.w_user_inc = self.momentum * self.w_user_inc + self.epsilon * dw_user / self.batch_size
                        self.w_prob_inc = self.momentum * self.w_prob_inc + self.epsilon * dw_prob / self.batch_size
                        self.w_prob = self.w_prob - self.w_prob_inc
                        self.w_user = self.w_user - self.w_user_inc

                    if self.PFA:
                        s_counts = train_vec.loc[shuffled_order[batch_idx], 'sCount'].values
                        f_counts = train_vec.loc[shuffled_order[batch_idx], 'fCount'].values
                        batch_skill_id = np.array(train_vec.loc[shuffled_order[batch_idx], 'skill_id'], dtype='int32')

                        gamma_grad = 0.2 * np.multiply(gradlogloss, s_counts)
                            # + self._lambda * self.gamma[batch_skill_id]
                        rho_grad   = 0.2 * np.multiply(gradlogloss, f_counts)
                            # + self._lambda * self.rho[batch_skill_id]
                        dw_gamma = np.zeros(num_skill)
                        dw_rho = np.zeros(num_skill)
                        for i in range(self.batch_size):
                            dw_gamma[batch_skill_id[i]] += gamma_grad[i]
                            dw_rho[batch_skill_id[i]]   += rho_grad[i]
                        self.gamma_inc = self.momentum * self.gamma_inc + self.epsilon * dw_gamma / self.batch_size
                        self.rho_inc   = self.momentum * self.rho_inc   + self.epsilon * dw_rho   / self.batch_size
                        self.gamma = self.gamma - self.gamma_inc
                        self.rho   = self.rho   - self.rho_inc
                # dynamic model tuning
                if self.dyn_flag:

                    if 0:
                        Ix_user = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_prob[batch_prob_id, :]) \
                           + self._lambda * self.w_user[batch_user_id, :]

                        dw_user = np.zeros((num_user, self.num_feat))
                        for i in range(self.batch_size):
                            dw_user[batch_user_id[i], :] = dw_user[batch_user_id[i], :] + Ix_user[i, :]
                        self.w_user_inc = self.momentum * self.w_user_inc + self.epsilon * dw_user / self.batch_size

                        self.w_user = self.w_user - self.w_user_inc

                    for i in range(self.batch_size):
                        dw_h_prob= np.zeros(num_prob)
                        dw_h_user = np.zeros(num_user)
                        hist = train_order[shuffled_order[i]]
                        user_id = train_vec.loc[:, 'user_id'].values[shuffled_order[i]]
                        if len(hist) == 0:
                            continue
                        if len(hist) > 10:
                            hist = hist[0:9]
                        dw_h_prob[hist] += 2 * gradlogloss[i] * np.matmul(self.w_prob[hist, :],
                            np.transpose(self.w_prob[[batch_prob_id[i]], :])).flatten() * self.h_user[user_id] + self._lambda * self.h_prob[hist]
                        dw_h_user[user_id] += 2 * gradlogloss[i] * np.sum(np.matmul(self.w_prob[hist, :],
                            np.transpose(self.w_prob[[batch_prob_id[i]], :])).flatten() * self.h_prob[hist]) + self._lambda * self.h_user[user_id]
                    self.h_prob_inc = self.momentum * self.h_prob_inc + self.epsilon * dw_h_prob / self.batch_size
                    self.h_user_inc = self.momentum * self.h_user_inc + self.epsilon * dw_h_user / self.batch_size
                    # print('Change', np.linalg.norm(self.h_user_inc))
                    self.h_prob = self.h_prob - self.h_prob_inc
                    self.h_user = self.h_user - self.h_user_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    if self.dyn_flag:
                        logloss, auc, acc = self.calc_loss(train_vec, train_order, train_vec.loc[:, 'correct'].values)
                    else:
                        logloss, auc, acc = self.calc_loss(train_vec, None, train_vec.loc[:, 'correct'].values)

                    self.auc_train.append(auc)
                    self.acc_train.append(acc)
                    obj = logloss
                    self.logloss_train.append((obj / train_vec.shape[0]))

                # Compute validation error
                if batch == self.num_batches - 1:
                    if self.dyn_flag:
                        logloss, auc, acc = self.calc_loss(test_vec, test_order, test_vec.loc[:, 'correct'].values)
                    else:
                        logloss, auc, acc = self.calc_loss(test_vec, None, test_vec.loc[:, 'correct'].values)

                    self.auc_test.append(auc)
                    self.acc_test.append(acc)
                    self.logloss_test.append(logloss/pairs_test)

                    if not best_metric:
                        best_metric = auc
                    else:
                        if best_metric > auc:
                            patience_cnt += 1
                        else:
                            patience_cnt = 0
                            best_metric = auc

                    # print('Patient:', patience_cnt)
                    print('Training logloss: %f, Train ACC %f, Train AUC %f, Test logloss %f, Test ACC %f, Test AUC %f. Best so far %f.' \
                          % (self.logloss_train[-1], self.acc_train[-1], self.auc_train[-1], self.logloss_test[-1], self.acc_test[-1], self.auc_test[-1], best_metric))

                    if (patience_cnt >= self.patience) or (self.epoch == self.maxepoch):
                        if self.problem_dyn_embedding and (not self.dyn_flag):
                            print('Tune dynamic model...')
                            self.dyn_flag = True
                            best_metric = []
                            self.epoch = 0
                            patience_cnt = 0
                            self.maxepoch = 20
                        else:
                            stop_flag = True

    def calc_loss(self, X, order, y):
        x = self.calc_prediction(X, order)
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

    def calc_prediction(self, data, order):
        x = np.zeros(data.shape[0])
        if self.global_bias:
            x = x + self.beta_global
        if self.user or self.MF_prob:
            batch_user_id = np.array(data.loc[:, 'user_id'], dtype='int32')
        if self.user:
            x = x + self.beta_user[batch_user_id]
        if self.problem or self.MF_prob:
            batch_prob_id = np.array(data.loc[:, 'problem_id'], dtype='int32')
        if self.problem:
            x = x + self.beta_prob[batch_prob_id]
        if self.MF_prob:
            x = x + np.sum(np.multiply(self.w_user[batch_user_id, :], self.w_prob[batch_prob_id, :]), axis=1)
        if self.skill or self.MF_skill or self.PFA:
            batch_skill_id = np.array(data.loc[:, 'skill_id'], dtype='int32')
        if self.skill:
            x = x + self.beta_skill[batch_skill_id]
        if self.MF_skill:
            x = x + np.sum(np.multiply(self.w_user[batch_user_id, :], self.w_skill[batch_skill_id, :]), axis=1)
        if self.PFA:
            x1 = np.multiply(data.loc[:, 'sCount'].values, self.gamma[batch_skill_id])
            x2 = np.multiply(data.loc[:, 'fCount'].values, self.rho[batch_skill_id])
            x = x + x1 + x2
        if self.dyn_flag:
            sum_prob = np.zeros((data.shape[0], self.num_feat))
            for i in range(self.batch_size):
                hist = order[i]
                user_id = data.loc[:, 'user_id'].values[i]
                if len(hist) == 0:
                    continue
                if len(hist) > 10:
                    hist = hist[0:9]
                temp = np.outer(self.h_prob[hist] * self.h_user[user_id], np.ones(self.num_feat))
                sum_prob[i, :] = np.sum(np.multiply(temp, self.w_prob[hist, :]), axis=0)
            x = x + np.sum(np.multiply(sum_prob, self.w_prob[batch_prob_id, :]), axis=1)

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
            self.MF_prob = parameters.get('MF_prob', False)
            self.num_feat = parameters.get('num_feat', 5)
            self.problem = parameters.get('problem', False)
            self.user = parameters.get('user', True)
            self.skill = parameters.get('skill', False)
            self.MF_skill = parameters.get('MF_skill', False)
            self.PFA = parameters.get('PFA', False)
            self.problem_dyn_embedding = parameters.get('problem_dyn_embedding', False)

