#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:12:06 2019

@author: lxu
"""

import numpy as np
import sklearn as sklearn
import itertools

class IRT(object):
    def __init__(self, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=300, batch_size=1000, \
                 problem=False, multi_skills=False, user_skill=False, user_prob=False, PFA=False, MF=False, num_feat=5, MF_skill=True):

        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
        self.multi_skills = multi_skills
        self.user_skill = user_skill
        self.user_prob = user_prob
        self.PFA = PFA
        self.MF  = MF
        self.num_feat = num_feat
        self.problem = problem
        self.MF_skill = MF_skill


        self.beta_skill = None
        self.beta_prob = None
        self.beta_user = None
        self.beta_global = None

        self.gamma = None  #  success
        self.rho = None  # failure

        self.w_prob = None  # problem feature vectors
        self.w_user = None  # user feature vectors
        self.w_skill = None  # skill feature vectors

        self.alpha_user = None
        self.alpha_prob = None
        self.alpha_skill = None

        self.beta_skill_inc = None
        self.beta_prob_inc = None
        self.beta_user_inc = None
        self.beta_global_inc = None

        self.gamma_inc = None
        self.rho_inc = None

        self.w_prob_inc = None  # problem feature vectors
        self.w_user_inc = None  # user feature vectors
        self.w_skill_inc = None  # skill feature vectors

        self.logloss_train = []
        self.logloss_test = []
        self.auc_train = []
        self.auc_test = []
        self.baseline_auc_test= []

        self.classification_train = []
        self.classification_test = []


    def fit(self, train_vec, test_vec,  num_user, num_skill, num_prob):
        pairs_train = train_vec.shape[0]
        pairs_test = test_vec.shape[0]

        # average scores
        self.mean_inv = np.mean(train_vec['correct'])
        self.epoch = 0
        # initialization
        self.beta_skill =  0.1 * np.random.randn(num_skill)
        self.beta_prob = 0.1 * np.random.randn(num_prob)
        self.beta_user = 0.1 * np.random.randn(num_user)
        self.beta_global = 0.1 * np.random.randn(1)

        self.alpha_skill =  0.1 * np.random.randn(num_skill)
        self.alpha_prob = 0.1 * np.random.randn(num_prob)
        self.alpha_user = 0.1 * np.random.randn(num_user)

        self.gamma = 0.0 * np.random.randn(num_skill)
        self.rho   = 0.0 * np.random.randn(num_skill)

        self.w_prob = 0.1 * np.random.rand(num_prob, self.num_feat)  # problem latent matrix
        self.w_user = 0.1 * np.random.randn(num_user, self.num_feat)  # user latent matrix
        self.w_skill = 0.1 * np.random.randn(num_skill, self.num_feat)  # user latent matrix
        
        self.beta_skill_inc = np.zeros(num_skill)
        self.beta_prob_inc = np.zeros(num_prob)
        self.beta_user_inc = np.zeros(num_user)
        self.beta_global_inc = np.zeros(1)

        self.alpha_skill_inc =  0.1 * np.random.randn(num_skill)
        self.alpha_prob_inc = 0.1 * np.random.randn(num_prob)
        self.alpha_user_inc = 0.1 * np.random.randn(num_user)

        self.gamma_inc = np.zeros(num_skill)
        self.rho_inc = np.zeros(num_skill)

        self.w_prob_inc = np.zeros((num_prob, self.num_feat))
        self.w_user_inc = np.zeros((num_user, self.num_feat))
        self.w_skill_inc = np.zeros((num_skill, self.num_feat))

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
                gradlogloss = -  np.multiply(y,1-linkx) + np.multiply(1-y, linkx)

                # compute the gradient
                dw_beta_skill = np.zeros(num_skill)
                dw_beta_prob = np.zeros(num_prob)
                dw_beta_user = np.zeros(num_user)
                dw_beta_global = 2 * np.sum(gradlogloss) + self._lambda * self.beta_global

                dw_gamma = np.zeros(num_skill)
                dw_rho = np.zeros(num_skill)

                dw_alpha_skill = np.zeros(num_skill)
                dw_alpha_prob = np.zeros(num_prob)
                dw_alpha_user = np.zeros(num_user)

                if self.PFA:
                    s_counts = train_vec.loc[shuffled_order[batch_idx], 'sCount'].values
                    f_counts = train_vec.loc[shuffled_order[batch_idx], 'fCount'].values
                batch_userID = np.array(train_vec.loc[shuffled_order[batch_idx], 'user_id'], dtype='int32')

                if self.multi_skills:
                    batch_skillIDs = train_vec.loc[shuffled_order[batch_idx], 'skill_ids'].values
                else:
                    batch_skillID = train_vec.loc[shuffled_order[batch_idx], 'skill_id'].values

                if self.MF_skill:
                    dw_skill = np.zeros((num_skill, self.num_feat))
                    dw_user = np.zeros((num_user, self.num_feat))
                    if self.multi_skills:
                        for i in range(self.batch_size):
                            for idx, skill in enumerate(batch_skillIDs[i]):
                                dw_skill[skill, :] += 2 * np.multiply(gradlogloss[i, np.newaxis], self.w_user[batch_userID[i], :]) \
                                    + self._lambda * (self.w_skill[skill, :])
                                dw_user[batch_userID[i], :] += 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_skill[skill, :]) \
                                    + self._lambda * self.w_user[batch_userID[i], :]
                    else:
                        Ix_user = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_skill[batch_skillID, :]) \
                           + self._lambda * self.w_user[batch_userID, :]
                        Ix_skill = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_user[batch_userID, :]) \
                           + self._lambda * (self.w_skill[batch_skillID, :])
                        for i in range(self.batch_size):
                            dw_skill[batch_skillID[i], :] = dw_skill[batch_skillID[i], :] + Ix_skill[i, :]
                            dw_user[batch_userID[i], :] = dw_user[batch_userID[i], :] + Ix_user[i, :]

                if self.multi_skills:
                    for i in range(self.batch_size):
                        for idx, skill in enumerate(batch_skillIDs[i]):
                            dw_beta_skill[skill]  += 2 * gradlogloss[i] / len(batch_skillIDs[i]) + self._lambda * self.beta_skill[skill]
                            if self.PFA:
                                dw_gamma[skill] += 0.2 * gradlogloss[i] * s_counts[i][idx] + self._lambda * self.gamma[skill]
                                dw_rho[skill]   += 0.2 * gradlogloss[i] * f_counts[i][idx] + self._lambda * self.rho[skill]
                            if self.user_skill:
                                dw_alpha_skill[skill]  += 2 * gradlogloss[i] * self.alpha_user[batch_userID[i]] / len(batch_skillIDs[i]) + self._lambda * self.alpha_skill[skill]
                                dw_alpha_user[batch_userID[i]] +=  2 * gradlogloss[i] * self.alpha_skill[skill] / len(batch_skillIDs[i]) + self._lambda * self.alpha_user[batch_userID[i]]


                else:
                    beta_skill_grad  = 2 * gradlogloss + self._lambda * self.beta_skill[batch_skillID]

                    if self.PFA:
                        gamma_grad = 0.2 * np.multiply(gradlogloss, train_vec.loc[shuffled_order[batch_idx], 'sCount'].values) + \
                            self._lambda * self.gamma[batch_skillID]
                        rho_grad   = 0.2 * np.multiply(gradlogloss, train_vec.loc[shuffled_order[batch_idx], 'fCount'].values) + \
                            self._lambda * self.rho[batch_skillID]
                    if self.user_skill:
                        alpha_user_grad = 2 * gradlogloss * self.alpha_skill[batch_skillID] + self._lambda * self.alpha_user[batch_userID]
                        alpha_skill_grad = 2 * gradlogloss * self.alpha_user[batch_userID] + self._lambda * self.alpha_skill[batch_skillID]

                    # loop to aggreate the gradients of the same element
                    for i in range(self.batch_size):
                        dw_beta_skill[batch_skillID[i]]  += beta_skill_grad[i]
                        if self.user_skill:
                            dw_alpha_skill[batch_skillID[i]]  += alpha_skill_grad[i]
                            dw_alpha_user[batch_userID[i]]  += alpha_user_grad[i]

                        if self.PFA:
                            dw_gamma[batch_skillID[i]] += gamma_grad[i]
                            dw_rho[batch_skillID[i]]   += rho_grad[i]


                beta_user_grad = 2 * gradlogloss + self._lambda * self.beta_user[batch_userID]
                for i in range(self.batch_size):
                    dw_beta_user[batch_userID[i]]  += beta_user_grad[i]

                if self.problem:
                    batch_probID = np.array(train_vec.loc[shuffled_order[batch_idx], 'problem_id'], dtype='int32')
                    beta_prob_grad = 2 * gradlogloss + self._lambda * self.beta_prob[batch_probID]
                    for i in range(self.batch_size):
                        dw_beta_prob[batch_probID[i]]  += beta_prob_grad[i]
                if self.MF:
                    Ix_user = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_prob[batch_probID, :]) \
                       + self._lambda * self.w_user[batch_userID, :]

                    Ix_prob = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_user[batch_userID, :]) \
                       + self._lambda * (self.w_prob[batch_probID, :])    # np.newaxis :increase the dimension
                    dw_prob = np.zeros((num_prob, self.num_feat))
                    if not self.MF_skill:
                        dw_user = np.zeros((num_user, self.num_feat))
                    for i in range(self.batch_size):
                        dw_prob[batch_probID[i], :] = dw_prob[batch_probID[i], :] + Ix_prob[i, :]
                        dw_user[batch_userID[i], :] = dw_user[batch_userID[i], :] + Ix_user[i, :]


                if self.user_prob:
                    for i in range(self.batch_size):
                        dw_alpha_prob[batch_probID[i]]  += 2 * gradlogloss[i] * self.alpha_user[batch_userID[i]] + self._lambda * self.alpha_prob[batch_probID[i]]
                        dw_alpha_user[batch_userID[i]] +=  2 * gradlogloss[i] * self.alpha_prob[batch_probID[i]] + self._lambda * self.alpha_user(batch_userID[i])

                # Update with momentum
                self.beta_skill_inc  = self.momentum * self.beta_skill_inc  + self.epsilon * dw_beta_skill  / self.batch_size

                self.beta_user_inc = self.momentum * self.beta_user_inc + self.epsilon * dw_beta_user / self.batch_size
                self.beta_global_inc = self.momentum * self.beta_global_inc + self.epsilon * dw_beta_global / self.batch_size
                if self.PFA:
                    self.gamma_inc = self.momentum * self.gamma_inc + self.epsilon * dw_gamma / self.batch_size
                    self.rho_inc   = self.momentum * self.rho_inc   + self.epsilon * dw_rho   / self.batch_size

                if self.MF or self.MF_skill:
                    self.w_user_inc = self.momentum * self.w_user_inc + self.epsilon * dw_user / self.batch_size
                    if self.MF:
                        self.w_prob_inc = self.momentum * self.w_prob_inc + self.epsilon * dw_prob / self.batch_size
                    if self.MF_skill:
                        self.w_skill_inc = self.momentum * self.w_skill_inc + self.epsilon * dw_skill / self.batch_size

                if self.user_prob or self.user_skill:
                    self.alpha_user_inc = self.momentum * self.alpha_user_inc + self.epsilon * dw_alpha_user / self.batch_size
                    if self.user_prob:
                        self.alpha_prob_inc = self.momentum * self.alpha_prob_inc + self.epsilon * dw_alpha_prob / self.batch_size
                    if self.user_skill:
                        self.alpha_skill_inc = self.momentum * self.alpha_skill_inc + self.epsilon * dw_alpha_skill / self.batch_size

                if self.problem:
                    self.beta_prob_inc = self.momentum * self.beta_prob_inc + self.epsilon * dw_beta_prob / self.batch_size

                # gradien descent
                self.beta_skill  = self.beta_skill  - self.beta_skill_inc

                self.beta_user = self.beta_user - self.beta_user_inc
                self.beta_global = self.beta_global - self.beta_global_inc
                if self.PFA:
                    self.gamma = self.gamma - self.gamma_inc
                    self.rho   = self.rho   - self.rho_inc
                if self.MF:
                    self.w_user = self.w_user - self.w_user_inc
                    if self.MF:
                        self.w_prob = self.w_prob - self.w_prob_inc
                    if self.MF_skill:
                        self.w_skill = self.w_skill - self.w_skill_inc

                if self.user_skill or  self.user_prob:
                    self.alpha_user = self.alpha_user - self.alpha_user_inc
                    if self.user_skill:
                        self.alpha_skill = self.alpha_skill - self.alpha_skill_inc
                    if self.user_prob:
                        self.alpha_prob = self.alpha_prob - self.alpha_prob_inc
                if self.problem:
                    self.beta_prob = self.beta_prob - self.beta_prob_inc
                # select models
                # self.beta_skill.fill(0)
                # self.beta_prob.fill(0)
                # self.beta_user.fill(0)
                # self.beta_global.fill(0)

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    x = self.CalcPrediction(train_vec)
                    expnx = np.exp(-x)
                    linkx = np.divide(1, (1+expnx))
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)

                    y = train_vec.loc[:, 'correct'].values
                    logloss =  np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx))
                    auc = sklearn.metrics.roc_auc_score(np.array(y, dtype=bool), linkx)
                    self.auc_train.append(auc)
                    obj = logloss + 0.5 * self._lambda * ( np.linalg.norm(self.beta_skill) ** 2 + np.linalg.norm(self.beta_prob) ** 2 + \
                         np.linalg.norm(self.beta_user) ** 2) + self.beta_global ** 2

                    self.logloss_train.append((obj / train_vec.shape[0]))

                # Compute validation error
                if batch == self.num_batches - 1:
                    x = self.CalcPrediction(test_vec)
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
                        print('Training logloss: %f, Test logloss %f, Train AUC %f, Test AUC %f' \
                              % (self.logloss_train[-1], self.logloss_test[-1], self.auc_train[-1], self.auc_test[-1]) )

    def CalcPrediction(self, data):
        batch_userID = np.array(data.loc[:, 'user_id'], dtype='int32')

        if self.PFA:
            s_counts = data.loc[:, 'sCount'].values
            f_counts = data.loc[:, 'fCount'].values
            x1 = np.zeros(data.shape[0])
            x2 = np.zeros(data.shape[0])

        if self.multi_skills:
            batch_skillIDs = data.loc[:, 'skill_ids'].values
            x3 = np.zeros(data.shape[0])
            x4 = np.zeros(data.shape[0])
            for i in range(data.shape[0]):
                if self.PFA:
                    x1[i] = np.sum(np.multiply(self.gamma[batch_skillIDs[i]], s_counts[i]))
                    x2[i] = np.sum(np.multiply(self.rho[batch_skillIDs[i]], f_counts[i]))
                x3[i] = np.sum(self.beta_skill[batch_skillIDs[i]]) / len(batch_skillIDs[i])
                x4[i] = np.sum(self.alpha_skill[batch_skillIDs[i]]) / len(batch_skillIDs[i])
            x = self.beta_user[batch_userID] + x3  + self.beta_global
            if self.user_skill:
                x = x + np.multiply(x4, self.alpha_user[batch_userID] )
        else:
            batch_skillID = np.array(data.loc[:, 'skill_id'], dtype='int32')
            if self.PFA:
                x1 = np.multiply(data.loc[:, 'sCount'].values, self.gamma[batch_skillID])
                x2 = np.multiply(data.loc[:, 'fCount'].values, self.rho[batch_skillID])
            x = self.beta_user[batch_userID] + self.beta_skill[batch_skillID]  + self.beta_global
            if self.user_skill:
                x = x + np.multiply(self.alpha_skill[batch_skillID], self.alpha_user[batch_userID])
        if self.problem:
            batch_probID = np.array(data.loc[:, 'problem_id'], dtype='int32')
            x = x + self.beta_prob[batch_probID]
        if self.user_prob:
            x = x + np.multiply(self.alpha_prob[batch_probID], self.alpha_user[batch_userID])
        if self.PFA:
            x = x + x1 + x2
        if self.MF:
            new_w_user = self.w_user[batch_userID, :]
            x = x + np.sum(np.multiply(new_w_user, self.w_prob[batch_probID, :]), axis=1)
        if self.MF_skill:
            if self.multi_skills:
                w_skill = np.zeros((data.shape[0], self.num_feat))
                batch_skillIDs = data.loc[:, 'skill_ids'].values
                for i in range(data.shape[0]):
                    w_skill[i, :] = np.sum(self.w_skill[batch_skillIDs[i],:], axis=0) / len(batch_skillIDs[i])
                x = x + np.sum(np.multiply(w_skill, self.w_user[batch_userID, :]), axis=1)
            else:
                x = x + np.sum(np.multiply(self.w_skill[batch_skillID, :], self.w_user[batch_userID, :]), axis=1)
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
            self.multi_skills = parameters.get('multi_skills', False)
            self.user_skill = parameters.get('user_skill', False)
            self.user_prob = parameters.get('user_prob', False)
            self.PFA = parameters.get('PFA', False)
            self.MF = parameters.get('MF', False)
            self.num_feat = parameters.get('num_feat', 5)
            self.problem = parameters.get('problem', False)
            self.MF_skill = parameters.get('MF_skill', False)
        if self.MF:
            self.problem = True
        if self.user_prob:
            self.problem = True
