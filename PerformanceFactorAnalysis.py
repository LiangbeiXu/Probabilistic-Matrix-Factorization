#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:12:06 2019

@author: lxu
"""

import numpy as np
import sklearn as sklearn
import itertools

class PFA(object):
    def __init__(self, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=300, batch_size=1000, dynamic=True):
        
        

        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
        self.dynamic = dynamic

        self.beta = None  # I
        self.gamma = None  # 
        self.rho = None  #
        self.alpha = None
        self.beta_inc = None
        self.gamma_inc = None
        self.rho_inc = None
        self.alpha_inc = None

        self.logloss_train = []
        self.logloss_test = []
        self.auc_train = []
        self.auc_test = []
        self.baseline_auc_test= []
        
        self.classification_train = []
        self.classification_test = []
        
        
    def fit_single_skill(self, train_vec, test_vec,  num_user, num_skill, num_problem, multi_skills):
        pairs_train = train_vec.shape[0]  
        pairs_test = test_vec.shape[0]  
        
        # average scores
        self.mean_inv = np.mean(train_vec['correct'])
        self.epoch = 0
        # initialization
        self.beta =  0.1 * np.random.randn(num_skill)
        self.beta2 = 0.1 * np.random.randn(num_problem)
        self.gamma = 0.1 * np.random.randn(num_skill)
        self.rho   = 0.1 * np.random.randn(num_skill)
        self.alpha = 0.1 * np.random.randn(num_user)
        
        self.beta_inc = np.zeros(num_skill)
        self.beta2_inc = np.zeros(num_problem)
        self.gamma_inc = np.zeros(num_skill)
        self.rho_inc = np.zeros(num_skill)
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
                batch_UserID = np.array(train_vec.loc[shuffled_order[batch_idx], 'user_id'], dtype='int32')

                batch_skillID = np.array(train_vec.loc[shuffled_order[batch_idx], 'skill_id'], dtype='int32')
                batch_ProbID = np.array(train_vec.loc[shuffled_order[batch_idx], 'problem_id'], dtype='int32')

                # compute obj function
                sCounts = train_vec.loc[shuffled_order[batch_idx], 'sCount'].values
                fCounts = train_vec.loc[shuffled_order[batch_idx], 'fCount'].values

                x1 = np.multiply(train_vec.loc[shuffled_order[batch_idx], 'sCount'].values, self.gamma[batch_skillID])
                x2 = np.multiply(train_vec.loc[shuffled_order[batch_idx], 'fCount'].values, self.rho[batch_skillID])
                x = self.alpha[batch_UserID] + self.beta[batch_skillID] + x1 + x2 + self.beta2[batch_ProbID]
                expnx = np.exp(-x)
                linkx = np.divide(1, (1+expnx))
                # logx = np.log(linkx)
                # lognx = np.log(1-linkx)

                y = train_vec.loc[shuffled_order[batch_idx], 'correct'].values
                gradlogloss = -  np.multiply(y,1-linkx) + np.multiply(1-y, linkx)

                # compute the gradient
                beta_grad = 2 * gradlogloss + self._lambda * self.beta[batch_skillID]
                beta2_grad = 2 * gradlogloss + self._lambda * self.beta2[batch_ProbID]
                gamma_grad = 0.2 * np.multiply(gradlogloss, train_vec.loc[shuffled_order[batch_idx], 'sCount'].values) + \
                    self._lambda * self.gamma[batch_skillID]
                rho_grad = 0.2 * np.multiply(gradlogloss, train_vec.loc[shuffled_order[batch_idx], 'fCount'].values) + \
                    self._lambda * self.rho[batch_skillID]
                alpha_grad = 2 * gradlogloss + self._lambda * self.alpha[batch_UserID] 
                        
                dw_beta = np.zeros(num_skill)
                dw_beta2 = np.zeros(num_problem)
                dw_gamma = np.zeros(num_skill)
                dw_rho = np.zeros(num_skill)
                dw_alpha = np.zeros(num_user)
               

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_beta[batch_skillID[i]] += beta_grad[i]
                    dw_beta2[batch_ProbID[i]] += beta2_grad[i]
                    dw_gamma[batch_skillID[i]] += gamma_grad[i]
                    dw_rho[batch_skillID[i]] += rho_grad[i]
                    dw_alpha[batch_UserID[i]] += alpha_grad[i]

                # Update with momentum
                self.beta_inc = self.momentum * self.beta_inc + self.epsilon * dw_beta / self.batch_size   
                self.beta2_inc = self.momentum * self.beta2_inc + self.epsilon * dw_beta2 / self.batch_size  
                self.gamma_inc = self.momentum * self.gamma_inc + self.epsilon * dw_gamma / self.batch_size 
                self.rho_inc = self.momentum * self.rho_inc + self.epsilon * dw_rho / self.batch_size 
                self.alpha_inc = self.momentum * self.alpha_inc + self.epsilon * dw_alpha / self.batch_size
                        

                # gradien descent 
                self.beta = self.beta - self.beta_inc
                self.beta2 = self.beta2 - self.beta2_inc
                self.gamma = self.gamma - self.gamma_inc
                self.rho = self.rho - self.rho_inc
                self.alpha = self.alpha - self.alpha_inc

                if not self.dynamic:
                    self.gamma = np.zeros(num_skill)
                    self.rho = np.zeros(num_skill)


                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    batch_skillID = np.array(train_vec.loc[:, 'skill_id'], dtype='int32')
                    batch_UserID = np.array(train_vec.loc[:, 'user_id'], dtype='int32')
                    batch_ProbID = np.array(train_vec.loc[:, 'problem_id'], dtype='int32')
                    x1 = np.multiply(train_vec.loc[:, 'sCount'].values, self.gamma[batch_skillID])
                    x2 = np.multiply(train_vec.loc[:, 'fCount'].values, self.rho[batch_skillID])
                    x = self.alpha[batch_UserID] + self.beta[batch_skillID] + self.beta2[batch_ProbID] + x1 + x2
                    expnx = np.exp(-x)            
                    linkx = np.divide(1, (1+expnx))          
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)

                    y = train_vec.loc[:, 'correct'].values
                    logloss =  np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx))

                    obj = logloss \
                          + 0.5 * self._lambda * (np.linalg.norm(self.beta) ** 2 + np.linalg.norm(self.gamma) ** 2 + + np.linalg.norm(self.rho) ** 2)
                    
                
                    self.logloss_train.append((obj / pairs_train))
                
                # Compute validation error
                if batch == self.num_batches - 1:   
                    batch_skillID = np.array(test_vec.loc[:, 'skill_id'], dtype='int32')
                    batch_UserID = np.array(test_vec.loc[:, 'user_id'], dtype='int32')
                    batch_ProbID = np.array(test_vec.loc[:, 'problem_id'], dtype='int32')
                    x1 = np.multiply(test_vec.loc[:, 'sCount'].values, self.gamma[batch_skillID])
                    x2 = np.multiply(test_vec.loc[:, 'fCount'].values, self.rho[batch_skillID])
                    x = self.alpha[batch_UserID] + self.beta[batch_skillID] + self.beta2[batch_ProbID] + x1 + x2
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
                   
                        
    # TODO    
    def predict(self, invID):
        x = 1
        # x  = np.dot(self.w_Item, self.w_User[int(invID), :])
        expnx = np.exp(-x)            
        pred = np.divide(1, (1+expnx))     
        # return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot 点乘
        return pred

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)
            self.dynamic = parameters.get("dynamic", True)
