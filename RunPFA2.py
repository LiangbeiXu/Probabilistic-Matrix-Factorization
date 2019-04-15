#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:58:33 2019

@author: lxu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PerformanceFactorAnalysis import PFA
from PreprocessAssistment import PreprocessAssistmentPFASingleSkill, PreprocessAssistmentSkillBuilder
import matplotlib.pyplot as plt

# file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills.csv"
file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills_section.csv"
# data = PreprocessAssistmentPFASingleSkill(file_path, False)
data, num_skills = PreprocessAssistmentSkillBuilder(file_path)

print('statistics ', data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), num_skills)

datanp = data.values


trainnp, testnp = train_test_split(datanp, test_size=0.2)
#
columnNames = list(data.head(0)) 
train = pd.DataFrame(data=trainnp, columns=columnNames)
test = pd.DataFrame(data=testnp, columns=columnNames)

pfa = PFA()
pfa.set_params({ "epsilon": 0.5, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 50, "num_batches": 300,
                    "batch_size": 1000, "dynamic": True})

# pfa.fit_single_skill(train, test, len(np.unique(data['user_id'])),  len(np.unique(data['skill_id'])), len(np.unique(data['problem_id'])), False)
pfa.fit_single_skill(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])), True)
plt.plot(range(pfa.maxepoch), pfa.logloss_train, marker='o', label='Training Data')
plt.plot(range(pfa.maxepoch), pfa.logloss_test, marker='v', label='Test Data')
plt.plot(range(pfa.maxepoch), pfa.auc_test, marker='*', label='AUC: Test Data')
plt.title('The Truncated Assitment Dataset Learning Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()
