#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:56:26 2019

@author: lxu
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PerformanceFactorAnalysis import PFA
from copy import deepcopy
import matplotlib.pyplot as plt

# use split problem ids

def PreprocessAssistment(file_path):
    data = pd.read_csv(file_path,dtype={'skill_name':np.str, 'order_id':np.int, \
        'problem_id':np.int, 'user_id':np.int, 'correct':np.int})
    data.drop(axis=1,columns='skill_name', inplace=True)
    print('# of records: %d.' %(data.shape[0]))
    data.dropna(inplace=True)
    print('After dropping NaN rows, # of records: %d.' %(data.shape[0]))
    data.sort_values('order_id',ascending=True, inplace=True)
    return ReMapID(data)

# make the IDs in dataframe continuous from 0 to number-1
def ReMapID(data): 
    pid = np.unique(data['problem_id'])
    pid.sort()
    uid = np.unique(data['user_id'])
    uid.sort()
    sid = np.unique(data['skill_id'])
    sid.sort()
    # change dtype of skill_id
    data.skill_id = data.skill_id.astype(np.int)
    print(data.shape[0],  len(np.unique(uid)), len(np.unique(pid)), len(np.unique(sid)))
    pdic = dict(zip(pid, list(range(0, len(pid)))))
    udic = dict(zip(uid, list(range(0, len(uid)))))
    sdic = dict(zip(sid, list(range(0, len(sid))))) 
    data['problem_id'] = data.problem_id.replace(pdic)
    data['skill_id'] = data.skill_id.replace(sdic)
    data['user_id'] = data.user_id.replace(udic)
    return data


def PreprocessAssistmentProblemSkill(file_path, model_flag):
    data = PreprocessAssistment(file_path)

    num_users = len(np.unique(data['user_id']))
    data_new_np = data.values
     # problem history record
    hist_rec = []
    hist_rec_list = []
    for i in range(num_users):
        hist_rec.append([])

    for index in range(data_new_np.shape[0]):
        user_id = data_new_np[index][1]
        if model_flag == 'problem':
            item_id = data_new_np[index][2]
        elif model_flag == 'skill':
            item_id = data_new_np[index][4]
        hist_rec_list.append([])
        hist_rec_list[index] = hist_rec[user_id].copy()
        if(data_new_np[index, 3]==1):
            hist_rec[user_id].append((item_id))
    columnNames = list(data.head(0)) 
    data_new = pd.DataFrame(data=data_new_np, columns=columnNames) 
    data_new['hist'] = hist_rec_list
    return data_new

def PreprocessAssistmentPFASingleSkill(file_path, single_skill_only):
    data = PreprocessAssistment(file_path)
    if single_skill_only:
        data = DeleteMultipleSkillsProblem(data)
        data = ReMapID(data)
    # num_problems = len(np.unique(data['problem_id']))
    num_users = len(np.unique(data['user_id']))
    num_skills = len(np.unique(data['skill_id']))
    data_new_np = data.values
    # success. failure counts 
    sCnt = np.zeros(shape=(num_users, num_skills))
    fCnt = np.zeros(shape=(num_users, num_skills))
    sCntList = []
    fCntList = []

    #for i in range(data_new_np.shape[0]):
    #    sCntList.append([])
    #    fCntList.append([])
        
    for index in range(data_new_np.shape[0]):
        user_id = data_new_np[index][1]    
        skill_id = data_new_np[index][4]
        
        sCntList.append(deepcopy(sCnt[user_id][skill_id]))
        fCntList.append(deepcopy(fCnt[user_id][skill_id])) 
        # update counts 
        if data_new_np[index, 3]:
            sCnt[user_id][skill_id]  += 1
        else:
            fCnt[user_id][skill_id]  += 1
    # new columns contain skill information for each problem
    columnNames = list(data.head(0)) 
    data_new = pd.DataFrame(data=data_new_np, columns=columnNames)     
    data_new['sCount'] = sCntList
    data_new['fCount'] = fCntList
    return data_new


def GenerateProblemSkillsMap(data):
    skill_id = data.loc[:,'skill_id'].values
    problem_id = data.loc[:,'problem_id'].values
    num_problems = len(np.unique(problem_id))
    prob_skill_map = []
    prob_skill_map_temp = []
    for i in range(num_problems):
        prob_skill_map_temp.append(set())
        prob_skill_map.append(list())
    for i in range(len(skill_id)):
        prob_skill_map_temp[problem_id[i]].add(skill_id[i])

    # set to list
    for i in range(num_problems):
        prob_skill_map[i] = list(prob_skill_map_temp[i])
    return prob_skill_map


def DeleteMultipleSkillsProblem(data):
    prob_skill_map = GenerateProblemSkillsMap(data)
    datanp = data.values
    to_delete  = list()
    for i in range(data.shape[0]):
        if(len(prob_skill_map[data.loc[i, 'problem_id']])>=2):
            to_delete.append(deepcopy(i)) 
    datanp = np.delete(datanp, to_delete, axis=0)
    columnNames = list(data.head(0)) 
    data_new = pd.DataFrame(data=datanp, columns=columnNames)
    return data_new

def PreprocessAssistmentSkillBuilder(file_path):

    data = PreprocessAssistment(file_path)

    # num_problems = len(np.unique(data['problem_id']))
    num_users = len(np.unique(data['user_id']))
    num_skills = len(np.unique(data['skill_id']))
    prob_skill_map = GenerateProblemSkillsMap(data)
    # reconstruct the dataframe by merging records of same order_id
    datanp = data.to_numpy()
    num_records = datanp.shape[0]
    order_id_set = set()
    new_data = list()
    for i in range(num_records):
        # order_in not found
        if not (datanp[i, 0] in order_id_set):
            order_id_set.add(datanp[i, 0])
            new_data.append(datanp[i,:])
    data_new_np = np.asarray(new_data)

    # generate skill_ids for each problem
    skill_id_values = list()
    for i in range(data_new_np.shape[0]):
        skill_id_values.append(prob_skill_map[data_new_np[i,2]])


    # success. failure counts 
    sCnt = np.zeros(shape=(num_users, num_skills))
    fCnt = np.zeros(shape=(num_users, num_skills))
    sCntList = []
    fCntList = []

    for i in range(data_new_np.shape[0]):
        sCntList.append(list())
        fCntList.append(list())
        
    for index in range(data_new_np.shape[0]):
        user_id = data_new_np[index][1]    
        skill_ids = skill_id_values[index]
        for skill_id in skill_ids:
            sCntList[index].append(deepcopy(sCnt[user_id][skill_id]))
            fCntList[index].append(deepcopy(fCnt[user_id][skill_id]))
            # update counts 
            if datanp[index, 3]:
                sCnt[user_id][skill_id]  += 1
            else:
                fCnt[user_id][skill_id]  += 1

    # problem history record
    hist_rec = []
    hist_rec_list = []
    for i in range(num_users):
        hist_rec.append([])

    for index in range(data_new_np.shape[0]):
        user_id = data_new_np[index][1]
        problem_id = data_new_np[index][2]
        hist_rec_list.append([])
        hist_rec_list[index] = hist_rec[user_id].copy()
        if(data_new_np[index, 3]==1):
            hist_rec[user_id].append((problem_id))

    # new columns contain skill information for each problem
    columnNames = list(data.head(0)) 
    data_new = pd.DataFrame(data=data_new_np, columns=columnNames) 

    # add columns 
    data_new['skill_ids'] = skill_id_values
    data_new.drop(axis=1,columns='skill_id', inplace=True)
    data_new['sCount'] = sCntList
    data_new['fCount'] = fCntList
    data_new['hist'] = hist_rec_list

    return data_new