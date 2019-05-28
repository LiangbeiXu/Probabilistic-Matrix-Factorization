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
import random
import pickle

# use split problem ids
def PreprocessAssistment15(file_path):
    data = pd.read_csv(file_path,dtype={ 'log_id':np.int, \
        'sequence_id':np.int, 'user_id':np.int, 'correct':np.float})
    # data = data.rename(columns={'log_id': 'order_id', 'sequence_id': 'skill_id'})
    # print('# of records: %d.' %(data.shape[0]))
    data.dropna(inplace=True)
    # print('After dropping NaN rows, # of records: %d.' %(data.shape[0]))
    data.sort_values('order_id',ascending=True, inplace=True)
    data.correct = data.correct.astype(np.int)
    data.skill_id = data.skill_id.astype(np.int)
    return ReMapID_skill_user(data)

def PreprocessAssistment(file_path):
    data = pd.read_csv(file_path,dtype={'skill_name':np.str, 'order_id':np.int, \
        'problem_id':np.int, 'user_id':np.int, 'correct':np.int, 'original':np.int})
    data.drop(axis=1,columns='skill_name', inplace=True)
    # print('# of records: %d.' %(data.shape[0]))
    data.dropna(inplace=True)
    # print('After dropping NaN rows, # of records: %d.' %(data.shape[0]))
    data.sort_values('order_id',ascending=True, inplace=True)
    data.drop(data[ data.original==0 ].index, axis=0, inplace=True)
    # print('After dropping scafolding problems, # of records: %d.' %(data.shape[0]))
    data.skill_id = data.skill_id.astype(np.int)
    return ReMapID_prob_skill_user(data)


# make the IDs in dataframe continuous from 0 to number-1, slow method
def ReMapID(data, fields):
    for field in fields:
        xid = np.unique(data[field])
        xid.sort()
        xdic = dict(zip(xid, list(range(0, len(xid)))))
        for i in range(data.shape[0]):
            data.loc[i, field] = xdic.get(data.loc[i, field])
    return data


# make the IDs in dataframe continuous from 0 to number-1
def ReMapID_prob_skill_user(data):
    pid = np.unique(data['problem_id'])
    pid.sort()
    uid = np.unique(data['user_id'])
    uid.sort()
    sid = np.unique(data['skill_id'])
    sid.sort()
    # change dtype of skill_id
    data.skill_id = data.skill_id.astype(np.int)
    # print(data.shape[0],  len(np.unique(uid)), len(np.unique(pid)), len(np.unique(sid)))
    pdic = dict(zip(pid, list(range(0, len(pid)))))
    udic = dict(zip(uid, list(range(0, len(uid)))))
    sdic = dict(zip(sid, list(range(0, len(sid)))))
    data['problem_id'] = data.problem_id.replace(pdic)
    data['skill_id'] = data.skill_id.replace(sdic)
    data['user_id'] = data.user_id.replace(udic)
    return data


# make the IDs in dataframe continuous from 0 to number-1
def ReMapID_skill_user(data):

    uid = np.unique(data['user_id'])
    uid.sort()
    sid = np.unique(data['skill_id'])
    sid.sort()
    # change dtype of skill_id
    data.skill_id = data.skill_id.astype(np.int)
    # print(data.shape[0],  len(np.unique(uid)), len(np.unique(sid)))

    udic = dict(zip(uid, list(range(0, len(uid)))))
    sdic = dict(zip(sid, list(range(0, len(sid)))))

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
        data.skill_id = data.skill_id.astype(np.int)
        data = ReMapID_prob_skill_user(data)
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
        prob_skill_map[i].sort()
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

def PreprocessAssistmentSkillBuilder(file_path, option='multiskills'):
    # three options: multiskills, newskills, nonmultiskills
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

    multi_prob_set = []
    prob_skill_map_dummy = prob_skill_map
    new_skill_idx = num_skills
    multi_skills_set = []
    for idx in range(len(prob_skill_map_dummy)):
        skills = prob_skill_map_dummy[idx]
        if len(skills) > 1 and option == 'newskills':
            multi_prob_set.append(idx)
            prob_skill_map[idx] = [new_skill_idx]
            new_skill_idx += 1
            multi_skills_set.append(skills)

    for i in range(num_records):
        # order_in not found
        if not (datanp[i, 0] in order_id_set) :
            if (len(prob_skill_map[datanp[i, 2]]) > 1) and option == 'nonmultiskills':
                continue
            else:
                order_id_set.add(datanp[i, 0])
                new_data.append(datanp[i,:])

    data_new_np = np.asarray(new_data)

    # generate skill_ids for each problem
    skill_id_values = list()
    for i in range(data_new_np.shape[0]):
        if option == 'multiskills':
            skill_id_values.append(prob_skill_map[data_new_np[i,2]])
        else:
            skill_id_values.append(prob_skill_map[data_new_np[i,2]][0])


    columnNames = list(data.head(0))
    data_new = pd.DataFrame(data=data_new_np, columns=columnNames)
    data_new.drop(axis=1,columns='skill_id', inplace=True)
    if option == 'multiskills':
        data_new['skill_ids'] = skill_id_values
    elif option == 'newskills':
        data_new['skill_id'] = skill_id_values
        num_skills = new_skill_idx
    else:
        data_new['skill_id'] = skill_id_values
        # CAREFUL: we delete entries with multiskills. need to remap IDs
        data_new = ReMapID_prob_skill_user(data_new)
        skill_id_values = data_new['skill_id'].values
        data_new_np = data_new.values
        # new statistics
        prob_skill_map = GenerateProblemSkillsMap(data_new)
        num_users = len(np.unique(data_new['user_id']))
        num_skills = len(np.unique(data_new['skill_id']))

    # success. failure counts
    sCnt = np.zeros(shape=(num_users, num_skills),dtype=np.int)
    fCnt = np.zeros(shape=(num_users, num_skills),dtype=np.int)
    sCntList = []
    fCntList = []

    for i in range(data_new_np.shape[0]):
        sCntList.append(0)
        fCntList.append(0)

    for index in range(data_new_np.shape[0]):
        user_id = data_new_np[index][1]
        skill_ids = skill_id_values[index]
        if option == 'multiskills':
            for skill_id in skill_ids:
                sCntList[index] = (deepcopy(sCnt[user_id][skill_id]))
                fCntList[index] = (deepcopy(fCnt[user_id][skill_id]))
                # update counts
                if datanp[index, 4] == 1:
                    sCnt[user_id][skill_id]  += 1
                else:
                    fCnt[user_id][skill_id]  += 1
        else:
            skill_id = skill_ids
            sCntList[index]= (deepcopy(sCnt[user_id][skill_id]))
            fCntList[index]= (deepcopy(fCnt[user_id][skill_id]))
            # update counts
            if datanp[index, 4] == 1:
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

    if 0:
        # new columns contain skill information for each problem
        columnNames = list(data.head(0))
        data_new = pd.DataFrame(data=data_new_np, columns=columnNames)
        data_new.drop(axis=1,columns='skill_id', inplace=True)
        if option == 'multiskills':
            data_new['skill_ids'] = skill_id_values
        else:
            data_new['skill_id'] = skill_id_values

    # add columns
    data_new['sCount'] = sCntList
    data_new['fCount'] = fCntList
    data_new['hist'] = hist_rec_list


    return data_new, num_skills, prob_skill_map


def PreprocessAssistment15SkillBuilder(file_path):

    data = PreprocessAssistment15(file_path)

    # num_problems = len(np.unique(data['problem_id']))
    num_users = len(np.unique(data['user_id']))
    num_skills = len(np.unique(data['skill_id']))
    num_records = data.shape[0]


    # success. failure counts
    sCnt = np.zeros(shape=(num_users, num_skills), dtype=np.int)
    fCnt = np.zeros(shape=(num_users, num_skills), dtype=np.int)
    sCntList = np.zeros(shape=(num_records), dtype=np.int)
    fCntList = np.zeros(shape=(num_records), dtype=np.int)
    data_np = data.values
    col_names = list(data.columns.values)
    col_indexes = range(len(col_names))
    col_dic = dict(zip(col_names, list(range(0, len(col_names)))))
    for index in range(data_np.shape[0]):
        user_id = data_np[index,col_dic.get('user_id')]
        skill_id = data_np[index,col_dic.get('skill_id')]

        sCntList[index]=(deepcopy(sCnt[user_id][skill_id]))
        fCntList[index]=(deepcopy(fCnt[user_id][skill_id]))
        # update counts
        if data_np[index,col_dic.get('correct')]==1:
            sCnt[user_id][skill_id]  += 1
        else:
            fCnt[user_id][skill_id]  += 1
    # add columns
    data['sCount'] = sCntList
    data['fCount'] = fCntList
    if True:
        hist_rec = []
        hist_rec_list = []
        for i in range(num_users):
            hist_rec.append([])

        for index in range(data_np.shape[0]):
            user_id = data_np[index,col_dic.get('user_id')]
            skill_id = data_np[index,col_dic.get('skill_id')]
            hist_rec_list.append([])
            hist_rec_list[index] = hist_rec[user_id].copy()
            if(data_np[index, 3]==1):
                if len( hist_rec[user_id]) < 10:
                    hist_rec[user_id].append(skill_id)
                else:
                    hist_rec[user_id][0:-1] = hist_rec[user_id][1:len(hist_rec[user_id])]
                    hist_rec[user_id][len(hist_rec[user_id])-1] = skill_id
        data['hist'] = hist_rec_list


    return data

def SplitSeq(data, validation_rate, testing_rate, shuffle=True):
    def split(dt):
        return [[value[0] for value in seq] for seq in dt], [[value[1] for value in seq] for seq in dt]

    seqs = data
    if shuffle:
        random.shuffle(seqs)

    # Get testing data
    test_idx = random.sample(range(0, len(seqs)-1), int(len(seqs) * testing_rate))
    X_test, y_test = split([value for idx, value in enumerate(seqs) if idx in test_idx])
    seqs = [value for idx, value in enumerate(seqs) if idx not in test_idx]

    # Get validation data
    val_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * validation_rate))
    X_val, y_val = split([value for idx, value in enumerate(seqs) if idx in val_idx])

    # Get training data
    X_train, y_train = split([value for idx, value in enumerate(seqs) if idx not in val_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test


def Data2Seq(data):
    # Step 2 - Convert to sequence by student id
    students_seq = data.groupby("user_id", as_index=True)["skill_id", "correct"].apply(lambda x: x.values.tolist()).tolist()

    # Step 3 - Rearrange the skill_id
    seqs_by_student = {}
    skill_ids = {}
    num_skill = 0

    for seq_idx, seq in enumerate(students_seq):
        for (skill, answer) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            if skill not in skill_ids:
                skill_ids[skill] = num_skill
                num_skill += 1

            seqs_by_student[seq_idx].append((skill_ids[skill], answer))

    return list(seqs_by_student.values()), num_skill
# TODO

def RemoveLeastItemUser(data):
    num_user = len(np.unique(data['user_id']))
    num_prob = len(np.unique(data['problem_id']))
    prob_cnt_per_user = np.zeros((num_user))
    for i in range(data.shape[0]):
        prob_cnt_per_user[data.loc[i, 'user_id']] += 1

def from_csv():

    dataset_name =  'Assistment09-problem.csv'
    file_path = '../' + dataset_name
    data = pd.read_csv(file_path)
    print(dataset_name, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])))
    dataset_name =  'Assistment09-skill.csv'
    file_path = '../' + dataset_name
    data = pd.read_csv(file_path)
    print(dataset_name, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])))
    dataset_name =  'Assistment15-skill.csv'
    file_path = '../' + dataset_name
    data = pd.read_csv(file_path)
    print(dataset_name, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])))

    dataset_name =  'Assistment09-problem-new_skill.csv'
    file_path = '../' + dataset_name
    data = pd.read_csv(file_path)
    print(dataset_name, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))

    dataset_name =  'Assistment09-problem-single_skill.csv'
    file_path = '../' + dataset_name
    data = pd.read_csv(file_path)
    print(dataset_name, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))

def to_csv():
    dataDir = '/home/lxu/Documents/StudentLearningProcess/'
    if 0:
        # Assistment 09
        dataPath = dataDir + 'skill_builder_data_corrected_withskills_section.csv'
        items = ['skill', 'problem']
        for item in items:
            data = PreprocessAssistmentProblemSkill(dataPath, item)
            pickle_name = 'Assistment09-' + item + '.pickle'

            #
            csv_name = 'Assistment09-' + item + '.csv'
            data_new = data[['user_id', item+'_id', 'correct', 'order_id']]
            data_new.to_csv(dataDir + csv_name, index=False)

            pickle_out = open(dataDir+ pickle_name,"wb")
            pickle.dump(data, pickle_out)
            pickle_out.close()
            # read the pickle data to verify
            pickle_in = open(dataDir + pickle_name, 'rb')
            data_in = pickle.load(pickle_in)
            # print(data_in.head())
            for col in data_in.columns:
                print(col)
            print(data_in.iloc[0:10,:])
    if 0:
        # another dataset
        dataPath = dataDir + 'skill_builder_data_corrected_withskills_section.csv'

        data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(dataPath)
        csv_name = 'Assistment09-' + 'problem-history' + '.csv'
        data_new = data[['user_id', 'problem_id', 'correct', 'order_id']]
        data_new.to_csv(dataDir + csv_name, index=False)

        pickle_name = 'Assistment09-' + 'skillbuilder' + '.pickle'
        pickle_out = open(dataDir+ pickle_name,"wb")
        data_new = {'data':data, 'skill_num': num_skills, 'prob_skill_map': prob_skill_map}
        pickle.dump(data_new, pickle_out)
        pickle_out.close()
        # read the pickle data to verify
        pickle_in = open(dataDir + pickle_name, 'rb')
        dummy = pickle.load(pickle_in)
        data_in = dummy['data']
        # print(data_in.head())
        # for col in data_in.columns:
        #     print(col)
        print(data_in.iloc[0:10,:])

    if 1:
        # another dataset
        dataPath = dataDir + 'skill_builder_data_corrected_withskills_section.csv'

        data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(dataPath)
        csv_name = 'Assistment09-' + 'problem' + '.csv'
        data_new = data[['user_id', 'problem_id', 'correct', 'order_id', 'skill_ids', 'sCount', 'fCount']]
        data_new.to_csv(dataDir + csv_name, index=False)

        data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(dataPath, option='newskills')
        csv_name = 'Assistment09-' + 'problem-new_skill' + '.csv'
        data_new = data[['user_id', 'problem_id', 'correct', 'order_id', 'skill_id', 'sCount', 'fCount']]
        # data_new.rename(columns = {'skill_ids':'skill_id'})
        data_new.to_csv(dataDir + csv_name, index=False)

        data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(dataPath, option='nonmultiskills')
        csv_name = 'Assistment09-' + 'problem-single_skill' + '.csv'
        data_new = data[['user_id', 'problem_id', 'correct', 'order_id', 'skill_id', 'sCount', 'fCount']]
        # data_new.rename(columns = {'skill_ids':'skill_id'})
        data_new.to_csv(dataDir + csv_name, index=False)
    if 0:
        # Assistment 15
        dataPath = dataDir + '2015_ASSISTment.csv'
        items = ['skill']
        for item in items:
            data = PreprocessAssistment15(dataPath)
            csv_name = 'Assistment15-' + item + '.csv'
            data_new = data[['user_id', item+'_id', 'correct', 'order_id', 'sCount', 'fCount']]
            data_new.to_csv(dataDir + csv_name, index=False)

            # pickle_name = 'Assistment15-' + item + '.pickle'
            # pickle_out = open(dataDir+ pickle_name,"wb")
            # pickle.dump(data, pickle_out)
            # pickle_out.close()
            # read the pickle data to verify
            pickle_name = 'Assistment15-skill.pickle'
            pickle_in = open(dataDir + pickle_name, 'rb')
            data_in = pickle.load(pickle_in)
            # print(data_in.head())
            for col in data_in.columns:
                print(col)
            print(data_in.dtypes)
            print(data_in.iloc[0:10,:])




if __name__ == '__main__':
    # to_csv()
    # from_csv()

