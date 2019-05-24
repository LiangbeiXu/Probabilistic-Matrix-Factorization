
from PreprocessAssistment import PreprocessAssistmentSkillBuilder

from IRT import IRT
from BPMFSkillEncoded import BPMFSkillEncoded

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def PlotLoss(model, model_name, dataset_name):
    plt.plot(range(model.maxepoch), model.acc_train, marker='o', label='ACC: Train Data')
    plt.plot(range(model.maxepoch), model.acc_test, marker='v', label='ACC: Test Data')
    plt.plot(range(model.maxepoch), model.auc_train, marker='+', label='AUC: Train Data')
    plt.plot(range(model.maxepoch), model.auc_test, marker='*', label='AUC: Test Data')
    plt.title(dataset_name + 'Learning Curve ' + model_name)
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()


test_size = 20000

# Assistment dataset
dataset_name =  'skill_builder_data_corrected_withskills_section.csv'
file_path = '/home/lxu/Documents/StudentLearningProcess/' + dataset_name
data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(file_path)
print('ASSISTment 09-10 statistics ', data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), num_skills)
data = data.sort_values(by=['order_id'])
# prepare data
datanp = data.values
testnp = datanp[-int(test_size):,:]
trainnp = datanp[0:-int(test_size),:]

# testnp = datanp[-int(0.1*datanp.shape[0]):,:]
# trainnp = datanp[0:-int(0.1*datanp.shape[0]+1),:]
# trainnp, testnp = train_test_split(datanp, test_size=0.2)
#
columnNames = list(data.head(0))
train = pd.DataFrame(data=trainnp, columns=columnNames)
test = pd.DataFrame(data=testnp, columns=columnNames)

# store all the models
models = []

 
name =  'global + skill'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'multi_skills': True, 'user':False })
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])

 
name =  'global + skill + PFA'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'multi_skills': True, 'user':False,'PFA':True })
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])

 
name =  'global + user + skill'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'multi_skills': True })
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])

 
name =  'global + user + skill + PFA'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'multi_skills': True, 'PFA': True, 'problem': False})
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])

 
name =  'global + user + prob + skill'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'multi_skills': True, 'PFA': False, 'problem': True})
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])


 
name =  'global + user + prob + PFA'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'multi_skills': True, 'PFA': True, 'problem': True})
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])




name =  'globa + user + prob + MF'
print(dataset_name, name)
model= IRT(epsilon=4, _lambda=0.1, momentum=0.8, maxepoch=50, num_batches=300, batch_size=1000,\
                 problem=True, multi_skills=False, user_skill=False, user_prob=False, PFA=False, MF=True,\
                 num_feat=16, MF_skill=False, user=True, skill_dyn_embeddding=False, skill=False, global_bias=False)
model.fit(train, test, len(np.unique(data['user_id'])), num_skills, len(np.unique(data['problem_id'])))
models.append([dataset_name, name, model])



# skill and problem
ratings = data.loc[:,['user_id', 'problem_id', 'correct', 'skill_ids', 'sCount', 'fCount']].values
order = list(data.loc[:]['hist'].values)

test = ratings[-int(test_size):,:]
train = ratings[0:-int(test_size),:]
# print(len(order))
order_test = order[-int(test_size):]
order_train = order[0:-int(test_size)]



name =  'globa + user + prob + encoded prob_latent_matrix + dynamic'
print(dataset_name, name)
model= BPMFSkillEncoded()
model.set_params({"num_feat": num_skills, "epsilon": 5, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                "batch_size": 1000, 'dynamic':True})

model.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), prob_skill_map)
models.append([dataset_name, name, model])

 
name =  'globa + user + prob + encoded prob_latent_matrix'
print(dataset_name, name)
model= BPMFSkillEncoded()
model.set_params({"num_feat": num_skills, "epsilon": 5, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                "batch_size": 1000, 'dynamic':False})
model.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), prob_skill_map)
models.append([dataset_name, name, model])

 
name =  'globa + user + prob + encoded prob_latent_matrix + dynamic'
print(dataset_name, name)
model= BPMFSkillEncoded()
model.set_params({"num_feat": num_skills, "epsilon": 5, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                "batch_size": 1000, 'dynamic':True})

model.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), prob_skill_map)
models.append([dataset_name, name, model])



# Assistment-15

dataset_name =  'Assistment15-skill.csv'
file_path = '../' + dataset_name
data = pd.read_csv(file_path)
print('ASSISTment 14-15 statistics ', data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])))
# prepare data
datanp = data.iloc[:,0:-1].values
hist = data.iloc[:,data.shape[1]-1].values

#testnp = datanp[-int(0.1*datanp.shape[0]):,:]
#trainnp = datanp[0:-int(0.1*datanp.shape[0]-1),:]
#hist_test = hist[-int(0.1*len(hist)):]
#hist_train = hist[0:-int(0.1*len(hist)-1)]

testnp = datanp[-int(test_size):,:]
trainnp = datanp[0:-int(test_size),:]
hist_test = hist[-int(test_size):]
hist_train = hist[0:-int(test_size)]

# trainnp, testnp, hist_train, hist_test = train_test_split(datanp, hist, test_size=0.2)
#
columnNames = list(data.head(0))
train = pd.DataFrame(data=trainnp, columns=columnNames[0:-1])
test = pd.DataFrame(data=testnp, columns=columnNames[0:-1])
train['hist'] = hist_train
test['hist'] = hist_test

 
name = 'global  + skill'
print(dataset_name, name)
model= IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'user':False})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10)
models.append([dataset_name, name, model])


 
name =  'global + skill + pfa'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000,'PFA':True, 'user':False})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])

 
name =  'global + user  + skill '
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])

 
name =  'global + user  + skill + mf'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000,'PFA':False, 'MF_skill':True})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])

 
name =  'global + user  + skill + pfa'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000,'PFA':True})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])

 
name =  'global + user  + skill + user_skill'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000,'user_skill':True})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])


name =  'global + user  + skill + pfa + user_skill'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000,'PFA':True, 'user_skill':True})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])


 
name =  'global + user  + skill + mf + pfa + user_skill'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000,'PFA':True, 'MF_skill':True, 'user_skill': True})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])

 
name =  'global + user + skill + dynamic skill embedding + PFA'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'skill_dyn_embeddding':True, 'PFA':True})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])


 
name =  'global + user + skill + dynamic skill embedding'
print(dataset_name, name)
model = IRT()
model.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'skill_dyn_embeddding':True, 'PFA':False})
model.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT()
models.append([dataset_name, name, model])



#for model in models:
#    PlotLoss(model[2], model[1], model[0])
