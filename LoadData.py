from numpy import *
import random


def load_rating_data(file_path='ml-100k/u.data'):
    

    

    prefer = []
    order= []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating, ts) = line.split(',')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
        order.append([])
    data = array(prefer)
    num_user = len(unique(data[:, 0]))
    hist_rec = []
    for i in range(num_user):
        hist_rec.append([])
    for i in range(len(order)):
        uid = data[i,0]
        iid = data[i,1]
        order[i] = hist_rec[int(uid)].copy() 
        if data[i,2]==1:
            hist_rec[int(uid)].append(int(iid))
    return data, order



    
    
    
def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = array(train_data)
    test_data = array(test_data)
    return train_data, test_data
