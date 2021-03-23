import numpy as np
import random
import copy
import sys

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: 
              t = np.random.randint(1, itemnum + 1)
              while t in ts:
                t = np.random.randint(1, itemnum + 1)
              neg[idx] = t
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    one_batch = []
    for i in range(batch_size):
      one_batch.append(sample())

    return zip(*one_batch)

# split data in train, validation and test sets for training
def split_data(dataset_name):
    num_users = 0
    num_items = 0

    user_dict = {}
    train = {}
    val = {}
    test = {}
    # assume user/item index starting from 1
    file = open('./data/%s.txt' % dataset_name, 'r')
    for l in file:
        u, i = l.rstrip().split(' ')
        u = int(u)
        i = int(i)
        num_users = max(u, num_users)
        num_items = max(i, num_items)
        if u not in user_dict:
          user_dict[u] = []
        user_dict[u].append(i)

    for user in user_dict:
        num_reviews = len(user_dict[user])
        if num_reviews < 3:
            train[user] = user_dict[user]
            val[user] = []
            test[user] = []
        else:
            train[user] = user_dict[user][:-2]
            val[user] = []
            val[user].append(user_dict[user][-2])
            test[user] = []
            test[user].append(user_dict[user][-1])

    return train, val, test, num_users, num_items

def evaluate(model, dataset, hyperparameters, tp='val'):
  [train, data, usernum, itemnum] = copy.deepcopy(dataset)
  NDCG = 0.0
  HitRate = 0.0
  valid_user = 0.0

  if usernum>10000:
    users = random.sample(range(1, usernum + 1), 10000)
  else:
    users = range(1, usernum + 1)

  for u in users:
    if len(train[u]) < 1 or len(data[u]) < 1: continue

    seq = np.zeros([hyperparameters['max_len']], dtype=np.int32)
    idx = hyperparameters['max_len'] - 1
    
    if tp != 'val':
      seq[idx] = data[u][0]
      idx -= 1

    for i in reversed(train[u]):
      seq[idx] = i
      idx -= 1
      if idx == -1: break    

    rated = set(train[u])
    rated.add(0)
    item_idx = [data[u][0]] 

    for _ in range(100):
      t = np.random.randint(1, itemnum + 1)
      while t in rated: t = np.random.randint(1, itemnum + 1)
      item_idx.append(t)

    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
    predictions = predictions[0]

    rank = predictions.argsort().argsort()[0].item()

    valid_user += 1

    if rank < 10:
        NDCG += 1 / np.log2(rank + 2)
        HitRate += 1
    if valid_user % 100 == 0:
        print('.', end="")
        sys.stdout.flush()

  return NDCG / valid_user, HitRate / valid_user