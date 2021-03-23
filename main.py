import numpy as np
import random
import torch
import os
import time
import copy

from utils import *
from model import *

hyperparameters = {
    'dataset_name': 'ml-1m', 
    'dir': 'trained_model',
    'batch_size': 128,
    'lr': 1e-3, 
    'max_len': 50, 
    'num_hidden_units': 50, 
    'num_blocks': 2,
    'num_epochs': 201, 
    'num_heads': 1, 
    'dropout': True,
    'dropout_val': 0.5, 
    'device': 'cuda', 
    'inference_only': False,
    'state_dict_path': False,
    'l2_emb': 0.0
}

if not os.path.isdir(hyperparameters['dataset_name'] + '_' + hyperparameters['dir']):
    os.makedirs(hyperparameters['dataset_name'] + '_' + hyperparameters['dir'])

with open(hyperparameters['dataset_name'] + '_' + hyperparameters['dir'] + '/params.txt', 'w') as file:
    file.write('\n'.join([str(key) + ',' + str(hyperparameters[key]) for key in hyperparameters]))

file.close()

train, val, test, num_users, num_items = split_data(hyperparameters['dataset_name'])

batch_len = len(train)//hyperparameters['batch_size'] 
file = open(hyperparameters['dataset_name'] + '_' + hyperparameters['dir']+'/logs.txt', 'w')

model = SASRecModel(num_users, num_items, hyperparameters).to(hyperparameters['device']) 

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_uniform_(param.data)
    except:
        pass

model.train()
epoch_num = 1

bce_criterion = torch.nn.BCEWithLogitsLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'], betas=(0.9, 0.9))

u, seq, pos, neg = sample_function(train, num_users, num_items, hyperparameters['batch_size'], hyperparameters['max_len'], np.random.randint(2e9)) #sampler.next_batch()
for epoch in range(epoch_num, hyperparameters['num_epochs'] + 1):

    for step in range(batch_len): 

        batch_data = sample_function(train, num_users, num_items, hyperparameters['batch_size'], hyperparameters['max_len'], np.random.randint(2e9)) #sampler.next_batch()
        u, seq, pos, neg = batch_data

        u = np.array(u)
        seq = np.array(seq)
        pos = np.array(pos)
        neg = np.array(neg)
        
        pos_output, neg_output = model(u, seq, pos, neg)
        pos_y = torch.ones(pos_output.shape, device=hyperparameters['device'])
        neg_y = torch.zeros(neg_output.shape, device=hyperparameters['device'])
        idx = np.where(pos != 0)

        adam_optimizer.zero_grad()
        loss = bce_criterion(pos_output[idx], pos_y[idx])
        loss = loss + bce_criterion(neg_output[idx], neg_y[idx])
        
        for param in model.item_embeddings.parameters(): 
          loss = loss + (hyperparameters['l2_emb'] * torch.norm(param))

        loss.backward()
        adam_optimizer.step()
        
        print("loss for Epoch Number{} iteration Number {}: {}".format(epoch, step, loss.item()))

    if epoch % 25 == 0:
        model.eval()

        print('Evaluating', end='')
        test_scr = evaluate(model, [train, test, num_users, num_items], hyperparameters, 'test')
        val_scr = evaluate(model, [train, val, num_users, num_items], hyperparameters)
        print(" ")
        print('epoch:%d, valid (NDCG@10: %.4f, HitRate@10: %.4f), test (NDCG@10: %.4f, HitRate@10: %.4f)'
                % (epoch, val_scr[0], val_scr[1], test_scr[0], test_scr[1]))

        file.write(str(val_scr) + ' ' + str(test_scr) + '\n')
        file.flush()

        model.train()

    if epoch == hyperparameters['num_epochs']:
        path = hyperparameters['dataset_name'] + '_' + hyperparameters['dir']
        model_filename = 'SASRecModel.pth' 
        model_filename = model_filename.format(hyperparameters['num_epochs'], hyperparameters['lr'], hyperparameters['num_blocks'], hyperparameters['num_heads'], hyperparameters['num_hidden_units'], hyperparameters['max_len'])
        torch.save(model.state_dict(), os.path.join(path, model_filename))

file.close()
print("Training Completed")