import torch
import numpy as np
import copy

class PW_FeedForward(torch.nn.Module):
    def __init__(self, num_hidded_units, dropout):
        super(PW_FeedForward, self).__init__()

        self.conv_1 = torch.nn.Conv1d(num_hidded_units, num_hidded_units, kernel_size=1)
        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv1d(num_hidded_units, num_hidded_units, kernel_size=1)
        self.dropout_2 = torch.nn.Dropout(p=dropout)

    def forward(self, input):
        output = self.conv_1(input.transpose(-1, -2))
        output = self.dropout_1(output)
        output = self.relu_1(output)
        output = self.conv_2(output)
        output = self.dropout_2(output)
        output = output.transpose(-1, -2)
        output = output + input
        return output

class SASRecModel(torch.nn.Module):
    def __init__(self, num_users, num_items, hyperparameters):
        super(SASRecModel, self).__init__()
        self.num_items = num_items
        self.num_users = num_users

        self.item_embeddings = torch.nn.Embedding(self.num_items+1, hyperparameters['num_hidden_units'], padding_idx=0)
        self.positional_embeddings = torch.nn.Embedding(hyperparameters['max_len'], hyperparameters['num_hidden_units'])
        self.dropout_embeddings = torch.nn.Dropout(hyperparameters['dropout_val'])

        self.PW_feedforward_layernorm_lis = torch.nn.ModuleList()
        self.PW_feedforward_layers_lis = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hyperparameters['num_hidden_units'], eps=1e-8)
        
        self.layernorm_lis = torch.nn.ModuleList() 
        self.attention_layers_lis = torch.nn.ModuleList()

        self.device = hyperparameters['device']

        for i in range(hyperparameters['num_blocks']):

            PW_feedforward_layernorm = torch.nn.LayerNorm(hyperparameters['num_hidden_units'], eps=1e-7)
            self.PW_feedforward_layernorm_lis.append(PW_feedforward_layernorm)

            PW_feedforward = PW_FeedForward(hyperparameters['num_hidden_units'], hyperparameters['dropout_val'])
            self.PW_feedforward_layers_lis.append(PW_feedforward)


            layernorm = torch.nn.LayerNorm(hyperparameters['num_hidden_units'], eps=1e-7)
            self.layernorm_lis.append(layernorm)

            attention_layer =  torch.nn.MultiheadAttention(hyperparameters['num_hidden_units'],
                                                            hyperparameters['num_heads'],
                                                            hyperparameters['dropout_val'])
            self.attention_layers_lis.append(attention_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_embeddings(torch.LongTensor(log_seqs).to(self.device))
        seqs = seqs * (self.item_embeddings.embedding_dim ** 0.5)
        position = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs = seqs + self.positional_embeddings(torch.LongTensor(position).to(self.device))
        seqs = self.dropout_embeddings(seqs)
        
        mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attn_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers_lis)):
            seqs = torch.transpose(seqs, 0, 1)
            out = self.layernorm_lis[i](seqs)
            outputs, _ = self.attention_layers_lis[i](out, seqs, seqs, 
                                            attn_mask=attn_mask)

            seqs = out + outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.PW_feedforward_layernorm_lis[i](seqs)
            seqs = self.PW_feedforward_layers_lis[i](seqs)
            seqs = seqs *  (~mask.unsqueeze(-1))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        features_logs = self.log2feats(log_seqs) 

        pos_embedding = self.item_embeddings(torch.LongTensor(pos_seqs).to(self.device))
        neg_embedding = self.item_embeddings(torch.LongTensor(neg_seqs).to(self.device))

        pos_outputs = (features_logs * pos_embedding).sum(dim=-1)
        neg_outputs = (features_logs * neg_embedding).sum(dim=-1)
        return pos_outputs, neg_outputs

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        features_logs = self.log2feats(log_seqs)
        features = features_logs[:, -1, :] 
        item_embedding = self.item_embeddings(torch.LongTensor(item_indices).to(self.device)) # (U, I, C)
        outputs = item_embedding.matmul(features.unsqueeze(-1)).squeeze(-1)
        return outputs