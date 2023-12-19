import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import math
from copy import deepcopy


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):
 
        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k) 
        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1)) 
            attn += dis_M_

        attn = F.softmax(attn, dim=-1) 
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v) 
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v

class ConvMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 4*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k_1, k_2, v = torch.chunk(x, 4, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn_self   = torch.matmul(q, k_1) 
        attn_others = torch.matmul(q, k_2) 

        qmask_array = qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1)
        attn = torch.where(qmask_array.unsqueeze(1), attn_self, attn_others)

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1)) 
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v

class ConvOtherMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn_self   = torch.matmul(q, k_1)
        attn_others = torch.matmul(q, k_2) 
        attn_conv   = torch.matmul(q, k) 
        
        qmask_array = qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1)
        attn = torch.where(qmask_array.unsqueeze(1), attn_self, attn_others) + attn_conv

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v        

class ConvPosMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k1_p, k2_p = torch.chunk(y, 3, dim=-1) 
        q_p     = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k1_p    = k1_p.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k2_p    = k2_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q, k_1) + torch.matmul(q_p, k1_p)
        attn_others = torch.matmul(q, k_2) + torch.matmul(q_p, k2_p) 

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        attn = attn_self + attn_others

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      

class ConvPosMultiHeadAttn_rep(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, q_1, q_2, k, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        q_1 = q_1.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        q_2 = q_2.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)   
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q1_p, q2_p, k_p = torch.chunk(y, 3, dim=-1) 
        q1_p    = q1_p.view(max_len, self.n_head, -1).transpose(0, 1)
        q2_p    = q2_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k_p     = k_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q_1, k) + torch.matmul(q1_p, k_p)
        attn_others = torch.matmul(q_2, k) + torch.matmul(q2_p, k_p)  

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        attn = attn_self + attn_others

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      

class ConvPosMultiHeadAttn_rep_pos(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, q_1, q_2, k, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        q_1 = q_1.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        q_2 = q_2.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)   
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q1_p, q2_p, k_p = torch.chunk(y, 3, dim=-1) 
        q1_p    = q1_p.view(max_len, self.n_head, -1).transpose(0, 1)
        q2_p    = q2_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k_p     = k_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q_1, k)
        attn_others = torch.matmul(q_2, k)
        attn_pos    = torch.matmul(q1_p, k_p)

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)

        attn = attn_self + attn_others + attn_pos

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      

class ConvPosMultiHeadAttn_pos(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k1_p, k2_p = torch.chunk(y, 3, dim=-1) 
        q_p     = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k1_p    = k1_p.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k2_p    = k2_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q, k_1)
        attn_others = torch.matmul(q, k_2)
        attn_pos    = torch.matmul(q_p, k2_p)

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        attn = attn_self + attn_others + attn_pos

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      

class ConvPosDivMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 2*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, trans_type, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k_p = torch.chunk(y, 2, dim=-1) 
        q_p    = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k_p    = k_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn  = torch.matmul(q, k) + torch.matmul(q_p, k_p)
        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        if trans_type == 'self':
            attn.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        elif trans_type == 'other':
            attn.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        else:
            print("Error Value !")

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v     

class ConvPosMultiHeadAttn_Seg(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        """

        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.token_type_embeddings = nn.Embedding(9, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()

        x = x + self.token_type_embeddings(qmask)

        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k1_p, k2_p = torch.chunk(y, 3, dim=-1) 
        q_p     = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k1_p    = k1_p.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k2_p    = k2_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q, k_1) + torch.matmul(q_p, k1_p)
        attn_others = torch.matmul(q, k_2) + torch.matmul(q_p, k2_p)    

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        attn = attn_self + attn_others

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      


class MultiHeadAttn_Order(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)
        attn = attn/self.scale   
        mask_all = torch.triu(torch.ones_like(attn[0,0]), diagonal=1, out=None).eq(1)[None,:] + mask.eq(0)[:,None]
        attn.masked_fill_(mask=mask_all[:, None], value=float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      


class ConvPosMultiHeadAttn_Order(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.9, scale=False, batch_first=True):

        super().__init__()
        assert d_model%n_head==0

        self.n_head = n_head

        self.qkv_linear = nn.Linear(d_model, 4*d_model, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model //n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model //n_head, 3*d_model , bias=False)

        self.device = 'cpu'
        self.scale = math.sqrt(2 * d_model //n_head) if scale else 1

        self.fc = nn.Linear(d_model ,d_model, bias=False)

    def _prepare_embeds(self, embed, batch_size, max_len):
        q, k_1, k_2, v = self.qkv_linear(embed).chunk(4, dim=-1)

        q   = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v   = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        return q, k_1, k_2, v

    def _prepare_pos(self, umask, max_len):
        y = self.qk_pos(self.pos_embed(umask))
        q_p, k_p_1, k_p_2 = torch.chunk(y, 3, dim=-1) 
        q_p   = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k_p_1   = k_p_1.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k_p_2   = k_p_2.view(max_len, self.n_head, -1).permute(1, 2, 0)
        return q_p, k_p_1, k_p_2 

    def _cal_attn(self, q, k_1, k_2, q_p, k_p_1, k_p_2):

        pos_attn_1, pos_attn_2 = torch.matmul(q_p, k_p_1), torch.matmul(q_p, k_p_2)
        attn_self_list  = torch.matmul(q, k_1)+pos_attn_1
        attn_other_list = torch.matmul(q, k_2)+pos_attn_2

        return attn_self_list, attn_other_list

    def _mask_add(self, attn_self, attn_other, max_len, umask, qmask):

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))
        mask_order = torch.triu(torch.ones_like(attn_self[0,0]), diagonal=1, out=None).eq(1)[None,:] + umask.eq(0)[:,None]

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=0.0) 
        attn_other.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=0.0)
        attn_add   = (attn_self+attn_other)/self.scale
        attn_order = attn_add.masked_fill(mask=mask_order[:,None], value=float('-inf')) if not self.training else attn_add
        attn_all_list = self.dropout_layer(F.softmax(attn_order, dim=-1))

        return attn_all_list

    def _cal_v(self, attn_list, v_list, batch_size, max_len):
        v  = torch.matmul(attn_list,  v_list).transpose(1, 2).reshape(batch_size, max_len, -1)
        return self.fc(v)

    def forward(self, embed, umask, qmask, use_Gaussian=False):
        self.device = embed.device
        batch_size, max_len, _ = embed.size()

        q, k_1, k_2, v = self._prepare_embeds(embed, batch_size, max_len)
        q_p, k_p_1, k_p_2 = self._prepare_pos(umask, max_len)

        attn_self_list, attn_other_list  = self._cal_attn(q, k_1, k_2, q_p, k_p_1, k_p_2)
        attn_all_list = self._mask_add(attn_self_list, attn_other_list, max_len, umask, qmask)

        v = self._cal_v(attn_all_list, v, batch_size, max_len)

        return v


class ConvPosMultiHeadAttn_SELF(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k1_p, k2_p = torch.chunk(y, 3, dim=-1) 
        q_p     = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k1_p    = k1_p.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k2_p    = k2_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q, k_1) + torch.matmul(q_p, k1_p)
        attn_others = torch.matmul(q, k_2) + torch.matmul(q_p, k2_p)

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        attn_self.masked_fill_(mask=qmask_array[:, None].eq(0.0), value=float('-inf'))
        attn = attn_self

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      


class ConvPosMultiHeadAttn_OTHER(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):

        super().__init__()
        assert d_model%n_head==0

        self.shift = nn.Parameter( torch.abs(torch.randn(1)) + 0.001)
        self.bias  = nn.Parameter(-torch.abs(torch.randn(1)))

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 5*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)
        self.qk_pos = nn.Linear(d_model//n_head, 3*d_model, bias=False)

        self.device = 'cpu'

        if scale:
            self.scale = math.sqrt(2 * d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask, qmask, use_Gaussian=False):

        self.device = x.device

        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, k_1, k_2, v = torch.chunk(x, 5, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_1 = k_1.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        k_2 = k_2.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        y = self.qk_pos(self.pos_embed(mask))
        q_p, k1_p, k2_p = torch.chunk(y, 3, dim=-1) 
        q_p     = q_p.view(max_len, self.n_head, -1).transpose(0, 1)
        k1_p    = k1_p.view(max_len, self.n_head, -1).permute(1, 2, 0)
        k2_p    = k2_p.view(max_len, self.n_head, -1).permute(1, 2, 0)

        attn_self   = torch.matmul(q, k_1) + torch.matmul(q_p, k1_p)
        attn_others = torch.matmul(q, k_2) + torch.matmul(q_p, k2_p)   

        qmask_array = 1.0*(qmask.unsqueeze(-1).repeat(1, 1, max_len) == qmask.unsqueeze(-2).repeat(1, max_len, 1))

        eye_tmp = torch.eye(max_len)[None, None, :, :].to(self.device)
        attn_others.masked_fill_(mask=qmask_array[:, None].eq(1.0), value=float('-inf'))
        attn_others.masked_fill_(mask=eye_tmp.eq(1.0), value=0)
        attn = attn_others

        attn = attn/self.scale        
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_

        attn = F.softmax(attn, dim=-1)
        
        attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v      


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout, r_w_bias=None, r_r_bias=None, scale=False):

        super().__init__()

        self.shift = nn.Parameter( torch.randn(1) + 0.001)
        self.bias  = nn.Parameter(-torch.randn(1))

        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(d_model//n_head, 0, 1200)

        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(n_head, d_model // n_head)))
        else:
            self.r_r_bias = r_r_bias 
            self.r_w_bias = r_w_bias

    def forward(self, x, mask, use_Gaussian=False):

        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)

        qkv = self.qkv_linear(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)
        E_ = torch.einsum('bnqd,ld->bnql', k, pos_embed)
        BD = B_ + D_  
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE

        attn = attn / self.scale

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        if use_Gaussian:
            square_distance = compute_squared_EDM_method4(max_len).to(q.device)
            shift_M = self.shift * torch.ones(max_len, max_len).to(q.device)
            bias_M =  self.bias  * torch.ones(max_len, max_len).to(q.device)
            dis_M = -(shift_M * square_distance + bias_M)
            dis_M_ = dis_M.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.n_head, 1, 1))
            attn += dis_M_    

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)

        return v

    def _shift(self, BD):

        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)
        BD = BD[:, :, :, max_len:]
        return BD
    
    def _transpose_shift(self, E):

        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)

        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len)*2+1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1,-2)

        return E

class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(feedforward_dim, d_model),
                                nn.Dropout(dropout))

    def forward(self, x, mask, qmask, use_Gaussian=False):

        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask, qmask, use_Gaussian)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                scale=False, dropout_attn=None, pos_embed=None, batch_first=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model
        self.batch_first = batch_first

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)


        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'transformer_order':
            self_attn = MultiHeadAttn_Order(d_model, n_head, dropout_attn, scale=scale)            
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'convtrans':
            self_attn = ConvMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'convother':
            self_attn = ConvOtherMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'convpos':
            self_attn = ConvPosMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'convpos_seg':
            self_attn = ConvPosMultiHeadAttn_Seg(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'convpos_order':
            self_attn = ConvPosMultiHeadAttn_Order(d_model, n_head, dropout_attn, scale=scale)  
        elif attn_type == 'convpos_self':
            self_attn = ConvPosMultiHeadAttn_SELF(d_model, n_head, dropout_attn, scale=scale)                      
        elif attn_type == 'convpos_other':
            self_attn = ConvPosMultiHeadAttn_OTHER(d_model, n_head, dropout_attn, scale=scale)               
        else:
            raise ValueError('attn_type not supported: {}'.format(attn_type))

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                        for _ in range(num_layers)])

    def forward(self, x, mask, qmask, use_Gaussian=False):

        if not self.batch_first:
            x     = x.transpose(0,1)
            mask  = mask.transpose(0,1)
            qmask = qmask.transpose(0,1)

        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask, qmask, use_Gaussian)
        
        return x if self.batch_first else x.transpose(0,1)


def make_positions(tensor, padding_idx):

    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx

def compute_squared_EDM_method4(n):
    X = np.expand_dims(np.arange(n), -1)
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (n,1))
    return torch.FloatTensor(H + H.T - 2*G)

class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:

            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):

        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):

            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):

        return int(1e5)

class LearnedPositionalEmbedding(nn.Embedding):

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):

        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)

class RelativeEmbedding(nn.Module):
    def forward(self, input):

        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:

            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(int(-seq_len/2), round(seq_len/2 + 1e-5)).to(input.device).long() + self.origin_shift
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed

class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):

    def __init__(self, embedding_dim, padding_idx, init_size=1568):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        weights = self.get_embedding(
            init_size+1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb

