from __future__ import print_function
from cProfile import label
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from pytorch_metric_learning import losses, miners
import numpy as np

from torchcrf import CRF as LCRF


class CRF(nn.Module):

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.self_transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.other_transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.one_pad = nn.Parameter(torch.log(torch.ones(num_tags, num_tags)), requires_grad=False)
        self.eye_pad = nn.Parameter(torch.log(torch.eye(num_tags)),             requires_grad=False)

        self.begin_transitions = nn.Parameter(torch.log(torch.eye(num_tags)),             requires_grad=False)

        self.begin_seg_path = nn.Parameter(torch.empty(0, self.num_tags, self.num_tags, dtype=torch.long), requires_grad=False)
        self.continual_seg_value = nn.Parameter((1 - torch.eye(self.num_tags, dtype=torch.float32)) * -1e6, requires_grad=False)        
        self.output_path = nn.Parameter(torch.arange(num_tags)[:, None].repeat(1, num_tags), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:

        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        nn.init.uniform_(self.self_transitions, -0.1, 0.1)
        nn.init.uniform_(self.other_transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            qmask: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:

        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size = mask.shape
        conv_id = self._make_convid(qmask, mask)
        seq_range = torch.arange(seq_length).to(conv_id.device)
        conv_bool = (seq_range[1:][:, None] != conv_id[1:]).T * mask[1:].T
        conv_end = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool]
        conv_conti = []
        for idx, conv_e in enumerate(conv_end):
            if conv_e.dim() == 0:
                conv_conti.append(torch.ones(1, dtype=torch.bool))
            else:
                conv_conti.append(qmask[conv_e[:-1], idx] == qmask[conv_e[1:]+1, idx])

        conv_gap        = torch.zeros_like(conv_id, dtype=torch.bool)
        conv_gap_begin  = torch.zeros_like(conv_id, dtype=torch.bool)
        for idx in range(batch_size):
            if len(conv_end[idx]) == 0:
                continue

            conv_end_gap_begin_idx  = torch.nonzero(~conv_conti[idx])
            conv_end_gap_begin      = conv_end[idx][conv_end_gap_begin_idx] + 1
            conv_end_gap_stop       = conv_end[idx][conv_end_gap_begin_idx+1]

            conv_gap_begin[conv_end_gap_begin, idx]     = True
            conv_gap_begin[conv_end[idx][-1] + 1, idx]  = True

            conv_gap[1 : conv_end[idx][0]  + 1, idx] = True
            for s_i, e_i in zip(conv_end_gap_begin, conv_end_gap_stop):
                conv_gap[s_i + 1 : e_i + 1, idx] = True
            conv_gap[conv_end[idx][-1] + 2 : , idx] = True

        conv_inertia   = conv_id > 0
        for idx in range(batch_size):
            conv_inertia[conv_end[idx][1:]+1, idx] = conv_conti[idx]
        conv_contagion = conv_id != seq_range[:,None]


        numerator = self._compute_score(emissions, tags, mask, conv_id, conv_inertia, conv_contagion)

        denominator = self._compute_normalizer(emissions, mask, conv_id, conv_inertia, conv_contagion, conv_gap_begin, conv_gap)

        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
                qmask: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:

        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size = mask.shape
        conv_id = self._make_convid(qmask, mask)
        seq_range = torch.arange(seq_length).to(conv_id.device)
        conv_bool = (seq_range[1:][:, None] != conv_id[1:]).T * mask[1:].T
        conv_end = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool]

        conv_conti = []
        for idx, conv_e in enumerate(conv_end):
            if conv_e.dim() == 0:
                conv_conti.append(torch.ones(1, dtype=torch.bool))
            else:
                conv_conti.append(qmask[conv_e[:-1], idx] == qmask[conv_e[1:]+1, idx])        

        path_list, _ = self._viterbi_decode(emissions, mask, conv_id, conv_conti, conv_end)

        return path_list

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:            
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, 
            tags: torch.LongTensor,
            mask: torch.ByteTensor, 
            conv_id: torch.LongTensor,
            conv_inertia: torch.BoolTensor,
            conv_contagion: torch.BoolTensor) -> torch.Tensor:

        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        batch_range = torch.arange(batch_size)

        score = self.start_transitions[tags[0]]
        score +=emissions[0, batch_range, tags[0]]

        for i in range(1, seq_length):

            self_score  = self.self_transitions[tags[conv_id[i]-1, batch_range], tags[i]] * conv_inertia[i]
            other_score = self.other_transitions[tags[i - 1], tags[i]] * conv_contagion[i]
            score += (self_score + other_score) * mask[i]

            score += emissions[i, batch_range, tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1

        last_tags = tags[seq_ends, batch_range]

        score += self.end_transitions[last_tags]        

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, 
            mask: torch.ByteTensor, 
            conv_id: torch.LongTensor,
            conv_inertia: torch.BoolTensor, 
            conv_contagion: torch.BoolTensor,
            conv_gap_begin: torch.BoolTensor,
            conv_gap: torch.BoolTensor,
            ) -> torch.Tensor:

        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size  = mask.shape
        batch_range = torch.arange(batch_size)

        seq_ends = mask.long().sum(dim=0) - 1

        A0 = (self.start_transitions + emissions[0])[:,:,None].repeat(1, 1,  self.num_tags)
        A =  torch.where((conv_id[1] == 0)[:,None, None], 
                        self.other_transitions[None, :, :] + A0,
            torch.logsumexp(self.self_transitions[None, :, :, None] + A0[:,:,None,:], dim=1))

        for i in range(1, seq_length-1):

            A_next_1 = torch.logsumexp( A.transpose(1,2)[:, :, None, :] + 
                            torch.where(~conv_gap_begin[i][:, None, None],
                            torch.where(~conv_gap[i][:, None, None],
                                        self.self_transitions.T[None,:,:],
                                        self.eye_pad[None,:,:]),
                                        self.one_pad[None,:,:])[:, None, :, :],
                                        dim=3)

            A_next_2 = torch.where((conv_contagion[i])[:, None, None],
                                    A_next_1,
                                    A_next_1.transpose(1,2))

            A_next_3 = A_next_2 + emissions[i][:, :, None]

            A_next_4 = A_next_3 + torch.where((conv_contagion[i+1])[:,None,None], 
                                        self.other_transitions[None, :, :], 
                                        self.one_pad[None, :, :])

            A_next_5 = torch.logsumexp(torch.where(conv_gap[i+1][:,None,None],
                                    self.self_transitions.T[None, :, :],
                                                self.eye_pad[None, :, :])[:, :, None, :]
                                                + A_next_4.transpose(1,2)[:, None, :, :],
                                                dim = 3)

            A = torch.where((seq_ends <= i)[:, None, None], A, A_next_5)

        A_end = torch.where(conv_contagion[seq_ends, batch_range][:,None,None],
                            torch.logsumexp(A.transpose(1,2)[:,:,None,:]+ self.one_pad[None,None,:,:], dim=3),
                            torch.logsumexp(A[:,:,None,:]               + self.eye_pad[None,None,:,:], dim=3))
        A_end = (A_end + self.end_transitions[None, :, None] + emissions[seq_ends, batch_range][:, :, None])[:, :, 0]

        return torch.logsumexp(A_end, dim=1)
        
    def _viterbi_decode(self,
                        emissions: torch.FloatTensor,
                        mask: torch.ByteTensor, 
                        conv_id: torch.LongTensor,
                        conv_conti,
                        conv_end) -> (List[List[int]], List[List[float]]):
   
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        best_tags_list = []
        best_score_list = []

        seq_len = mask.sum(0) - 1

        for idx in range(batch_size):
            if len(conv_end[idx]) == 0:
                if seq_len[idx] == 0:
                    broadcast_score = self.start_transitions[None,:] + emissions[0:1, idx] + self.end_transitions[None,:]
                    max_index = torch.argmax(broadcast_score)
                    best_tags_list.append([max_index.cpu().long().item()])
                    best_score_list.append(torch.max(broadcast_score).clone().detach())                    
                else:
                    continual_seg_value, continual_path_indices = self.__continual_viterbi(emissions[:seq_len[idx], idx], start_e=True)
                    broadcast_score = continual_seg_value + emissions[seq_len[idx]:seq_len[idx]+1, idx] + self.end_transitions[None,:]
                    max_index = torch.argmax(broadcast_score)

                    endp_row, endp_col = torch.div(max_index, self.num_tags, rounding_mode='trunc'), max_index%self.num_tags
                    best_tags_list.append(continual_path_indices[:, endp_row, endp_col].cpu().long().numpy().tolist() + [endp_col.cpu().item()])
                    best_score_list.append(torch.max(broadcast_score).clone().detach())
                continue


            continual_seg_value, continual_path_indices = self.__continual_viterbi(emissions[:conv_end[idx][0], idx], start_e=True)
            Val_tensor = (emissions[conv_end[idx][0]:conv_end[idx][0]+1,idx].T + self.other_transitions)[None, :, :] + \
                                                                                    continual_seg_value[:, :, None]
            broadcast_score, max_path = torch.max(Val_tensor, dim=0, keepdim=False)
            best_path = continual_path_indices[:, max_path, self.output_path]

            for i in range(1, len(conv_end[idx])):

                continual_seg_value, continual_path_indices = self.__continual_viterbi(emissions[conv_end[idx][i-1]+1:conv_end[idx][i], idx])
                broadcast_score, discontin_path_indices = self.__normal_viterbi(broadcast_score, 
                                                                                emissions[conv_end[idx][i]:conv_end[idx][i]+1,idx], 
                                                                                continual_seg_value, 
                                                                                conv_end[idx][i]-conv_end[idx][i-1]+1,
                                                                                conv_conti[idx][i-1])

                best_path_indices_ = continual_path_indices[:, discontin_path_indices[-1], self.output_path]
                best_path_indices = torch.cat([discontin_path_indices[0:1], best_path_indices_])
        
                best_path_ = best_path[:, best_path_indices[0],  discontin_path_indices[-1]]

                best_path  = torch.cat([best_path_, best_path_indices])

            continual_seg_value, continual_path_indices = self.__continual_viterbi(emissions[conv_end[idx][-1]+1:seq_len[idx], idx])
            Val_tensor = broadcast_score[:, :, None] + continual_seg_value[None, :, :]
            broadcast_score, max_path = torch.max(Val_tensor, dim=1, keepdim=False)

            best_path_ = best_path[:, self.output_path, max_path]
            best_path_indices_ = continual_path_indices[:, max_path, self.output_path.T]
            best_path_indices = torch.cat([self.output_path[None,:,:], best_path_indices_])
            best_path  = torch.cat([best_path_, best_path_indices])

            broadcast_score += emissions[seq_len[idx]:seq_len[idx]+1, idx] + self.end_transitions[None, :]
            max_index = torch.argmax(broadcast_score)

            endp_row, endp_col = torch.div(max_index, self.num_tags, rounding_mode='trunc'), max_index%self.num_tags

            best_tags_list.append(best_path[:, endp_row, endp_col].cpu().long().numpy().tolist() + [endp_col.cpu().item()])
            best_score_list.append(torch.max(broadcast_score).clone().detach())

        return best_tags_list, best_score_list

    def __normal_viterbi(self, last_potential_matrix: torch.FloatTensor,
                            emissions_one_part: torch.FloatTensor,                              
                            continual_seg_value: torch.FloatTensor,
                            len_conti = 2,
                            inertia = True) -> (torch.FloatTensor, torch.LongTensor):     
        
        assert len_conti >= 2           

        Val_tensor = (emissions_one_part.T + self.other_transitions)[:, :, None, None] + \
                                                last_potential_matrix[None, None, :, :] + \
                                        inertia*self.self_transitions.T[None, :, :, None] + \
                                                continual_seg_value.T[:, None, None, :]                                                   

        Val, indices = torch.max(Val_tensor.reshape(self.num_tags, self.num_tags, -1), dim=2, keepdim=False)

        max_path = torch.stack([torch.div(indices, self.num_tags, rounding_mode='trunc'), indices%self.num_tags])
        
        return Val, max_path

    def __continual_viterbi(self, emissions_one_part: torch.FloatTensor, start_e=False) -> (torch.FloatTensor, torch.LongTensor):      

        if start_e and len(emissions_one_part) == 0:
            return self.start_transitions[None, :].repeat(self.num_tags, 1), self.begin_seg_path
        elif len(emissions_one_part) == 0:
            return self.continual_seg_value, self.begin_seg_path

        Val = self.self_transitions + emissions_one_part[0:1].T
        if start_e:
            Val += self.start_transitions.unsqueeze(-1)

        output_path_ = self.output_path.unsqueeze(0)

        if len(emissions_one_part) == 1:
            return Val, output_path_

        for val_t in emissions_one_part[1:].unsqueeze(1):

            Val_tensor = Val.unsqueeze(1) + (self.self_transitions.T + val_t).unsqueeze(0)

            Val, max_path = torch.max(Val_tensor, dim=2)
            output_path_temp = output_path_[:, self.output_path, max_path]
            output_path_ = torch.cat([output_path_temp, max_path.unsqueeze(0)])

        return Val, output_path_

    def _make_convid(self, qmask: torch.LongTensor, umask: torch.ByteTensor):

        max_p = torch.max(qmask).item()
        min_p = torch.min(qmask).item()
        assert min_p >= 0

        seq_len, batch_size = qmask.size()
        mem_near = (-1) * torch.ones(batch_size, max_p+1, dtype=torch.int8).to(umask.device)
        conv_id  = torch.zeros_like(qmask).to(umask.device)
        batch_id = torch.arange(batch_size)

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask[i]]
            mem_near[batch_id, qmask[i]] = i

        conv_id += 1
        return conv_id * umask

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                                reduction='sum')
    def _bulid_conv_id(self, qmask: torch.Tensor, umask: torch.Tensor)  -> torch.Tensor:
        max_p = torch.max(qmask).item()
        min_p = torch.min(qmask).item()
        assert min_p >= 0

        seq_len, batch_size = qmask.size()
        mem_near = (-1) * torch.ones(batch_size, max_p+1, dtype=torch.int8)
        conv_id  = torch.zeros_like(qmask)
        batch_id = torch.arange(batch_size)

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask[i]]
            mem_near[batch_id, qmask[i]] = i

        conv_id += 1
        return conv_id * umask

    def forward(self, logits, labels, umask, qmask=None, nep=False):

        pred   = logits.transpose(0,1).contiguous().view(-1,logits.size()[2])
        target = labels.transpose(0,1).contiguous().view(-1)

        mask   = umask.transpose(0,1).contiguous().clone()
        if nep: mask[0] = False

        mask   = mask.view(-1,1)

        if type(self.weight)==type(None):
            loss = self.loss(pred*mask, target)/(torch.sum(mask) + 1e-5)
        else:
            loss = self.loss(pred*mask, target)\
                            /(torch.sum(self.weight[target]*mask.squeeze()) + 1e-5)
        return loss



class DialogueConLoss(nn.Module):
    def __init__(self, num_tags , batch_first=False, loss_weights=None):
        super(DialogueConLoss, self).__init__()

        self.CRF_loss       = CRF(num_tags , batch_first)

        self.Softmax_loss = MaskedNLLLoss(loss_weights.cuda()) if loss_weights is not None else MaskedNLLLoss()  

    def compute_opt_tags_seq(self, logits, identity_labels, umask):

        pred_ = self.CRF_loss.decode(logits, identity_labels, umask)

        len_pred = [len(p) for p in pred_]
        max_len = max(len_pred)
        pred_ = [p + [0]*(max_len - len_pred[i]) for i, p in enumerate(pred_)]

        return pred_

    def forward(self, logits_list, labels_dialog):

        logits, logits_nep = logits_list

        loss_crf  = self.CRF_loss(logits, *labels_dialog, 'token_mean') * (-1)
        
        loss_softmax     = self.Softmax_loss(logits, labels_dialog[0], labels_dialog[2])

        pred_crf     = self.compute_opt_tags_seq(logits, *labels_dialog[1:])

        lp_softmax     = logits.transpose(0,1).contiguous()
        pred_softmax    = torch.argmax(lp_softmax,-1).tolist()
        
        return [loss_crf , loss_softmax],  [pred_crf]












