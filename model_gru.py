import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Optional, overload
import math


class ConvRNNCellBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: torch.Tensor
    weight_hh: torch.Tensor

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks_x: int, num_chunks_y: int) -> None:
        super(ConvRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks_x * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks_y * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks_x * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks_y * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: torch.Tensor) -> None:
        if input.size(-1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(-1), self.input_size))

    def check_forward_hidden(self, input: torch.Tensor, hx: torch.Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(-1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(-1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for name, weight in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(weight, gain=1.0)           
            else:
                nn.init.uniform_(weight, -stdv, stdv)        


class ConvGRUCell(ConvRNNCellBase):

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(ConvGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks_x=4, num_chunks_y=6)

    def _conv_gru_cell(self, x: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor):

        W_ir, W_is, W_iz, W_il = torch.chunk(self.weight_ih, chunks=4, dim=0)
        b_ir, b_is, b_iz, b_il = torch.chunk(self.bias_ih  , chunks=4, dim=0)

        W_hr, W_hs, W_hrz, W_hsz, W_hn, W_hm = torch.chunk(self.weight_hh, chunks=6, dim=0)
        b_hr, b_hs, b_hrz, b_hsz, b_hn, b_hm = torch.chunk(self.bias_hh  , chunks=6, dim=0)

        r = torch.sigmoid(torch.matmul(x, W_ir.T)+b_ir + torch.matmul(hy, W_hr.T)+b_hr)
        s = torch.sigmoid(torch.matmul(x, W_is.T)+b_is + torch.matmul(hx, W_hs.T)+b_hs)

        z = torch.sigmoid(torch.matmul(x, W_iz.T)+b_iz + torch.matmul(hy, W_hrz.T)+b_hrz + torch.matmul(hx, W_hsz.T)+b_hsz)
        n = torch.tanh(torch.matmul(x, W_il.T)+b_il + r * (torch.matmul(hy, W_hn.T)+b_hn) + s * (torch.matmul(hx, W_hm.T)+b_hm))

        h_ = (1-z)*n + z*hx

        return h_

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None, hy: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        if hy is None:
            hy = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        self.check_forward_hidden(input, hy, '')
        return self._conv_gru_cell(
            input, hx, hy
        )


class ConvGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False) -> None: 
        super(ConvGRU, self).__init__()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bias           = bias
        self.batch_first    = batch_first
        self.dropout    = dropout
        self.bidirectional  = bidirectional

        self.device = 'cpu'

        self.layers = nn.ModuleList([ConvGRUCell(input_size, hidden_size, bias)] + [ConvGRUCell(hidden_size, hidden_size, bias)
                                                                                    for _ in range(num_layers-1)])
        if self.bidirectional:
            self.layers_r = nn.ModuleList([ConvGRUCell(input_size, hidden_size, bias)] + [ConvGRUCell(hidden_size, hidden_size, bias)
                                                                                            for _ in range(num_layers-1)])
        else:
            self.register_parameter('layers_r', None)

        self.dropout_layer = nn.Dropout(self.dropout)


    def _reverse(self, input: torch.Tensor, umask: torch.Tensor):

        conv_len = torch.sum(umask, dim=0)
        batch_size = input.size(1)
        input_ret = torch.zeros_like(input).to(self.device)

        for i in range(batch_size):
            input_ret[torch.arange(conv_len[i]-1, -1, -1), i] = input[torch.arange(conv_len[i]), i]
            
        return input_ret

    def _bulid_conv_id(self, qmask: torch.Tensor, umask: torch.Tensor)  -> torch.Tensor:
        max_p = torch.max(qmask).item()
        min_p = torch.min(qmask).item()
        assert min_p >= 0

        seq_len, batch_size = qmask.size()
        mem_near = (-1) * torch.ones(batch_size, max_p+1, dtype=torch.int8).to(self.device)
        conv_id  = torch.zeros_like(qmask).to(self.device)
        batch_id = torch.arange(batch_size)

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask[i]]
            mem_near[batch_id, qmask[i]] = i

        conv_id += 1
        return conv_id * umask

    def _build_conv_influ(self, qmask: torch.Tensor, 
                                umask: torch.Tensor,
                                conv_id: torch.Tensor,
                                self_shift = 3 , self_scale = 1,
                                other_shift = 0, other_scale = 2)  -> torch.Tensor:

        def conv_sigmoid(distance: torch.Tensor, shift=0, scale=1, pos=True) -> torch.Tensor:
            reversal = 1.0 if pos else -1.0
            return 1/(1+torch.exp(reversal*(-distance+shift)/scale))

        seq_len, batch_size = qmask.size()
        seq_range = torch.arange(seq_len).to(qmask.device)
        zeros_pad = torch.zeros(batch_size, dtype=torch.long).to(qmask.device)

        self_influ_scale  = (seq_range[:,None] - conv_id) + 1e2*(conv_id == 0)
        other_influ_scale = torch.zeros_like(qmask)

        conv_len, last_conv = zeros_pad, -1*torch.ones_like(qmask[0])

        for i in range(1, seq_len):

            other_influ_scale[i] = torch.where(qmask[i] == last_conv, conv_len+1, zeros_pad)

            conv_len = torch.where(qmask[i] == qmask[i-1], conv_len, other_influ_scale[i])

            last_conv = torch.where(qmask[i] != qmask[i-1], qmask[i-1], last_conv)
        
        return torch.stack((conv_sigmoid(self_influ_scale, pos=False, shift = self_shift,  scale = self_scale),
                            conv_sigmoid(other_influ_scale, pos=True, shift = other_shift, scale = other_scale))) * umask[None, :, :]

    def _compute_grucell(self, layer_id: int, 
                                input: torch.Tensor, 
                                conv_id: torch.Tensor, 
                                conv_inertia: torch.Tensor, 
                                conv_contagion: torch.Tensor, 
                                umask: torch.Tensor, 
                                influ_scale: torch.Tensor, 
                                bidirectional: bool = False) -> torch.Tensor:

        assert 0 <= layer_id < self.num_layers
        assert input.size(-1) == self.layers[layer_id].input_size

        seq_len, batch_size = conv_id.size()
        h_ = [torch.zeros(batch_size, self.hidden_size).to(self.device)]

        for j in range(seq_len):
            umask_j = umask[j].unsqueeze(1)
            hx      = umask_j * torch.cat([h_[k][i].unsqueeze(0) for i, k in enumerate(conv_id[j])]) * (conv_inertia[j])[:, None]
            hy      = umask_j * h_[-1]  * (conv_contagion[j])[:, None]
            h_t     = self.layers[layer_id](umask_j*input[j], hx*influ_scale[0, j][:,None], hy*influ_scale[1, j][:,None]) if not bidirectional else self.layers_r[layer_id](umask_j*input[j], hx*influ_scale[0, j][:,None], hy*influ_scale[1, j][:,None]) 
            h_.append(h_t)
        h_.pop(0)

        return torch.cat([h_t.unsqueeze(0) for h_t in h_])

    def forward(self, input: torch.FloatTensor, qmask: torch.LongTensor, umask: torch.LongTensor):

        assert input.size()[:-1] == qmask.size() == umask.size()

        self.device = input.device
        seq_length, batch_size = umask.shape
        seq_range = torch.arange(seq_length).to(self.device)

        if self.batch_first:
            h_      = input.trnaspose(0,1)
            qmask   = qmask.trnaspose(0,1)
            umask   = umask.transpose(0,1)
        else:
            h_ = input

        conv_id = self._bulid_conv_id(qmask, umask)
        conv_bool = (seq_range[1:][:, None] != conv_id[1:]).T * umask[1:].T
        conv_end = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool]
        conv_conti = []
        for idx, conv_e in enumerate(conv_end):
            if conv_e.dim() == 0:
                conv_conti.append(torch.ones(1, dtype=torch.bool))
            else:
                conv_conti.append(qmask[conv_e[:-1], idx] == qmask[conv_e[1:]+1, idx])
        conv_inertia   = conv_id > 0
        for idx in range(batch_size):
            conv_inertia[conv_end[idx][1:]+1, idx] = conv_conti[idx]
        conv_contagion = conv_id != seq_range[:,None]

        influ_scale = self._build_conv_influ(qmask, umask, conv_id)

        for i in range(self.num_layers):
            h_ = self._compute_grucell(i, h_, conv_id, conv_inertia, conv_contagion, umask, influ_scale)
            if i+1 != self.num_layers:
                h_ = self.dropout_layer(h_)

        if self.bidirectional:
            h_r     = self._reverse(input, umask)
            qmask_r = self._reverse(qmask, umask)

            conv_id_r = self._bulid_conv_id(qmask_r, umask)
            conv_bool_r = (seq_range[1:][:, None] != conv_id_r[1:]).T * umask[1:].T
            conv_end_r = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool_r]
            conv_conti_r = []
            for idx, conv_e in enumerate(conv_end_r):
                if conv_e.dim() == 0:
                    conv_conti_r.append(torch.ones(1, dtype=torch.bool))
                else:
                    conv_conti_r.append(qmask_r[conv_e[:-1], idx] == qmask_r[conv_e[1:]+1, idx])
            conv_inertia_r   = conv_id_r > 0
            for idx in range(batch_size):
                conv_inertia_r[conv_end_r[idx][1:]+1, idx] = conv_conti_r[idx]
            conv_contagion_r = conv_id_r != seq_range[:,None]

            influ_scale_r = self._build_conv_influ(qmask_r, umask, conv_id_r)

            for i in range(self.num_layers):
                h_r = self._compute_grucell(i, h_r, conv_id_r, conv_inertia_r, conv_contagion_r, umask, influ_scale_r, True)
                if i+1 != self.num_layers:
                    h_r = self.dropout_layer(h_r)
            h_r = self._reverse(h_r, umask)
            h_  = torch.cat([h_, h_r], dim=-1)
        
        return h_ if not self.batch_first else h_.transpose(0,1)

class ConvGRU_self(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False) -> None: 
        super(ConvGRU_self, self).__init__()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bias           = bias
        self.batch_first    = batch_first
        self.dropout    = dropout
        self.bidirectional  = bidirectional

        self.device = 'cpu'

        self.layers = nn.ModuleList([ConvGRUCell(input_size, hidden_size, bias)] + [ConvGRUCell(hidden_size, hidden_size, bias)
                                                                                    for _ in range(num_layers-1)])
        if self.bidirectional:
            self.layers_r = nn.ModuleList([ConvGRUCell(input_size, hidden_size, bias)] + [ConvGRUCell(hidden_size, hidden_size, bias)
                                                                                            for _ in range(num_layers-1)])
        else:
            self.register_parameter('layers_r', None)

        self.dropout_layer = nn.Dropout(self.dropout)


    def _reverse(self, input: torch.Tensor, umask: torch.Tensor):

        conv_len = torch.sum(umask, dim=0)
        batch_size = input.size(1)
        input_ret = torch.zeros_like(input).to(self.device)

        for i in range(batch_size):
            input_ret[torch.arange(conv_len[i]-1, -1, -1), i] = input[torch.arange(conv_len[i]), i]
            
        return input_ret

    def _bulid_conv_id(self, qmask: torch.Tensor, umask: torch.Tensor)  -> torch.Tensor:
        max_p = torch.max(qmask).item()
        min_p = torch.min(qmask).item()
        assert min_p >= 0

        seq_len, batch_size = qmask.size()
        mem_near = (-1) * torch.ones(batch_size, max_p+1, dtype=torch.int8).to(self.device)
        conv_id  = torch.zeros_like(qmask).to(self.device)
        batch_id = torch.arange(batch_size)

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask[i]]
            mem_near[batch_id, qmask[i]] = i

        conv_id += 1
        return conv_id * umask

    def _build_conv_influ(self, qmask: torch.Tensor, 
                                umask: torch.Tensor,
                                conv_id: torch.Tensor,
                                self_shift = 3 , self_scale = 1,
                                other_shift = 0, other_scale = 2)  -> torch.Tensor:

        def conv_sigmoid(distance: torch.Tensor, shift=0, scale=1, pos=True) -> torch.Tensor:
            reversal = 1.0 if pos else -1.0
            return 1/(1+torch.exp(reversal*(-distance+shift)/scale))

        seq_len, batch_size = qmask.size()
        seq_range = torch.arange(seq_len).to(qmask.device)
        zeros_pad = torch.zeros(batch_size, dtype=torch.long).to(qmask.device)

        self_influ_scale  = (seq_range[:,None] - conv_id) + 1e2*(conv_id == 0)
        other_influ_scale = torch.zeros_like(qmask)

        conv_len, last_conv = zeros_pad, -1*torch.ones_like(qmask[0])

        for i in range(1, seq_len):

            other_influ_scale[i] = torch.where(qmask[i] == last_conv, conv_len+1, zeros_pad)

            conv_len = torch.where(qmask[i] == qmask[i-1], conv_len, other_influ_scale[i])

            last_conv = torch.where(qmask[i] != qmask[i-1], qmask[i-1], last_conv)
        
        return torch.stack((conv_sigmoid(self_influ_scale, pos=False, shift = self_shift,  scale = self_scale),
                            conv_sigmoid(other_influ_scale, pos=True, shift = other_shift, scale = other_scale))) * umask[None, :, :]

    def _compute_grucell(self, layer_id: int, 
                                input: torch.Tensor, 
                                conv_id: torch.Tensor, 
                                conv_inertia: torch.Tensor, 
                                conv_contagion: torch.Tensor, 
                                umask: torch.Tensor, 
                                influ_scale: torch.Tensor, 
                                bidirectional: bool = False) -> torch.Tensor:

        assert 0 <= layer_id < self.num_layers
        assert input.size(-1) == self.layers[layer_id].input_size

        seq_len, batch_size = conv_id.size()
        h_ = [torch.zeros(batch_size, self.hidden_size).to(self.device)]

        for j in range(seq_len):
            umask_j = umask[j].unsqueeze(1)
            hx      = umask_j * torch.cat([h_[k][i].unsqueeze(0) for i, k in enumerate(conv_id[j])]) * (conv_inertia[j])[:, None]
            hy      = umask_j * h_[-1]  * (conv_contagion[j])[:, None]
            h_t     = self.layers[layer_id](umask_j*input[j], hx*influ_scale[0, j][:,None], 0*hy*influ_scale[1, j][:,None]) if not bidirectional else self.layers_r[layer_id](umask_j*input[j], hx*influ_scale[0, j][:,None], 0*hy*influ_scale[1, j][:,None]) 
            h_.append(h_t)
        h_.pop(0)

        return torch.cat([h_t.unsqueeze(0) for h_t in h_])

    def forward(self, input: torch.FloatTensor, qmask: torch.LongTensor, umask: torch.LongTensor):

        assert input.size()[:-1] == qmask.size() == umask.size()

        self.device = input.device
        seq_length, batch_size = umask.shape
        seq_range = torch.arange(seq_length).to(self.device)

        if self.batch_first:
            h_      = input.trnaspose(0,1)
            qmask   = qmask.trnaspose(0,1)
            umask   = umask.transpose(0,1)
        else:
            h_ = input

        conv_id = self._bulid_conv_id(qmask, umask)
        conv_bool = (seq_range[1:][:, None] != conv_id[1:]).T * umask[1:].T
        conv_end = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool]
        conv_conti = []
        for idx, conv_e in enumerate(conv_end):
            if conv_e.dim() == 0:
                conv_conti.append(torch.ones(1, dtype=torch.bool))
            else:
                conv_conti.append(qmask[conv_e[:-1], idx] == qmask[conv_e[1:]+1, idx])
        conv_inertia   = conv_id > 0
        for idx in range(batch_size):
            conv_inertia[conv_end[idx][1:]+1, idx] = conv_conti[idx]
        conv_contagion = conv_id != seq_range[:,None]


        influ_scale = self._build_conv_influ(qmask, umask, conv_id)

        for i in range(self.num_layers):
            h_ = self._compute_grucell(i, h_, conv_id, conv_inertia, conv_contagion, umask, influ_scale)
            if i+1 != self.num_layers:
                h_ = self.dropout_layer(h_)

        if self.bidirectional:
            h_r     = self._reverse(input, umask)
            qmask_r = self._reverse(qmask, umask)

            conv_id_r = self._bulid_conv_id(qmask_r, umask)
            conv_bool_r = (seq_range[1:][:, None] != conv_id_r[1:]).T * umask[1:].T
            conv_end_r = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool_r]
            conv_conti_r = []
            for idx, conv_e in enumerate(conv_end_r):
                if conv_e.dim() == 0:
                    conv_conti_r.append(torch.ones(1, dtype=torch.bool))
                else:
                    conv_conti_r.append(qmask_r[conv_e[:-1], idx] == qmask_r[conv_e[1:]+1, idx])
            conv_inertia_r   = conv_id_r > 0
            for idx in range(batch_size):
                conv_inertia_r[conv_end_r[idx][1:]+1, idx] = conv_conti_r[idx]
            conv_contagion_r = conv_id_r != seq_range[:,None]

            influ_scale_r = self._build_conv_influ(qmask_r, umask, conv_id_r)

            for i in range(self.num_layers):
                h_r = self._compute_grucell(i, h_r, conv_id_r, conv_inertia_r, conv_contagion_r, umask, influ_scale_r, True)
                if i+1 != self.num_layers:
                    h_r = self.dropout_layer(h_r)
            h_r = self._reverse(h_r, umask)
            h_  = torch.cat([h_, h_r], dim=-1)
        
        return h_ if not self.batch_first else h_.transpose(0,1)

class ConvGRU_other(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False) -> None: 
        super(ConvGRU_other, self).__init__()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.bias           = bias
        self.batch_first    = batch_first
        self.dropout    = dropout
        self.bidirectional  = bidirectional

        self.device = 'cpu'

        self.layers = nn.ModuleList([ConvGRUCell(input_size, hidden_size, bias)] + [ConvGRUCell(hidden_size, hidden_size, bias)
                                                                                    for _ in range(num_layers-1)])
        if self.bidirectional:
            self.layers_r = nn.ModuleList([ConvGRUCell(input_size, hidden_size, bias)] + [ConvGRUCell(hidden_size, hidden_size, bias)
                                                                                            for _ in range(num_layers-1)])
        else:
            self.register_parameter('layers_r', None)

        self.dropout_layer = nn.Dropout(self.dropout)


    def _reverse(self, input: torch.Tensor, umask: torch.Tensor):

        conv_len = torch.sum(umask, dim=0)
        batch_size = input.size(1)
        input_ret = torch.zeros_like(input).to(self.device)

        for i in range(batch_size):
            input_ret[torch.arange(conv_len[i]-1, -1, -1), i] = input[torch.arange(conv_len[i]), i]
            
        return input_ret

    def _bulid_conv_id(self, qmask: torch.Tensor, umask: torch.Tensor)  -> torch.Tensor:
        max_p = torch.max(qmask).item()
        min_p = torch.min(qmask).item()
        assert min_p >= 0

        seq_len, batch_size = qmask.size()
        mem_near = (-1) * torch.ones(batch_size, max_p+1, dtype=torch.int8).to(self.device)
        conv_id  = torch.zeros_like(qmask).to(self.device)
        batch_id = torch.arange(batch_size)

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask[i]]
            mem_near[batch_id, qmask[i]] = i

        conv_id += 1
        return conv_id * umask

    def _build_conv_influ(self, qmask: torch.Tensor, 
                                umask: torch.Tensor,
                                conv_id: torch.Tensor,
                                self_shift = 3 , self_scale = 1,
                                other_shift = 0, other_scale = 2)  -> torch.Tensor:

        def conv_sigmoid(distance: torch.Tensor, shift=0, scale=1, pos=True) -> torch.Tensor:
            reversal = 1.0 if pos else -1.0
            return 1/(1+torch.exp(reversal*(-distance+shift)/scale))

        seq_len, batch_size = qmask.size()
        seq_range = torch.arange(seq_len).to(qmask.device)
        zeros_pad = torch.zeros(batch_size, dtype=torch.long).to(qmask.device)

        self_influ_scale  = (seq_range[:,None] - conv_id) + 1e2*(conv_id == 0)
        other_influ_scale = torch.zeros_like(qmask)

        conv_len, last_conv = zeros_pad, -1*torch.ones_like(qmask[0])

        for i in range(1, seq_len):

            other_influ_scale[i] = torch.where(qmask[i] == last_conv, conv_len+1, zeros_pad)

            conv_len = torch.where(qmask[i] == qmask[i-1], conv_len, other_influ_scale[i])

            last_conv = torch.where(qmask[i] != qmask[i-1], qmask[i-1], last_conv)
        
        return torch.stack((conv_sigmoid(self_influ_scale, pos=False, shift = self_shift,  scale = self_scale),
                            conv_sigmoid(other_influ_scale, pos=True, shift = other_shift, scale = other_scale))) * umask[None, :, :]

    def _compute_grucell(self, layer_id: int, 
                                input: torch.Tensor, 
                                conv_id: torch.Tensor, 
                                conv_inertia: torch.Tensor, 
                                conv_contagion: torch.Tensor, 
                                umask: torch.Tensor, 
                                influ_scale: torch.Tensor, 
                                bidirectional: bool = False) -> torch.Tensor:

        assert 0 <= layer_id < self.num_layers
        assert input.size(-1) == self.layers[layer_id].input_size

        seq_len, batch_size = conv_id.size()
        h_ = [torch.zeros(batch_size, self.hidden_size).to(self.device)]

        for j in range(seq_len):
            umask_j = umask[j].unsqueeze(1)
            hx      = umask_j * torch.cat([h_[k][i].unsqueeze(0) for i, k in enumerate(conv_id[j])]) * (conv_inertia[j])[:, None]
            hy      = umask_j * h_[-1]  * (conv_contagion[j])[:, None]
            h_t     = self.layers[layer_id](umask_j*input[j], (1e-5)*hx*influ_scale[0, j][:,None], hy*influ_scale[1, j][:,None]) if not bidirectional else self.layers_r[layer_id](umask_j*input[j], (1e-5)*hx*influ_scale[0, j][:,None], hy*influ_scale[1, j][:,None]) 
            h_.append(h_t)
        h_.pop(0)

        return torch.cat([h_t.unsqueeze(0) for h_t in h_])

    def forward(self, input: torch.FloatTensor, qmask: torch.LongTensor, umask: torch.LongTensor):

        assert input.size()[:-1] == qmask.size() == umask.size()

        self.device = input.device
        seq_length, batch_size = umask.shape
        seq_range = torch.arange(seq_length).to(self.device)

        if self.batch_first:
            h_      = input.trnaspose(0,1)
            qmask   = qmask.trnaspose(0,1)
            umask   = umask.transpose(0,1)
        else:
            h_ = input

        conv_id = self._bulid_conv_id(qmask, umask)
        conv_bool = (seq_range[1:][:, None] != conv_id[1:]).T * umask[1:].T
        conv_end = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool]
        conv_conti = []
        for idx, conv_e in enumerate(conv_end):
            if conv_e.dim() == 0:
                conv_conti.append(torch.ones(1, dtype=torch.bool))
            else:
                conv_conti.append(qmask[conv_e[:-1], idx] == qmask[conv_e[1:]+1, idx])
        conv_inertia   = conv_id > 0
        for idx in range(batch_size):
            conv_inertia[conv_end[idx][1:]+1, idx] = conv_conti[idx]
        conv_contagion = conv_id != seq_range[:,None]


        influ_scale = self._build_conv_influ(qmask, umask, conv_id)

        for i in range(self.num_layers):
            h_ = self._compute_grucell(i, h_, conv_id, conv_inertia, conv_contagion, umask, influ_scale)
            if i+1 != self.num_layers:
                h_ = self.dropout_layer(h_)

        if self.bidirectional:
            h_r     = self._reverse(input, umask)
            qmask_r = self._reverse(qmask, umask)

            conv_id_r = self._bulid_conv_id(qmask_r, umask)
            conv_bool_r = (seq_range[1:][:, None] != conv_id_r[1:]).T * umask[1:].T
            conv_end_r = [torch.nonzero(conv_batch).squeeze(1) for conv_batch in conv_bool_r]
            conv_conti_r = []
            for idx, conv_e in enumerate(conv_end_r):
                if conv_e.dim() == 0:
                    conv_conti_r.append(torch.ones(1, dtype=torch.bool))
                else:
                    conv_conti_r.append(qmask_r[conv_e[:-1], idx] == qmask_r[conv_e[1:]+1, idx])
            conv_inertia_r   = conv_id_r > 0
            for idx in range(batch_size):
                conv_inertia_r[conv_end_r[idx][1:]+1, idx] = conv_conti_r[idx]
            conv_contagion_r = conv_id_r != seq_range[:,None]

            influ_scale_r = self._build_conv_influ(qmask_r, umask, conv_id_r)

            for i in range(self.num_layers):
                h_r = self._compute_grucell(i, h_r, conv_id_r, conv_inertia_r, conv_contagion_r, umask, influ_scale_r, True)
                if i+1 != self.num_layers:
                    h_r = self.dropout_layer(h_r)
            h_r = self._reverse(h_r, umask)
            h_  = torch.cat([h_, h_r], dim=-1)
        
        return h_ if not self.batch_first else h_.transpose(0,1)