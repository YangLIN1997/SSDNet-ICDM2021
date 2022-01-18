''' Define the model '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.Layers import EncoderLayer, DecoderLayer, SSMLayer
import torch.nn.functional as F
import math


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, _ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=50):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, device):
        position = torch.HalfTensor(np.zeros((x.shape[0],x.shape[1],self.pos_table.shape[2]))).to(device)

        position = position + self.pos_table[:, :x.size(1)].clone().detach()
        return torch.cat( (x, position),dim=2)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,id_enc, d_embedding, n_layers, n_head, d_k, d_v,
            d_model, d_inner, L=20, S=20, dropout=0.1, n_position=200, device='cpu'):

        super().__init__()

        self.d_embedding = d_embedding
        self.id_enc = id_enc
        self.position_enc = PositionalEncoding(d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device = device

    def forward(self, id, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        # -- Forward
        enc_output = self.position_enc(src_seq, self.device)        

        if id is not None:
            enc_output[:,:,-self.d_embedding:]+=self.id_enc(id)
        enc_output = self.dropout(enc_output)

        src_mask = None

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,id_enc, d_embedding, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, L=20, S=20, dropout=0.1, device='cpu'):

        super().__init__()
        # self.trg_word_emb = nn.Embedding(115, 20, padding_idx=0)
        self.d_embedding = d_embedding
        self.id_enc = id_enc
        self.position_enc = PositionalEncoding(d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device = device
        self.ssm = SSMLayer(d_model, d_inner, L, S, dropout=dropout,device=device)

    def forward(self, id, trg_seq, trg_mask, enc_output, y_hat, alpha, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.position_enc(trg_seq, self.device)
        if id is not None:
            dec_output[:,:,-self.d_embedding:]+=self.id_enc(id)
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)
        y_hat, alpha, Sigma = self.ssm(dec_output,alpha_0=0)

        if return_attns:
            return dec_output, y_hat, alpha[:,1:], Sigma, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, y_hat, alpha[:,1:], Sigma


class SSDNet(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, params, dataset):

        super().__init__()

        self.params = params
        self.dataset = dataset
        self.id_enc = None
        if self.dataset != 'Sanyo' and self.dataset != 'Hanergy':
            self.id_enc = torch.nn.Embedding(params.n_id, params.d_embedding, padding_idx=0)

        self.encoder = Encoder(
            id_enc=self.id_enc,n_position=params.n_position,
            d_embedding=params.d_embedding, d_model=params.d_model, d_inner=params.d_inner,
            n_layers=params.n_layers, n_head=params.n_head, d_k=params.d_k, d_v=params.d_v,
            L=params.predict_start,S=params.S,
            dropout=params.dropout, device = params.device)

        self.decoder = Decoder(
            id_enc=self.id_enc,n_position=params.n_position,
            d_embedding=params.d_embedding, d_model=params.d_model, d_inner=params.d_inner,
            n_layers=params.n_layers, n_head=params.n_head, d_k=params.d_k, d_v=params.d_v,
            L=params.predict_steps, S=params.S,
            dropout=params.dropout, device = params.device)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src_seq, trg_seq):

        src_mask = None
        trg_mask = get_subsequent_mask(trg_seq)

        if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
            enc_id = None
            dec_id = None
        else:
            enc_id = src_seq[:,:,-1].type(torch.long)
            dec_id = trg_seq[:, :, -1].type(torch.long)
            src_seq = src_seq[:,:,:-1]
            trg_seq = trg_seq[:,:,:-1]

        enc_output= self.encoder(enc_id, src_seq, src_mask,)
        dec_output, dec_y_hat, dec_alpha, dec_Sigma = self.decoder(dec_id, trg_seq, trg_mask, enc_output, None,
                                                        None, src_mask)

        return dec_y_hat.squeeze(-1),dec_Sigma.squeeze(-1)


    def test(self, src_seq, trg_seq):

        src_mask = None
        trg_mask = get_subsequent_mask(trg_seq)

        if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
            enc_id = None
            dec_id = None
        else:
            enc_id = src_seq[:,:,-1].type(torch.long)
            dec_id = trg_seq[:, :, -1].type(torch.long)
            src_seq = src_seq[:,:,:-1]
            trg_seq = trg_seq[:,:,:-1]

        enc_output = self.encoder(enc_id,src_seq, src_mask)
        for t in range(self.params.predict_steps):
            if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
                dec_output, dec_y_hat, dec_alpha, dec_Sigma, dec_slf_attn_list, dec_enc_attn_list = self.decoder(dec_id,trg_seq[:,0:t+1,:], trg_mask[:,0:t+1,0:t+1], enc_output,
                                                        None, None, src_mask,True)
            else:
                dec_output, dec_y_hat, dec_alpha, dec_Sigma, dec_slf_attn_list, dec_enc_attn_list = self.decoder(dec_id[:,0:t+1],trg_seq[:,0:t+1,:], trg_mask[:,0:t+1,0:t+1], enc_output,
                                                        None, None, src_mask,True)

            if t < (self.params.predict_steps - 1):
                trg_seq[:,t+1,0] = dec_y_hat[:,-1,0]

        return dec_y_hat.squeeze(-1),dec_alpha.squeeze(-1),dec_Sigma.squeeze(-1),dec_enc_attn_list[-1]


def loss_fn(mu: Variable, sigma: Variable, labels: Variable, predict_start):
    labels = labels[:,predict_start:]
    # print(mu.shape,labels.shape,sigma.shape)
    zero_index = (labels != 0)
    mask = sigma == 0
    sigma_index = zero_index * (~mask)
    distribution = torch.distributions.normal.Normal(mu[sigma_index], sigma[sigma_index])
    likelihood = distribution.log_prob(labels[sigma_index])
    difference = torch.abs(mu - labels)

    return -0.5*torch.mean(likelihood) + torch.mean(difference)


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return[diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(mu: torch.Tensor, labels: torch.Tensor, rou: float,relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = 2*torch.sum(torch.mul(rou-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
                      labels[zero_index] - mu[zero_index])).item()
        return [diff, 1]
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
                      labels[zero_index] - mu[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return[diff, summation]

def accuracy_ROU(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    labels = labels[zero_index]
    mu = mu[zero_index]
    if relative:
        diff = 2*torch.mean(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu)).item()
        return [diff, 1]
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu))
        summation = torch.sum(torch.abs(labels)).item()
        return[diff, summation]
    numerator = 0
    denominator = 0
    samples = mu
    pred_samples = mu.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def accuracy_ROU_(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels == 0)
    mu[zero_index] = 0
    if relative:
        diff = 2*torch.mean(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu), axis=1)
        return diff/1
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu), axis=1)
        summation = torch.sum(torch.abs(labels), axis=1)

        return diff/summation


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):

    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != np.inf), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[labels == 0] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result
