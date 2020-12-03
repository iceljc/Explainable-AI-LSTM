import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from setting import params


class SentimentModel(nn.Module):

	def __init__(self, encoder, generator, src_vocab_size, embed_size, pad_idx):
		super(SentimentModel, self).__init__()
		self.encoder = encoder
		self.src_embed = nn.Embedding(src_vocab_size, embed_size) # , padding_idx=pad_idx
		self.generator = generator

	def forward(self, src_idx, src_lengths):
		# encoder_hidden, encoder_final = self.encode(src_idx, src_lengths)
		encoder_hidden, fwd_final, bwd_final = self.encode(src_idx, src_lengths)
		return self.generator(fwd_final, bwd_final)

	def encode(self, src_idx, src_lengths):
		return self.encoder(self.src_embed(src_idx), src_lengths)


class Generator(nn.Module):
	
	def __init__(self, hidden_size, output_class):
		super(Generator, self).__init__()
		# l1_out_size = 2*hidden_size
		# self.dropout1 = nn.Dropout(params['DROPOUT_PROB'])
		# self.sm_fc1 = nn.Linear(2*hidden_size, l1_out_size, bias=False)
		# self.tanh = nn.Tanh()
		# self.dropout2 = nn.Dropout(params['DROPOUT_PROB'])
		# self.sm_fc2 = nn.Linear(l1_out_size, output_class, bias=False)

		l1_out_size = hidden_size
		self.dropout = nn.Dropout(params['DROPOUT_PROB'])
		self.linear_fwd = nn.Linear(hidden_size, output_class, bias=False)
		self.linear_bwd = nn.Linear(hidden_size, output_class, bias=False)
		# self.tanh = nn.Tanh()
		# self.linear_out = nn.Linear(l1_out_size, output_class, bias=False)

	def forward(self, x_fwd, x_bwd):
		# dropout1 = self.dropout1(x)
		# project1 = self.sm_fc1(dropout1)
		# # tanh = self.tanh(project1)
		# dropout2 = self.dropout2(project1)
		# out = self.sm_fc2(dropout2)

		# forward
		x_fwd = self.dropout(x_fwd)
		proj_fwd = self.linear_fwd(x_fwd)

		# backward
		x_bwd = self.dropout(x_bwd)
		proj_bwd = self.linear_bwd(x_bwd)

		out = proj_fwd + proj_bwd
		# proj = self.tanh(proj)
		# drop = self.dropout(proj)
		# out = self.linear_out(out)

		return out


class Encoder(nn.Module):
	"""Encodes a sequence of word embeddings"""
	def __init__(self, input_size, hidden_size, num_layers=1, dropout_prob=0.5):
		super(Encoder, self).__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.rnn = nn.LSTM(input_size = input_size,
							hidden_size = hidden_size,
							num_layers = num_layers,
							batch_first = True,
							bidirectional = True)


	def forward(self, x, lengths):
		packed = pack_padded_sequence(x, lengths, batch_first=True)
		output, final = self.rnn(packed) # final: (h_n, c_n)
		output, _ = pad_packed_sequence(output, batch_first=True)
		final = final[0]

		fwd_final = final[0:final.size(0):2]
		bwd_final = final[1:final.size(0):2]

		# hidden_final = torch.cat([fwd_final, bwd_final], dim=2)

		# return output, hidden_final
		return output, fwd_final, bwd_final


class SimpleLossCompute:
	"""A simple loss compute and train function."""

	def __init__(self, criterion, opt=None, is_train=False):
		self.criterion = criterion
		self.opt = opt
		self.is_train = is_train

	def __call__(self, y_pred, y_true, norm):

		# print(y_pred.shape, y_true.shape)
		y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
		# y_true = torch.zeros(y_pred.shape).scatter_(1,  y_true_index.cpu(),1).float().cuda()
		loss = self.criterion(y_pred, y_true)
		loss = loss/norm # norm: number of sequences in a batch

		if self.is_train:
			loss.backward()

		if self.opt is not None:
			# self.opt.zero_grad()
			self.opt.step()
			self.opt.zero_grad()

		tmp_loss = loss.item()*norm
		del loss
		# torch.cuda.empty_cache()

		return tmp_loss


def make_model(src_vocab, output_class, embed_size=256, hidden_size=512, num_layers=1, dropout=0.5):

	# pretrain_embed = nn.Embedding(src_vocab_len, embed_size)
	# pretrain_embed.weight.requires_grad = False
	# # nn.init.xavier_uniform_(pretrain_embed.state_dict()['weight'])
	# word_dict = src_vocab.stoi

	pad_idx = src_vocab.stoi["<pad>"]
	unk_idx = src_vocab.stoi["<unk>"]

	model = SentimentModel(
		Encoder(embed_size, hidden_size, num_layers=num_layers, dropout_prob=dropout),
		Generator(hidden_size, output_class),
		len(src_vocab),
		embed_size, 
		pad_idx)

	# nn.init.xavier_uniform_(model.src_embed.weight.data)
	model.src_embed.weight.data.copy_(src_vocab.vectors)
	model.src_embed.weight.data[pad_idx] = torch.empty(embed_size).normal_(0, 0.1)
	model.src_embed.weight.data[unk_idx] = torch.empty(embed_size).normal_(0, 0.1)
	# model.src_embed.weight.requires_grad = False

	return model




