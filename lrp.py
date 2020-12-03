import torch
import torch.nn as nn
import numpy as np
import math, copy, time
from torchtext import data, datasets
from setting import params
from utils import print_data_info, print_examples, count_correct_prediction, lookup_words
from model import make_model, SimpleLossCompute
from init_weight import init_weight
from data import rebatch
from biLSTM import *


def run_lrp(data_iter, vocab, model_file):
	net = BiLSTM(model_file=model_file)
	vocab_size = len(vocab)
	method = "all"
	linear = "epsilon"
	alpha = 2
	backward = False
	print("Method - ", method)
	print("Linear - ", linear)
	print("alpha - ", alpha)
	correct, total = 0, 0
	pos_total, neg_total, neu_total = 0, 0, 0
	pos_true, neg_true, neu_true = 0, 0, 0
	
	# idx = [0, 66, 196, 197, 221]
	idx = [16]
	max_id = max(idx)
	idx = set(idx)

	for i, batch in enumerate(data_iter):
		src = batch.src.cpu().numpy()[0]
		true_label = batch.trg.cpu().numpy()[0]

		if i == 216:
			# ['daring', ',', 'mesmerizing', 'and', 'exceedingly', 'hard', 'to', 'forget', '.']
			src = list(src)
			src = src[4:-1]
			src[1] = vocab.stoi['easy']
			src[-1] = vocab.stoi['remember']
			src = np.array(src)
			net.set_input(src)
			probs = net.forward()
			pred_label = np.argmax(probs)
			Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method, linear=linear, alpha=alpha)
			R_words = np.sum(Rx_fw + Rx_bw, axis=1)
			words = lookup_words(src, vocab)

			print()
			print('Predicted probs:', probs)
			print('Predicted label: ', pred_label)
			print(words)
			print(R_words)
			print()
			break


		# if i == 180:
		# 	add = ["fail", "to", "be"]
		# 	add_idx = []
		# 	for w in add:
		# 		add_idx.append(vocab.stoi[w])
		# 	src = np.array(add_idx + list(src))
		# 	net.set_input(src)
		# 	probs = net.forward()
		# 	pred_label = np.argmax(probs)
		# 	Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method, linear=linear, alpha=alpha)
		# 	R_words = np.sum(Rx_fw + Rx_bw, axis=1)
		# 	words = lookup_words(src, vocab)

		# 	print()
		# 	print('Predicted probs:', probs)
		# 	print('Predicted label: ', pred_label)
		# 	print(words)
		# 	print(R_words)
		# 	print()
		# 	break


		# if i == 189:
		# 	net.set_input(src)
		# 	probs = net.forward()
		# 	pred_label = np.argmax(probs)
			
		# 	if backward:
		# 		Gx, Gx_rev = net.backward(pred_label)
		# 		R_words = (np.linalg.norm(Gx + Gx_rev, ord=2, axis=1))**2
		# 	else:
		# 		Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method, linear=linear, alpha=alpha, eps=0.001, bias_factor=0.0)
		# 		R_words = np.sum(Rx_fw + Rx_bw, axis=1)

		# 	words = lookup_words(src, vocab)

		# 	print()
		# 	# print(Rx_fw.sum()+Rx_bw.sum()+R_rest.sum())
		# 	# print(Rx_fw.sum()+Rx_bw.sum())
		# 	print('Predicted probs:', probs)
		# 	print('True label: ', true_label)
		# 	print('Predicted label: ', pred_label)
		# 	print(words)
		# 	print(R_words)
		# 	print()
		# 	break

		# total += 1
		# if true_label == pred_label:
		# 	if pred_label == 0:
		# 		pos_true += 1
		# 	elif pred_label == 1:
		# 		neg_true += 1
		# 	elif pred_label == 2:
		# 		neu_true += 1

		# 	correct += 1

		# if true_label == 0:
		# 	pos_total += 1
		# elif true_label == 1:
		# 	neg_total += 1
		# elif true_label == 2:
		# 	neu_total += 1

		

	# print("Test accuracy: ", correct/total * 100, correct, total)
	# print("Pos accuracy: ", pos_true/pos_total*100, pos_true, pos_total)
	# print("Neg accuracy: ", neg_true/neg_total*100, neg_true, neg_total)
	# print("Neu accuracy: ", neu_true/neu_total*100, neu_true, neu_total)


		# if i == 564:
		# 	src = src[1:-1]
		# 	# ['this', 'is', 'pretty', 'dicey', 'material', '.']
		# 	src[2] = vocab.stoi['not']
		# 	net.set_input(src)
		# 	probs = net.forward()
		# 	pred_label = np.argmax(probs)
		# 	Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
		# 	R_words = np.sum(Rx_fw + Rx_bw, axis=1)
		# 	words = lookup_words(src, vocab)

		# 	print()
		# 	print('Predicted probs:', probs)
		# 	print('True label: ', true_label)
		# 	print('Predicted label: ', pred_label)
		# 	print(words)
		# 	print(R_words)
		# 	print()
		# 	break


		# if i == 180:
		# 	src = src[1:-1]
		# 	add = ["not"]
		# 	add_idx = []
		# 	for w in add:
		# 		add_idx.append(vocab.stoi[w])
		# 	src = np.array(add_idx + list(src))
		# 	net.set_input(src)
		# 	probs = net.forward()
		# 	pred_label = np.argmax(probs)
		# 	Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
		# 	R_words = np.sum(Rx_fw + Rx_bw, axis=1)
		# 	words = lookup_words(src, vocab)

		# 	print()
		# 	print('Predicted probs:', probs)
		# 	print('True label: ', true_label)
		# 	print('Predicted label: ', pred_label)
		# 	print(words)
		# 	print(R_words)
		# 	print()

		# 	add = ["not"]
		# 	add_idx = []
		# 	for w in add:
		# 		add_idx.append(vocab.stoi[w])
		# 	src = np.array(add_idx + list(src))
		# 	net.set_input(src)
		# 	probs = net.forward()
		# 	pred_label = np.argmax(probs)
		# 	Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
		# 	R_words = np.sum(Rx_fw + Rx_bw, axis=1)
		# 	words = lookup_words(src, vocab)

		# 	print('Predicted probs:', probs)
		# 	print('True label: ', true_label)
		# 	print('Predicted label: ', pred_label)
		# 	print(words)
		# 	print(R_words)
		# 	print()

			# break

		# if i in idx:
		#   src = src[1:-1]
		#   net.set_input(src)
		#   probs = net.forward()
		#   pred_label = np.argmax(probs)

		#   # R_words = np.zeros(len(src))
		#   # for method in methods:
		#   #   Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
		#   #   R_words += np.tanh(np.sum(Rx_fw + Rx_bw, axis=1))

		#   # R_words = R_words / len(methods)

		#   Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
		#   R_words = np.sum(Rx_fw + Rx_bw, axis=1)
		#   words = lookup_words(src, vocab)
		#   # print(Rx_fw.sum() + Rx_bw.sum() + R_rest.sum())

		#   print()
		#   print(i)
		#   print('Predicted probs:', probs)
		#   print('True label: ', true_label)
		#   print('Predicted label: ', pred_label)
		#   print(words)
		#   print(R_words)
		#   print()
		#   print("="*10)

		# elif i > max(idx):
		#   break

		# if pred_label == true_label and pred_label == 1:
		# 	# Rx_fw, Rx_bw, R_rest = net.lrp_prop(pred_label, method=method)
		# 	# R_words = np.sum(Rx_fw + Rx_bw, axis=1)
		# 	# words = lookup_words(src, vocab)

		# 	print()
		# 	print(i)
		# 	print('Predicted probs:', probs)
		# 	# print('True label: ', true_label)
		# 	print('Predicted label: ', pred_label)
			# print(words)
			# print(R_words)
			# print()

			# break
		

		# print('Sanity check: ')
		# Rx_fw, Rx_bw, R_rest = net.lrp(pred_label, eps=0.001, bias_factor=1.0)
		# print(Rx_fw.sum()+Rx_bw.sum()+R_rest.sum())

		# break

	#   total += 1
	#   if pred_label == true_label:
	#       correct += 1

	# print(correct / total * 100)







if __name__ == '__main__':
	
	seed = 42
	# np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	DEVICE = params['DEVICE']

	UNK_TOKEN = "<unk>"
	PAD_TOKEN = "<pad>"    
	# SOS_TOKEN = "<s>"
	# EOS_TOKEN = "</s>"
	LOWER = True

	SRC = data.Field(sequential=True, tokenize='spacy', batch_first=True, lower=LOWER, include_lengths=True, 
		unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)
	# , init_token=SOS_TOKEN, eos_token=EOS_TOKEN

	LABEL = data.Field(sequential=False)

	#############################
	train_data, valid_data, test_data = datasets.SST.splits(SRC, LABEL)

	# build vocab
	SRC.build_vocab(train_data, max_size=30000,vectors="glove.6B.300d", unk_init=torch.Tensor.normal_)
	LABEL.build_vocab(train_data)

	PAD_INDEX = SRC.vocab.stoi[PAD_TOKEN]
	# SOS_INDEX = SRC.vocab.stoi[SOS_TOKEN]
	# EOS_INDEX = SRC.vocab.stoi[EOS_TOKEN]
	# print(LABEL.vocab.freqs.most_common(10))

	#############################
	# define iterator
	train_iter = data.BucketIterator(train_data, batch_size=params['BATCH_SIZE'],
		device=DEVICE, sort_within_batch=True, 
		sort_key=lambda x: len(x.text), 
		train=True, repeat=False)

	# train_iter = data.Iterator(train_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	test_iter = data.Iterator(test_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	# print_data_info(train_data, valid_data, test_data, SRC, LABEL)

	#############################
	

	run_lrp((rebatch(PAD_INDEX, b) for b in test_iter), vocab=SRC.vocab, model_file='sa_model4.pt')

	
	# true: pos, pred: pos, words: ['a', 'masterpiece', 'four', 'years', 'in', 'the', 'making', '.'], relevance: [ 0.13147668  1.88773544  0.72980414  0.71849921 -0.02894038  0.04202898 0.21119616 -0.01914013] 
	# true: neg, pred: neg, words: , relevance: 
	# true: neu, pred: neu, words: ['effective', 'but', 'too', '-', 'tepid', 'biopic'], relevance: [-0.39086432  0.0311396   0.18368873  0.16158153  1.10337837 -0.17609372]

	# true: pos, pred: neg, words: , relevance:
	# true: neg, pred: pos, words: ['it', 'took', '19', 'predecessors', 'to', 'get', 'this', '?'], relevance: [ 0.30267079 -0.08001834  0.43165412  0.63442843  0.20579954  0.08087923  0.31217654 -0.55773837]











