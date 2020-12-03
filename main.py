import torch
import torch.nn as nn
import numpy as np
import math, copy, time
from torchtext import data, datasets
from setting import params
from utils import print_data_info, print_examples, count_correct_prediction
from model import make_model, SimpleLossCompute
from init_weight import init_weight
from data import rebatch


def run_epoch(data_iter, model, loss_compute, print_every=100):

	total_loss, total_tokens, print_tokens, total_nseqs = 0.0, 0.0, 0.0, 0.0
	correct_preds = 0.0

	for i, batch in enumerate(data_iter, 1):
		start = time.time()
		out = model(batch.src, batch.src_lengths)
		loss = loss_compute(out, batch.trg, batch.nseqs)

		y_pred = out.contiguous().view(-1, out.size(-1))
		y_true = batch.trg
		correct = count_correct_prediction(y_pred, y_true)

		total_loss += loss
		total_tokens += batch.ntokens
		print_tokens += batch.ntokens
		total_nseqs += batch.nseqs
		correct_preds += correct

		if model.training and i%print_every == 0:
			elapsed = time.time() - start
			print("Epoch step: %d, Loss: %f, Tokens per Sec: %f" %
				(i, loss/batch.nseqs, print_tokens/elapsed))
			start = time.time()
			print_tokens = 0

	return total_loss/total_nseqs, correct_preds/total_nseqs



def train(model, num_epochs, learning_rate, print_every=100):
	model.cuda()
	criterion = nn.CrossEntropyLoss(reduction="sum") # , ignore_index=PAD_INDEX
	# criterion = nn.NLLLoss(reduction="sum") # 
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)

	train_losses, valid_losses = [], []
	train_ac, valid_ac = [], []

	for epoch in range(1, num_epochs+1):
		model.train()
		train_loss, train_accuracy = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), model, SimpleLossCompute(criterion, optimizer, is_train=True), print_every=print_every)
		print("Epoch %d, Train loss: %f, Train accuracy: %0.3f" %(epoch, train_loss, train_accuracy*100))
		train_losses.append(train_loss)
		train_ac.append(train_accuracy)

		model.eval()
		with torch.no_grad():
			valid_loss, valid_accuracy = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), model, SimpleLossCompute(criterion, None, is_train=False))

		print("Epoch %d, Valid loss: %f, Valid accuracy: %0.3f" %(epoch, valid_loss, valid_accuracy*100))
		valid_losses.append(valid_loss)
		valid_ac.append(valid_accuracy)

	print("Saving the model...")
	with open('sa_model4.pt', 'wb') as f:
		torch.save(model, f)

	return train_losses, valid_losses, train_ac, valid_ac


def run_test(data_iter, model):

	total_seqs = 0
	correct = 0
	model.eval()

	for i, batch in enumerate(data_iter, 1):
		
		with torch.no_grad():
			out = model(batch.src, batch.src_lengths)
			y_pred = out.contiguous().view(-1, out.size(-1))
			y_true = batch.trg

			total_seqs += batch.nseqs
			_, label_pred = torch.max(y_pred, dim=1)
			# print(y_true.data.item(), label_pred.data.item())
			if y_true.data.item() == label_pred.data.item():
				correct += 1

	test_accuracy = correct * 1.0 / total_seqs * 100

	return test_accuracy




if __name__ == '__main__':
	
	seed = 42
	np.random.seed(seed)
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

	valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	test_iter = data.Iterator(test_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	# train_iter, valid_iter, test_iter = data.BucketIterator.splits(
	#   (train_data, valid_data, test_data), batch_size=params['BATCH_SIZE'], 
	#   device=DEVICE, sort_within_batch=True)

	# print data info
	print_data_info(train_data, valid_data, test_data, SRC, LABEL)

	#############################
	print("Building the model ...")

	model = make_model(SRC.vocab, output_class=3, embed_size=params['EMBEDDING_DIM'], 
		hidden_size=params['HIDDEN_SIZE'], num_layers=params['NUM_LAYERS'], 
		dropout=params['DROPOUT_PROB'])

	model.apply(init_weight)
	#############################

	print("Start training ... ")
	# print(SRC.vocab.itos[1], SRC.vocab.itos[2], SRC.vocab.itos[3], SRC.vocab.itos[4])

	train_losses, valid_losses, train_ac, valid_ac = train(model, num_epochs=params['NUM_EPOCHS'], learning_rate=params['LEARNING_RATE'])

	print("Start test ... ")

	print_examples((rebatch(PAD_INDEX, b) for b in test_iter), model, n=5, src_vocab=SRC.vocab)
	# sos_idx=SOS_INDEX, eos_idx=EOS_INDEX, 
	print()

	test_accuracy = run_test((rebatch(PAD_INDEX, b) for b in test_iter), model)

	print('Test accuracy: ', test_accuracy)
		







