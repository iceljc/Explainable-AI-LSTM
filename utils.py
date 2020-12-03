import numpy as np
import torch


def count_correct_prediction(y_pred, y_true):
	correct = 0
	_, label_pred = torch.max(y_pred, dim=1)

	y_true = y_true.cpu().numpy()
	label_pred = label_pred.cpu().numpy()

	for i in range(y_true.shape[0]):
		if label_pred[i] == y_true[i]:
			correct += 1

	return correct



def print_data_info(train_data, valid_data, test_data, src_field, label_field):

	print("Dataset size:")
	print("Training size: {}".format(len(train_data)))
	print("Validation size: {}".format(len(valid_data)))
	print("Test size: {}".format(len(test_data)))
	print()

	print("First training example:")
	# print(vars(train_data[0]))
	print("src:", " ".join(vars(train_data[0])['text']))
	print("label:", vars(train_data[0])['label'], "\n")

	# print(label_field.vocab.freqs.most_common(10))

	print("Most common words (src):")
	print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
	
	print("Number of words:", len(src_field.vocab))



def lookup_words(x, vocab=None):
	if vocab is not None:
		x = [vocab.itos[i] for i in x]

	return [str(t) for t in x]


def lookup_label(idx):
	if idx == 0:
		return 'Positive'
	elif idx == 1:
		return 'Negative'
	else:
		return 'Neutral'


def print_examples(data_iter, model, n, src_vocab):
	# sos_idx, eos_idx, 
	model.eval()
	count = 0

	for i, batch in enumerate(data_iter):
		src = batch.src[0]
		label = batch.trg
		nseq = batch.nseqs

		if nseq != 1:
			print("Pleas process one sentence at a time.")
			break

		# src = src[1:] if src[0] == sos_idx else src
		# src = src[:-1] if src[-1] == eos_idx else src

		with torch.no_grad():
			out = model(batch.src, batch.src_lengths)
			out = out.contiguous().view(-1, out.size(-1))
			_, label_pred = torch.max(out, dim=1)

		print("\nExample %d" % (i+1))
		print("Source: ", " ".join(lookup_words(src, vocab=src_vocab)), '\n')
		print("Target: ", lookup_label(label.cpu().numpy()[0]))
		print("Prediction: ", lookup_label(label_pred.cpu().numpy()[0]))

		count += 1

		if count == n:
			break
















