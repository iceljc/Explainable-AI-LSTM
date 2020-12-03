import numpy as np
from numpy import newaxis as na

def lrp_linear(h_in, W, b, h_out, R_out, bias_nb_units, eps, bias_factor=0.0):
	"""
	LRP for a linear layer with input dim D and output dim M.
	input:
	- h_in:            forward pass input, of shape (D,)
	- W:              connection weights, of shape (D, M)
	- b:              biases, of shape (M,)
	- h_out:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
	- R_out:           relevance at layer output, of shape (M,)
	- bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
	- eps:            stabilizer (small positive number)
	- bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
	
	output:
	- Rin:            relevance at layer input, of shape (D,)
	"""

	# (1, M)
	sign_out = np.where(h_out[na,:]>=0, 1.0, -1.0) 

	# (D, M)
	numerator = (W * h_in[:, na]) + bias_factor*(b[na, :]*1.0 + eps*sign_out*1.0) / bias_nb_units

	# (1, M)
	denominator = h_out[na,:] + (eps*sign_out*1.0)

	# (D, M)
	ratio = numerator / denominator * R_out[na, :]

	Rin = ratio.sum(axis=1)

	return Rin


# def lrp_linear_new(h_in, W, b, h_out, R_out, bias_nb_units, eps, bias_factor=0.0):
#   beta = 0.5
#   eps = 0.1

#   connects = W * h_in[:, na]
#   connects = connects.transpose()

#   m, n = connects.shape
#   abs_sum = np.zeros(m)

#   for i in range(m):
#       abs_sum[i] = sum(map(lambda x: abs(x), connects[i]))
#       # abs_sum[i] = sum(connects[i]) / len(connects[i])
		
#   abs_sum = np.tile(abs_sum, (n, 1))
#   connects = connects.transpose()
#   pos = connects + abs_sum
#   neg = connects - abs_sum

#   pos_sum = pos.sum(axis=0)[na, :]
#   neg_sum = neg.sum(axis=0)[na, :]

#   # pos_sum = pos_sum + eps*np.where(pos_sum>=0, 1.0, -1.0)
#   # neg_sum = neg_sum + eps*np.where(neg_sum>=0, 1.0, -1.0)

#   tmp = (1 + beta) * pos / pos_sum - beta * neg / neg_sum
#   R_in = tmp * R_out[na, :]

#   return R_in.sum(axis=1)


def lrp_linear_new(h_in, W, b, h_out, R_out, bias_nb_units, method="zero", alpha=2, eps=0.01, bias_factor=0.0):
	alpha = alpha
	beta = 1 - alpha

	# linear split
	sign_out = np.where(h_out>=0, 1.0, -1.0)
	numerator = np.matmul(W.transpose(), h_in) + bias_factor*(b + eps*sign_out*1.0)
	denominator = h_out + (eps*sign_out*1.0)
	R_out = numerator / denominator * R_out

	# alpha-beta rule
	connects = W * h_in[:, na] + bias_factor * b[na, :] / bias_nb_units
	connects = connects.transpose()

	if method == 'abs':
		# tmp = abs(connects)
		# s = tmp.sum(axis=1) + eps
		# res = connects / s[:, na]
		# res = res.transpose()

		connects = abs(connects)
		s = connects.sum(axis=1) + eps
		res = connects / s[:, na]
		res = res.transpose()

	elif method == 'mod':
		tmp = connects >= 0
		pos = connects * tmp
		neg = connects - pos
		pos_sum = np.sum(pos, axis=1)
		neg_sum = np.sum(neg, axis=1)

		s = pos_sum + neg_sum
		sign_out = np.where(s>=0, 1.0, -1.0)

		pos_sum += (pos_sum >= 0)*eps - (pos_sum < 0)*eps
		neg_sum += (neg_sum >= 0)*eps - (neg_sum < 0)*eps

		pos_nonzero = np.count_nonzero(pos, axis=1)
		neg_nonzero = np.count_nonzero(neg, axis=1)
		
		total = np.maximum(pos_nonzero + neg_nonzero, np.ones(len(neg_nonzero)))
		beta = neg_nonzero / total
		alpha = 1 + beta

		res = alpha[:, na] * pos / pos_sum[:, na] - beta[:, na] * neg / neg_sum[:, na]
		# res = alpha * pos / pos_sum[:, na] + beta * neg / neg_sum[:, na]
		# res = res * sign_out[:, na]
		res = res.transpose()


	else:
		if method == 'zero':
			tmp = connects >= 0
			pos = connects * tmp
			neg = connects - pos

		elif method == 'avg':
			avg = np.mean(connects, axis=1)[:, na]
			tmp = connects >= avg
			pos = connects * tmp
			neg = connects - pos


		pos_sum = np.sum(pos, axis=1)
		neg_sum = np.sum(neg, axis=1)

		pos_sum += (pos_sum >= 0)*eps - (pos_sum < 0)*eps
		neg_sum += (neg_sum >= 0)*eps - (neg_sum < 0)*eps


		res = alpha * pos / pos_sum[:, na] + beta * neg / neg_sum[:, na]
		res = res.transpose()

	R_in = res * R_out[na, :]

	return R_in.sum(axis=1)

	# m, n = connects.shape
	# res = np.zeros(connects.shape)
	# for i in range(m):
	#   if method == "zero":
	#       standard = 0
	#       pos = sum(t for t in connects[i] if t >= standard)
	#       neg = sum(t for t in connects[i] if t < standard)

	#       # pos_count = sum(t > standard for t in connects[i])
	#       # neg_count = sum(t < standard for t in connects[i])

	#       # alpha = pos_count / (pos_count + neg_count)
	#       # beta = 1 - alpha 

	#   elif method == "avg":
	#       standard = np.mean(connects[i])
	#       pos = sum(t for t in connects[i] if t >= standard)
	#       neg = sum(t for t in connects[i] if t < standard)
	#       if pos >= 0:
	#           pos += eps
	#       else:
	#           pos -= eps

	#       if neg >= 0:
	#           neg += eps
	#       else:
	#           neg -= eps

	#   for j in range(n):
	#       if connects[i][j] >= standard:
	#           res[i][j] = alpha * connects[i][j] / pos
	#       elif connects[i][j] < standard:
	#           res[i][j] = beta * connects[i][j] / neg

	# res = res.transpose()
	# R_in = res * R_out[na, :]

	



def lrp_linear_split(h_in, W, b, h_out, R_out, eps=0.001, bias_factor=0.0):

	sign_out = np.where(h_out>=0, 1.0, -1.0)

	numerator = np.matmul(W, h_in) + bias_factor*(b + eps*sign_out*1.0)
	denominator = h_out + (eps*sign_out*1.0)

	ratio = numerator / denominator

	return ratio * R_out















