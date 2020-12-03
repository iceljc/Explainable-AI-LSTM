import numpy as np
import torch
import torch.nn.functional as F
from lrp_linear import lrp_linear, lrp_linear_new, lrp_linear_split
from lrp_product_ratio import lrp_product

class BiLSTM:

	def __init__(self, model_file='sa_model.pt'):

		model = torch.load(model_file).cpu()
		states = model.state_dict()

		self.embed = states['src_embed.weight'].numpy()

		# forward encoder
		self.Wih_fw = states['encoder.rnn.weight_ih_l0'].numpy()
		self.Whh_fw = states['encoder.rnn.weight_hh_l0'].numpy()
		self.bih_fw = states['encoder.rnn.bias_ih_l0'].numpy()
		self.bhh_fw = states['encoder.rnn.bias_hh_l0'].numpy()

		# backward encoder
		self.Wih_bw = states['encoder.rnn.weight_ih_l0_reverse'].numpy()
		self.Whh_bw = states['encoder.rnn.weight_hh_l0_reverse'].numpy()
		self.bih_bw = states['encoder.rnn.bias_ih_l0_reverse'].numpy()
		self.bhh_bw = states['encoder.rnn.bias_hh_l0_reverse'].numpy()

		# output layer
		self.Wout_fw = states['generator.linear_fwd.weight'].numpy()
		self.Wout_bw = states['generator.linear_bwd.weight'].numpy()

		self.hidden_size = int(self.Wih_fw.shape[0]/4)
		self.embed_size = self.embed.shape[1]
		


	def set_input(self, w, delete_ids=None):
		T = len(w)
		x = np.zeros((T, self.embed_size))
		x[:, :] = self.embed[w, :]

		if delete_ids is not None:
			x[delete_ids, :] = np.zeros((len(delete_ids), self.embed_size))

		self.T = T
		self.w = w
		self.x = x
		self.x_rev = x[::-1,:].copy()
		
		self.h_fw = np.zeros((T+1, self.hidden_size))
		self.c_fw = np.zeros((T+1, self.hidden_size))
		self.h_bw = np.zeros((T+1, self.hidden_size))
		self.c_bw = np.zeros((T+1, self.hidden_size))

		self.gates_ih_fw = np.zeros((T, 4*self.hidden_size))  
		self.gates_hh_fw = np.zeros((T, 4*self.hidden_size)) 
		self.gates_pre_fw = np.zeros((T, 4*self.hidden_size))  # gates pre-activation
		self.gates_fw = np.zeros((T, 4*self.hidden_size))  # gates activation
		
		self.gates_ih_bw = np.zeros((T, 4*self.hidden_size))  
		self.gates_hh_bw = np.zeros((T, 4*self.hidden_size)) 
		self.gates_pre_bw = np.zeros((T, 4*self.hidden_size))
		self.gates_bw = np.zeros((T, 4*self.hidden_size))

		# self.h_fw = torch.zeros(T+1, self.hidden_size, dtype=torch.double)
		# self.c_fw = torch.zeros(T+1, self.hidden_size, dtype=torch.double)
		# self.h_bw = torch.zeros(T+1, self.hidden_size, dtype=torch.double)
		# self.c_bw = torch.zeros(T+1, self.hidden_size, dtype=torch.double)


	def forward(self):

		# Wih_fw = torch.DoubleTensor(self.Wih_fw)
		# bih_fw = torch.DoubleTensor(self.bih_fw)

		# for t in range(self.T):
		#   inp = torch.DoubleTensor(self.x[t])
		#   # hx = torch.DoubleTensor(self.h_fw[t-1])
		#   gates_fw = F.linear(inp, Wih_fw, bih_fw) + F.linear(self.h_fw[t-1], Whh_fw, bhh_fw)
		#   self.gates_pre_fw[t] = gates_fw

		#   ingate_pre_fw, forgetgate_pre_fw, cellgate_pre_fw, outgate_pre_fw = gates_fw.chunk(4)

		#   ingate_fw = F.sigmoid(ingate_pre_fw)
		#   forgetgate_fw = F.sigmoid(forgetgate_pre_fw)
		#   cellgate_fw = F.tanh(cellgate_pre_fw)
		#   outgate_fw = F.sigmoid(outgate_pre_fw)

		#   self.gates_fw[t, idx_i] = ingate_fw
		#   self.gates_fw[t, idx_f] = forgetgate_fw
		#   self.gates_fw[t, idx_g] = cellgate_fw
		#   self.gates_fw[t, idx_o] = outgate_fw

		#   self.c_fw[t] = (forgetgate_fw * self.c_fw[t-1]) + (ingate_fw * cellgate_fw)
		#   self.h_fw[t] = outgate_fw * F.tanh(self.c_fw[t])



		d = self.hidden_size
		idx = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o
		idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o

		for t in range(self.T):
			# forward
			self.gates_ih_fw[t] = np.dot(self.Wih_fw, self.x[t])
			self.gates_hh_fw[t] = np.dot(self.Whh_fw, self.h_fw[t-1])
			self.gates_pre_fw[t] = self.gates_ih_fw[t] + self.gates_hh_fw[t] + self.bih_fw + self.bhh_fw
			self.gates_fw[t, idx] = 1.0 / (1.0 + np.exp(-self.gates_pre_fw[t, idx]))
			self.gates_fw[t, idx_g] = np.tanh(self.gates_pre_fw[t, idx_g])
			self.c_fw[t] = self.gates_fw[t, idx_f] * self.c_fw[t-1] + self.gates_fw[t, idx_i] * self.gates_fw[t, idx_g]
			self.h_fw[t] = self.gates_fw[t, idx_o] * np.tanh(self.c_fw[t])

			# backward
			self.gates_ih_bw[t] = np.dot(self.Wih_bw, self.x_rev[t])
			self.gates_hh_bw[t] = np.dot(self.Whh_bw, self.h_bw[t-1])
			self.gates_pre_bw[t] = self.gates_ih_bw[t] + self.gates_hh_bw[t] + self.bih_bw + self.bhh_bw
			self.gates_bw[t, idx] = 1.0 / (1.0 + np.exp(-self.gates_pre_bw[t, idx]))
			self.gates_bw[t, idx_g] = np.tanh(self.gates_pre_bw[t, idx_g])
			self.c_bw[t] = self.gates_bw[t, idx_f] * self.c_bw[t-1] + self.gates_bw[t, idx_i] * self.gates_bw[t, idx_g]
			self.h_bw[t] = self.gates_bw[t, idx_o] * np.tanh(self.c_bw[t])

		self.y_fw = np.dot(self.Wout_fw, self.h_fw[self.T-1])
		self.y_bw = np.dot(self.Wout_bw, self.h_bw[self.T-1])
		self.s = self.y_fw + self.y_bw

		return self.s.copy()


	def backward(self, sensitivity_class):
		T = self.T
		d = self.hidden_size
		C = self.Wout_fw.shape[0]
		idx = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o
		idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o

		self.dx = np.zeros(self.x.shape)
		self.dx_rev = np.zeros(self.x.shape)

		self.dh_fw = np.zeros((T+1, d))
		self.dc_fw = np.zeros((T+1, d))
		self.dgates_pre_fw = np.zeros((T, 4*d))
		self.dgates_fw = np.zeros((T, 4*d))

		self.dh_bw = np.zeros((T+1, d))
		self.dc_bw = np.zeros((T+1, d))
		self.dgates_pre_bw = np.zeros((T, 4*d))
		self.dgates_bw = np.zeros((T, 4*d))

		ds = np.zeros((C))
		ds[sensitivity_class] = 1.0
		dy_fw = ds.copy()
		dy_bw = ds.copy()

		self.dh_fw[T-1] = np.dot(self.Wout_fw.transpose(), dy_fw)
		self.dh_bw[T-1] = np.dot(self.Wout_bw.transpose(), dy_bw)

		for t in range(T-1, -1, -1):
			# forward
			self.dgates_fw[t, idx_o] = self.dh_fw[t] * np.tanh(self.c_fw[t])
			self.dc_fw[t] += self.dh_fw[t] * self.gates_fw[t, idx_o] * (1 - (np.tanh(self.c_fw[t]))**2)
			self.dgates_fw[t, idx_f] = self.dc_fw[t] * self.dc_fw[t-1]
			self.dc_fw[t-1] = self.dc_fw[t] * self.gates_fw[t, idx_f]
			self.dgates_fw[t, idx_i] = self.dc_fw[t] * self.gates_fw[t, idx_g]
			self.dgates_fw[t, idx_g] = self.dc_fw[t] * self.gates_fw[t, idx_i]
			self.dgates_pre_fw[t, idx] = self.dgates_fw[t, idx] * self.gates_fw[t, idx] * (1 - self.gates_fw[t, idx])
			self.dgates_pre_fw[t, idx_g] = self.dgates_fw[t, idx_g] * (1 - (self.gates_fw[t, idx_g])**2)
			self.dh_fw[t-1] = np.dot(self.Whh_fw.transpose(), self.dgates_pre_fw[t])
			self.dx[t] = np.dot(self.Wih_fw.transpose(), self.dgates_pre_fw[t])

			# backward
			self.dgates_bw[t, idx_o] = self.dh_bw[t] * np.tanh(self.c_bw[t])
			self.dc_bw[t] += self.dh_bw[t] * self.gates_bw[t, idx_o] * (1 - (np.tanh(self.c_bw[t]))**2)
			self.dgates_bw[t, idx_f] = self.dc_bw[t] * self.dc_bw[t-1]
			self.dc_bw[t-1] = self.dc_bw[t] * self.gates_bw[t, idx_f]
			self.dgates_bw[t, idx_i] = self.dc_bw[t] * self.gates_bw[t, idx_g]
			self.dgates_bw[t, idx_g] = self.dc_bw[t] * self.gates_bw[t, idx_i]
			self.dgates_pre_bw[t, idx] = self.dgates_bw[t, idx] * self.gates_bw[t, idx] * (1 - self.gates_bw[t, idx])
			self.dgates_pre_bw[t, idx_g] = self.dgates_bw[t, idx_g] * (1 - (self.gates_bw[t, idx_g])**2)
			self.dh_bw[t-1] = np.dot(self.Whh_bw.transpose(), self.dgates_pre_bw[t])
			self.dx_rev[t] = np.dot(self.Wih_bw.transpose(), self.dgates_pre_bw[t])

		return self.dx.copy(), self.dx_rev[::-1, :].copy()


	def lrp(self, label, eps=0.001, bias_factor=0.0):

		d = self.hidden_size
		T = self.T
		idx = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o
		idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o

		C = self.Wout_fw.shape[0] # number of classes

		Rx_fw = np.zeros(self.x.shape)
		Rx_bw = np.zeros(self.x.shape)
		
		Rh_fw = np.zeros((T+1, d))
		Rc_fw = np.zeros((T+1, d))
		Rg_fw = np.zeros((T, d)) # gate g only
		Rh_bw = np.zeros((T+1, d))
		Rc_bw = np.zeros((T+1, d))
		Rg_bw = np.zeros((T, d)) # gate g only
		
		Rout_mask = np.zeros((C))
		Rout_mask[label] = 1.0 

		# lrp_linear(h_in, W, b, h_out, R_out, bias_nb_units, eps, bias_factor)
		# Rh_fw[T-1] = self.h_fw[T-1]
		# Rh_bw[T-1] = self.h_bw[T-1]
		Rh_fw[T-1] = lrp_linear(self.h_fw[T-1], self.Wout_fw.transpose(), np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor)
		Rh_bw[T-1] = lrp_linear(self.h_bw[T-1], self.Wout_bw.transpose(), np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor)

		for t in range(T-1, -1, -1):
			# forward
			Rc_fw[t] += Rh_fw[t]
			Rc_fw[t-1] = lrp_linear(self.gates_fw[t, idx_f]*self.c_fw[t-1], np.identity(d), np.zeros((d)), self.c_fw[t], Rc_fw[t], 2*d, eps, bias_factor)
			Rg_fw[t] = lrp_linear(self.gates_fw[t, idx_i]*self.gates_fw[t, idx_g], np.identity(d), np.zeros((d)), self.c_fw[t], Rc_fw[t], 2*d, eps, bias_factor)
			Rx_fw[t] = lrp_linear(self.x[t], self.Wih_fw[idx_g].transpose(), self.bih_fw[idx_g]+self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d+self.embed_size, eps, bias_factor)
			Rh_fw[t-1] = lrp_linear(self.h_fw[t-1], self.Whh_fw[idx_g].transpose(), self.bih_fw[idx_g]+self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d+self.embed_size, eps, bias_factor)

			# backward
			Rc_bw[t] += Rh_bw[t]
			Rc_bw[t-1] = lrp_linear(self.gates_bw[t, idx_f]*self.c_bw[t-1], np.identity(d), np.zeros((d)), self.c_bw[t], Rc_bw[t], 2*d, eps, bias_factor)
			Rg_bw[t] = lrp_linear(self.gates_bw[t, idx_i]*self.gates_bw[t, idx_g], np.identity(d), np.zeros((d)), self.c_bw[t], Rc_bw[t], 2*d, eps, bias_factor)
			Rx_bw[t] = lrp_linear(self.x_rev[t], self.Wih_bw[idx_g].transpose(), self.bih_bw[idx_g]+self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d+self.embed_size, eps, bias_factor)
			Rh_bw[t-1] = lrp_linear(self.h_bw[t-1], self.Whh_bw[idx_g].transpose(), self.bih_bw[idx_g]+self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d+self.embed_size, eps, bias_factor)


		return Rx_fw, Rx_bw[::-1, :], Rh_fw[-1].sum()+Rc_fw[-1].sum()+Rh_bw[-1].sum()+Rc_bw[-1].sum()



	def lrp_prop(self, label, method='all', linear="epsilon", alpha=2, eps=0.001, bias_factor=0.0):

		d = self.hidden_size
		T = self.T
		idx = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o
		idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,f,g,o
		eps_linear = 0.001

		C = self.Wout_fw.shape[0] # number of classes

		Rx_fw = np.zeros(self.x.shape)
		Rx_bw = np.zeros(self.x.shape)
		
		Rh_fw = np.zeros((T+1, d))
		Rc_fw = np.zeros((T+1, d))
		Rg_fw = np.zeros((T, d)) # gate g
		Ro_fw = np.zeros((T, d)) # gate o
		Rf_fw = np.zeros((T, d)) # gate f
		Ri_fw = np.zeros((T, d)) # gate i

		Rh_bw = np.zeros((T+1, d))
		Rc_bw = np.zeros((T+1, d))
		Rg_bw = np.zeros((T, d)) # gate g
		Ro_bw = np.zeros((T, d)) # gate o
		Rf_bw = np.zeros((T, d)) # gate f
		Ri_bw = np.zeros((T, d)) # gate i

		
		Rout_mask = np.zeros((C))
		Rout_mask[label] = 1.0 

		# lrp_linear(h_in, W, b, h_out, R_out, bias_nb_units, eps, bias_factor)
		# Rh_fw[T-1] = self.h_fw[T-1]
		# Rh_bw[T-1] = self.h_bw[T-1]
		if linear == 'epsilon':
			Rh_fw[T-1] = lrp_linear(self.h_fw[T-1], self.Wout_fw.transpose(), np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor)
			Rh_bw[T-1] = lrp_linear(self.h_bw[T-1], self.Wout_bw.transpose(), np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor)
		else:
			Rh_fw[T-1] = lrp_linear_new(self.h_fw[T-1], self.Wout_fw.transpose(), np.zeros((C)), self.s, self.s*Rout_mask, 2*d, linear, alpha, eps_linear, bias_factor)
			Rh_bw[T-1] = lrp_linear_new(self.h_bw[T-1], self.Wout_bw.transpose(), np.zeros((C)), self.s, self.s*Rout_mask, 2*d, linear, alpha, eps_linear, bias_factor)

		for t in range(T-1, -1, -1):
			# forward
			ratio_o, ratio_c_cur = lrp_product(self.gates_fw[t, idx_o], np.tanh(self.c_fw[t]), eps, method)
			Ro_fw[t] = ratio_o * Rh_fw[t]
			Rc_fw[t] += ratio_c_cur * Rh_fw[t]

			ratio_f, ratio_c_prev = lrp_product(self.gates_fw[t, idx_f], self.c_fw[t-1], eps, method)
			first = lrp_linear(self.gates_fw[t, idx_f]*self.c_fw[t-1], np.identity(d), np.zeros((d)), self.c_fw[t], Rc_fw[t], 2*d, eps, bias_factor)
			Rf_fw[t] = ratio_f * first
			Rc_fw[t-1] = ratio_c_prev * first

			ratio_i, ratio_g = lrp_product(self.gates_fw[t, idx_i], self.gates_fw[t, idx_g], eps, method)
			second = lrp_linear(self.gates_fw[t, idx_i]*self.gates_fw[t, idx_g], np.identity(d), np.zeros((d)), self.c_fw[t], Rc_fw[t], 2*d, eps, bias_factor)
			Ri_fw[t] = ratio_i * second
			Rg_fw[t] = ratio_g * second

			if linear == 'epsilon':
				rx_o_fw = lrp_linear(self.x[t], self.Wih_fw[idx_o].transpose(), self.bih_fw[idx_o]+self.bhh_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], d+self.embed_size, eps, bias_factor)
				rx_f_fw = lrp_linear(self.x[t], self.Wih_fw[idx_f].transpose(), self.bih_fw[idx_f]+self.bhh_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], d+self.embed_size, eps, bias_factor)
				rx_i_fw = lrp_linear(self.x[t], self.Wih_fw[idx_i].transpose(), self.bih_fw[idx_i]+self.bhh_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], d+self.embed_size, eps, bias_factor)
				rx_g_fw = lrp_linear(self.x[t], self.Wih_fw[idx_g].transpose(), self.bih_fw[idx_g]+self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d+self.embed_size, eps, bias_factor)

				rh_o_fw = lrp_linear(self.h_fw[t-1], self.Whh_fw[idx_o].transpose(), self.bih_fw[idx_o]+self.bhh_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], d+self.embed_size, eps, bias_factor)
				rh_f_fw = lrp_linear(self.h_fw[t-1], self.Whh_fw[idx_f].transpose(), self.bih_fw[idx_f]+self.bhh_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], d+self.embed_size, eps, bias_factor)
				rh_i_fw = lrp_linear(self.h_fw[t-1], self.Whh_fw[idx_i].transpose(), self.bih_fw[idx_i]+self.bhh_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], d+self.embed_size, eps, bias_factor)
				rh_g_fw = lrp_linear(self.h_fw[t-1], self.Whh_fw[idx_g].transpose(), self.bih_fw[idx_g]+self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d+self.embed_size, eps, bias_factor)
			else:
				# Rx_o = lrp_linear_split(self.x[t], self.Wih_fw[idx_o], self.bih_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], eps, bias_factor)
				# Rx_f = lrp_linear_split(self.x[t], self.Wih_fw[idx_f], self.bih_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], eps, bias_factor)
				# Rx_i = lrp_linear_split(self.x[t], self.Wih_fw[idx_i], self.bih_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], eps, bias_factor)
				# Rx_g = lrp_linear_split(self.x[t], self.Wih_fw[idx_g], self.bih_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], eps, bias_factor)

				# Rh_o = lrp_linear_split(self.h_fw[t-1], self.Whh_fw[idx_o], self.bhh_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], eps, bias_factor)
				# Rh_f = lrp_linear_split(self.h_fw[t-1], self.Whh_fw[idx_f], self.bhh_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], eps, bias_factor)
				# Rh_i = lrp_linear_split(self.h_fw[t-1], self.Whh_fw[idx_i], self.bhh_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], eps, bias_factor)
				# Rh_g = lrp_linear_split(self.h_fw[t-1], self.Whh_fw[idx_g], self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], eps, bias_factor)

				rx_o_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_o].transpose(), self.bih_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)
				rx_f_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_f].transpose(), self.bih_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)
				rx_i_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_i].transpose(), self.bih_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)
				rx_g_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_g].transpose(), self.bih_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)

				rh_o_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_o].transpose(), self.bhh_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], d, linear, alpha, eps_linear, bias_factor)
				rh_f_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_f].transpose(), self.bhh_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], d, linear, alpha, eps_linear, bias_factor)
				rh_i_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_i].transpose(), self.bhh_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], d, linear, alpha, eps_linear, bias_factor)
				rh_g_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_g].transpose(), self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d, linear, alpha, eps_linear, bias_factor)

				# rx_o_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_o].transpose(), self.bih_fw[idx_o]+self.bhh_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rx_f_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_f].transpose(), self.bih_fw[idx_f]+self.bhh_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rx_i_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_i].transpose(), self.bih_fw[idx_i]+self.bhh_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rx_g_fw = lrp_linear_new(self.x[t], self.Wih_fw[idx_g].transpose(), self.bih_fw[idx_g]+self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)

				# rh_o_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_o].transpose(), self.bih_fw[idx_o]+self.bhh_fw[idx_o], self.gates_pre_fw[t, idx_o], Ro_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rh_f_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_f].transpose(), self.bih_fw[idx_f]+self.bhh_fw[idx_f], self.gates_pre_fw[t, idx_f], Rf_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rh_i_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_i].transpose(), self.bih_fw[idx_i]+self.bhh_fw[idx_i], self.gates_pre_fw[t, idx_i], Ri_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rh_g_fw = lrp_linear_new(self.h_fw[t-1], self.Whh_fw[idx_g].transpose(), self.bih_fw[idx_g]+self.bhh_fw[idx_g], self.gates_pre_fw[t, idx_g], Rg_fw[t], d+self.embed_size, linear, eps_linear, bias_factor)


			Rx_fw[t] = rx_o_fw + rx_f_fw + rx_i_fw + rx_g_fw
			Rh_fw[t-1] = rh_o_fw + rh_f_fw + rh_i_fw + rh_g_fw

			# backward
			ratio_o, ratio_c_cur = lrp_product(self.gates_bw[t, idx_o], np.tanh(self.c_bw[t]), eps, method)
			Ro_bw[t] = ratio_o * Rh_bw[t]
			Rc_bw[t] += ratio_c_cur * Rh_bw[t]

			ratio_f, ratio_c_prev = lrp_product(self.gates_bw[t, idx_f], self.c_bw[t-1], eps, method)
			first = lrp_linear(self.gates_bw[t, idx_f]*self.c_bw[t-1], np.identity(d), np.zeros((d)), self.c_bw[t], Rc_bw[t], 2*d, eps, bias_factor)
			Rf_bw[t] = ratio_f * first
			Rc_bw[t-1] = ratio_c_prev * first

			ratio_i, ratio_g = lrp_product(self.gates_bw[t, idx_i], self.gates_bw[t, idx_g], eps, method)
			second = lrp_linear(self.gates_bw[t, idx_i]*self.gates_bw[t, idx_g], np.identity(d), np.zeros((d)), self.c_bw[t], Rc_bw[t], 2*d, eps, bias_factor)
			Ri_bw[t] = ratio_i * second
			Rg_bw[t] = ratio_g * second

			if linear == 'epsilon':
				rx_o_bw = lrp_linear(self.x_rev[t], self.Wih_bw[idx_o].transpose(), self.bih_bw[idx_o]+self.bhh_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], d+self.embed_size, eps, bias_factor)
				rx_f_bw = lrp_linear(self.x_rev[t], self.Wih_bw[idx_f].transpose(), self.bih_bw[idx_f]+self.bhh_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], d+self.embed_size, eps, bias_factor)
				rx_i_bw = lrp_linear(self.x_rev[t], self.Wih_bw[idx_i].transpose(), self.bih_bw[idx_i]+self.bhh_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], d+self.embed_size, eps, bias_factor)
				rx_g_bw = lrp_linear(self.x_rev[t], self.Wih_bw[idx_g].transpose(), self.bih_bw[idx_g]+self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d+self.embed_size, eps, bias_factor)

				rh_o_bw = lrp_linear(self.h_bw[t-1], self.Whh_bw[idx_o].transpose(), self.bih_bw[idx_o]+self.bhh_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], d+self.embed_size, eps, bias_factor)
				rh_f_bw = lrp_linear(self.h_bw[t-1], self.Whh_bw[idx_f].transpose(), self.bih_bw[idx_f]+self.bhh_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], d+self.embed_size, eps, bias_factor)
				rh_i_bw = lrp_linear(self.h_bw[t-1], self.Whh_bw[idx_i].transpose(), self.bih_bw[idx_i]+self.bhh_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], d+self.embed_size, eps, bias_factor)
				rh_g_bw = lrp_linear(self.h_bw[t-1], self.Whh_bw[idx_g].transpose(), self.bih_bw[idx_g]+self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d+self.embed_size, eps, bias_factor)
			else:
				# Rx_o = lrp_linear_split(self.x_rev[t], self.Wih_bw[idx_o], self.bih_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], eps, bias_factor)
				# Rx_f = lrp_linear_split(self.x_rev[t], self.Wih_bw[idx_f], self.bih_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], eps, bias_factor)
				# Rx_i = lrp_linear_split(self.x_rev[t], self.Wih_bw[idx_i], self.bih_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], eps, bias_factor)
				# Rx_g = lrp_linear_split(self.x_rev[t], self.Wih_bw[idx_g], self.bih_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], eps, bias_factor)

				# Rh_o = lrp_linear_split(self.h_bw[t-1], self.Whh_bw[idx_o], self.bhh_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], eps, bias_factor)
				# Rh_f = lrp_linear_split(self.h_bw[t-1], self.Whh_bw[idx_f], self.bhh_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], eps, bias_factor)
				# Rh_i = lrp_linear_split(self.h_bw[t-1], self.Whh_bw[idx_i], self.bhh_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], eps, bias_factor)
				# Rh_g = lrp_linear_split(self.h_bw[t-1], self.Whh_bw[idx_g], self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], eps, bias_factor)

				rx_o_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_o].transpose(), self.bih_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)
				rx_f_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_f].transpose(), self.bih_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)
				rx_i_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_i].transpose(), self.bih_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)
				rx_g_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_g].transpose(), self.bih_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], self.embed_size, linear, alpha, eps_linear, bias_factor)

				rh_o_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_o].transpose(), self.bhh_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], d, linear, alpha, eps_linear, bias_factor)
				rh_f_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_f].transpose(), self.bhh_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], d, linear, alpha, eps_linear, bias_factor)
				rh_i_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_i].transpose(), self.bhh_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], d, linear, alpha, eps_linear, bias_factor)
				rh_g_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_g].transpose(), self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d, linear, alpha, eps_linear, bias_factor)


				# rx_o_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_o].transpose(), self.bih_bw[idx_o]+self.bhh_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rx_f_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_f].transpose(), self.bih_bw[idx_f]+self.bhh_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rx_i_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_i].transpose(), self.bih_bw[idx_i]+self.bhh_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rx_g_bw = lrp_linear_new(self.x_rev[t], self.Wih_bw[idx_g].transpose(), self.bih_bw[idx_g]+self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)

				# rh_o_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_o].transpose(), self.bih_bw[idx_o]+self.bhh_bw[idx_o], self.gates_pre_bw[t, idx_o], Ro_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rh_f_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_f].transpose(), self.bih_bw[idx_f]+self.bhh_bw[idx_f], self.gates_pre_bw[t, idx_f], Rf_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rh_i_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_i].transpose(), self.bih_bw[idx_i]+self.bhh_bw[idx_i], self.gates_pre_bw[t, idx_i], Ri_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
				# rh_g_bw = lrp_linear_new(self.h_bw[t-1], self.Whh_bw[idx_g].transpose(), self.bih_bw[idx_g]+self.bhh_bw[idx_g], self.gates_pre_bw[t, idx_g], Rg_bw[t], d+self.embed_size, linear, eps_linear, bias_factor)
			

			Rx_bw[t] = rx_o_bw + rx_f_bw + rx_i_bw + rx_g_bw
			Rh_bw[t-1] = rh_o_bw + rh_f_bw + rh_i_bw + rh_g_bw



		return Rx_fw, Rx_bw[::-1, :], Rh_fw[-1].sum()+Rc_fw[-1].sum()+Rh_bw[-1].sum()+Rc_bw[-1].sum()
		# return Rx_fw, Rx_bw[::-1, :], (Rh_fw[:-1], Rh_bw[:-1][::-1, :], Rc_fw[:-1], Rc_bw[:-1][::-1, :])











