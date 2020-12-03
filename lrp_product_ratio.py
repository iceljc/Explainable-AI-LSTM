import numpy as np
from numpy import newaxis as na

def sigmoid(x, a=-2):
	return np.power(1.0 + np.exp(-x), a)

def lrp_product(z_gate, z_signal, eps, method='all'):

	if method == 'all':
		return 0.0, 1.0

	elif method == 'relative':
		numerator_gate = sigmoid(z_gate)
		numerator_signal = sigmoid(z_signal)
		denominator = sigmoid(z_gate) + sigmoid(z_signal)

	elif method == 'abs':

		numerator_gate = sigmoid(abs(z_gate))
		numerator_signal = sigmoid(abs(z_signal))
		denominator = numerator_gate + numerator_signal

	elif method == 'skewed':
		gate_sig = sigmoid(z_gate)
		signal_sig = sigmoid(z_signal)

		gate_abs = sigmoid(abs(z_gate))
		signal_abs = sigmoid(abs(z_signal))

		a1 = gate_sig / (gate_sig + signal_sig)
		a2 = signal_sig / (gate_sig + signal_sig)

		b1 = gate_abs / (gate_abs + signal_abs)
		b2 = signal_abs / (gate_abs + signal_abs)

		denominator = a1 + a2 + b1 + b2
		numerator_gate = np.array([min(i, j, k, m) for i, j, k, m in zip(a1, a2, b1, b2)])
		return numerator_gate / denominator, 1 - numerator_gate / denominator

	elif method == 'avg':
		gate_sig = sigmoid(z_gate)
		signal_sig = sigmoid(z_signal)

		gate_abs = sigmoid(abs(z_gate))
		signal_abs = sigmoid(abs(z_signal))

		a1 = gate_sig / (gate_sig + signal_sig)
		a2 = signal_sig / (gate_sig + signal_sig)

		b1 = gate_abs / (gate_abs + signal_abs)
		b2 = signal_abs / (gate_abs + signal_abs)

		return (a1 + b1) / 2, (a2 + b2) / 2

	
	return numerator_gate / denominator, numerator_signal / denominator







