import numpy as np


class FF(object):
	"""A simple FeedForward neural network"""
	def __init__(self, layerDims):
		super(FF, self).__init__()
		n_weights = len(layerDims)-1
		self.weights = []
		for i in range(n_weights):
			self.weights.append(0.1*np.random.randn(layerDims[i+1], layerDims[i]))

	def sgd(self, X, y, epochs, eta, mb_size, Xtest, ytest):
		N = X.shape[1]
		n_mbs = int(np.ceil(N/mb_size))
		acc = self.eval_test(Xtest, ytest)

		updates = 0
		steps = [updates]
		test_acc = [acc]
		print("Starting training, test accuracy: {0}".format(acc))

		for i in range(epochs):
			perm = np.random.permutation(N);
			for j in range(n_mbs):
				X_mb = X[:,perm[j*mb_size:(j+1)*mb_size]]
				y_mb = y[:,perm[j*mb_size:(j+1)*mb_size]]

				grads = self.backprop(X_mb, y_mb)

				for k,grad in enumerate(grads):
					self.weights[k] = self.weights[k] - (eta/mb_size)*grad

				updates = updates + 1
				if updates%50 == 0:
					steps.append(updates)
					test_acc.append(self.eval_test(Xtest, ytest))

			acc = self.eval_test(Xtest, ytest)
			print("Done epoch {0}, test accuracy: {1}".format(i+1, acc))

		steps = np.asarray(steps)
		steps = steps/n_mbs

		return steps, test_acc

	def backprop(self,X,y):
		# X is a matrix of size input_dim*mb_size
		# y is a matrix of size output_dim*mb_size
		# you should return a list 'grads' of length(weights) such
		# that grads[i] is a matrix containing the gradients of the
		# loss with respect to weights[i].

		# ForwardPass
		w = []
		h = []
		s = []

		for i in range(len(self.weights)):
			w.append(self.weights[i])

		s.append(np.array(X))
		for i in range(len(self.weights)):
			h.append(np.dot(w[i], s[i]))
			s.append(FF.activation(h[-1]))

		# BackwardPass
		d = [0] * len(self.weights)

		loss_grad = FF.loss_deriv(s[-1], y)
		d[-1] = loss_grad * FF.activation_deriv(h[-1])

		n = len(self.weights)
		for i in range(n-1):
			d[n - i - 2] = (w[n-i-1].T @ d[n-i-1]) * FF.activation_deriv(h[n-i-2])

		# Gradients
		dw = []
		for i in range(len(self.weights)):
			dw.append(d[i] @ s[i].T)

		return dw
		
		# YOUR CODE HERE
		# w1 = self.weights[0]
		# w2 = self.weights[1]
		#
		# s0 = np.array(X)
		#
		# h1 = np.dot(w1, s0)
		# s1 = FF.activation(h1)
		#
		# h2 = np.dot(w2, s1)
		# s2 = FF.activation(h2)
		#
		# loss_grad = FF.loss_deriv(s2, y)
		# d2 = loss_grad * FF.activation_deriv(h2)
		#
		# d1 = (w2.T @ d2) * FF.activation_deriv(h1)
		#
		# dw2 = d2 @ s1.T
		# dw1 = d1 @ s0.T
		#
		# return [dw1, dw2]

	def predict(self,x):
		a = x
		for w in self.weights:
			a = FF.activation(np.dot(w,a))

		return a

	def eval_test(self,Xtest, ytest):
		ypred = self.predict(Xtest)
		ypred = ypred==np.max(ypred,axis=0)
		
		return np.mean(np.all(ypred==ytest,axis=0))

	@staticmethod
	def activation(x):
		return np.tanh(x)

	@staticmethod
	def activation_deriv(x):
		return 1-(np.tanh(x)**2)

	@staticmethod
	def loss_deriv(output, target):
		# Derivative of loss function with respect to the activations
		# in the output layer.
		# we use quadratic loss, where L=0.5*||output-target||^2
		
		# YOUR CODE HERE
		return output-target
