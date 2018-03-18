# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
	"""
	General-purpose module to encode a sequence using a RNN.
	It feeds the input through a RNN and returns all the hidden states.

	Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
	to get a single, fixed size vector representation of a sequence
	(e.g. by taking element-wise max of hidden states).
	Here, we're using the RNN as an "encoder" but we're not taking max;
	we're just returning all the hidden states. The terminology "encoder"
	still applies because we're getting a different "encoding" of each
	position in the sequence, and we'll use the encodings downstream in the model.

	This code uses a bidirectional GRU, but you could experiment with other types of RNN.
	"""

	def __init__(self, hidden_size, keep_prob):
		"""
		Inputs:
		  hidden_size: int. Hidden size of the RNN
		  keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
		"""
		self.hidden_size = hidden_size
		self.keep_prob = keep_prob
		self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
		self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
		self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
		self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

	def build_graph(self, inputs, masks):
		"""
		Inputs:
		  inputs: Tensor shape (batch_size, seq_len, input_size)
		  masks: Tensor shape (batch_size, seq_len).
			Has 1s where there is real input, 0s where there's padding.
			This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

		Returns:
		  out: Tensor shape (batch_size, seq_len, hidden_size*2).
			This is all hidden states (fw and bw hidden states are concatenated).
		"""
		with vs.variable_scope("RNNEncoder"):
			input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

			# Note: fw_out and bw_out are the hidden states for every timestep.
			# Each is shape (batch_size, seq_len, hidden_size).
			(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

			# Concatenate the forward and backward hidden states
			out = tf.concat([fw_out, bw_out], axis=2)

			# Apply dropout
			out = tf.nn.dropout(out, self.keep_prob)

			return out


class SimpleSoftmaxLayer(object):
	"""
	Module to take set of hidden states, (e.g. one for each context location),
	and return probability distribution over those states.
	"""

	def __init__(self):
		pass

	def build_graph(self, inputs, masks):
		"""
		Applies one linear downprojection layer, then softmax.

		Inputs:
		  inputs: Tensor shape (batch_size, seq_len, hidden_size)
		  masks: Tensor shape (batch_size, seq_len)
			Has 1s where there is real input, 0s where there's padding.

		Outputs:
		  logits: Tensor shape (batch_size, seq_len)
			logits is the result of the downprojection layer, but it has -1e30
			(i.e. very large negative number) in the padded locations
		  prob_dist: Tensor shape (batch_size, seq_len)
			The result of taking softmax over logits.
			This should have 0 in the padded locations, and the rest should sum to 1.
		"""
		with vs.variable_scope("SimpleSoftmaxLayer"):

			# Linear downprojection layer
			logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
			logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

			# Take softmax over sequence
			masked_logits, prob_dist = masked_softmax(logits, masks, 1)

			return masked_logits, prob_dist


class BaselineOutputLayer(object):
  	""" Module for the output layer used by the baseline model """

	def __init__(self, hidden_size):
		"""
		Inputs:
		  hidden_size: the size of the hidden states
		"""
		self.hidden_size = hidden_size

	def build_graph(self, blended_reps, context_mask):
		""" Adds the output layer for the baseline model to the TensorFlow graph

		Inputs:
		  output_reps: A tensor object containing the final blended representations
			  from the output layer
		  context_mask: Has 1s where there is real input, 0s where there's padding.
			  Tensor shape (batch_size, num_context)
		
		Outputs:
		  results: a dictionary holding the logits and probability distributions for
			  the start and end locations.
		"""
		# Combine the outputs from all layers to create a blended representation
		blended_reps = tf.concat(model_outputs, axis=2)

		# Apply fully connected layer to each blended representation
		# Note, blended_reps_final corresponds to b' in the handout
		# Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
		output_reps = tf.contrib.layers.fully_connected(blended_reps, 
														num_outputs=self.hidden_size,
														activation_fn=tf.nn.tanh) # Tensor shape (batch_size, context_len, hidden_size)

		# Use softmax layer to compute probability distribution for start location
		# Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
		with vs.variable_scope("StartDist"):
			softmax_layer_start = SimpleSoftmaxLayer()
			logits_start, probdist_start = softmax_layer_start.build_graph(output_reps, context_mask)

		# Use softmax layer to compute probability distribution for end location
		# Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
		with vs.variable_scope("EndDist"):
			softmax_layer_end = SimpleSoftmaxLayer()
			logits_end, probdist_end = softmax_layer_end.build_graph(output_reps, context_mask)

		results = {"logits_start": logits_start, 
				   "probdist_start": probdist_start,
				   "logits_end": logits_end, 
				   "probdist_end": probdist_end}
		return results



class DynamicPointingDecoder(object):
	""" Module for the Dynamic Pointing Decoder

	Implements the Dynamic Pointing Decoder described in "Dynamic Coattention Networks
	for Question Answering" by Xiong et al in 2017.
	"""

	def __init__(self, keep_prob, encoded_size):
		"""
		Inputs:
		  encoded_size: int, the size of the coattention encoded hidden states
		  keep_prob: scalar Tensor containing the keep probability for dropout

		In addition to these inputs, an LSTM cell for the decoder is initialized here
		Dropout can be added to the LSTM by specifying use_dropout=True in the build_graph
		method.
		"""
		self.keep_prob = keep_prob
		self.encoded_size = encoded_size
		self.lstm_cell = rnn_cell.LSTMCell(self.encoded_size) 


	def __setup_variables(self):
		""" Initializes the model parameters for the Dynamic Pointing Decoder """
		# TODO: Implement this function
		pass


	def __compute_start_positions(self):
		""" Computes the index of the start position for the answer span """
		# TODO: Implement this function
		pass

	def __compute_end_positions(self):
		""" Computes the index of the end position for the answer span """
		# TODO: Implement this function
		pass


	def __compute_hidden_states(self, encodings, starts, ends):
		""" Computes the hidden states using an decoder LSTM """
		# TODO: Implement this function
		pass


	def build_graph(encodings):
		""" Adds the Dynamic Pointing Decoder to the TensorFlow graph """
		# TODO: Implement this function
		starts = self.__compute_start_positions()
		ends = self.__compute_end_positions()
		hidden_states = self.__compute_hidden_states(encodings, starts, ends)
		pass


class Coattention(object):
	""" Module for Cooattention layer 

	Implements the Coattention layer described in "Dynamic Coattention Networks
	for Question Answering" by Xiong et al in 2017.
	"""
	def __init__(self, keep_prob, hidden_size):
		"""
		Inputs:
	  		hidden_size: int, the size of the encoder hidden states passed to the
				   the coattention layer
	  		keep_prob: Tensor containing a single scalar that is the keep 
				  probability (for dropout)

		In addition to storing these inputs, LSTM cells for the bidrectional
		LSTM used to generate the coattention encodings are initialized here.
		Dropout is added to the LSTM cells by default, but can be turned off by 
		specifying use_dropout=False in the build_graph method.
		"""
		self.keep_prob = keep_prob
		self.hidden_size = hidden_size
		self.rnn_cell_fw = rnn_cell.LSTMCell(2 * self.hidden_size)
		self.rnn_cell_bw = rnn_cell.LSTMCell(2 * self.hidden_size)

  
	def __setup_variables(self):
		""" Return a list of Tensor objects storing the model parameters.

		Initializes the model parameters associated with the Coattention layer.

		W: Weight matrix used for transforming question hidden states
			Tensor shape (hidden_size, hidden_size)
		b: Bias vector used for transforming question hidden states
			Tensor shape (hidden_size)
		c0: Sentinel context hidden state (makes it possible to attend to none
			of the provided context hidden states)
			Tensor shape (hidden_size)
		q0: Sentinel question hidden state (makes it possible to attend to none
			of the provided question hidden states)
			Tensor shape (hidden_size)
		
		Inputs:
		  None

		Outputs:
		  weights: a list containing the model parameters W, b, c0, and q0
		"""
		xavier = tf.contrib.layers.xavier_initializer()
		W = tf.get_variable(name='W', shape=[self.hidden_size, self.hidden_size], initializer=xavier)
		b = tf.get_variable(name='b', shape=[self.hidden_size], initializer=xavier)
		c0 = tf.get_variable(name='c0', shape=[self.hidden_size], initializer=xavier)          
		q0 = tf.get_variable(name='q0', shape=[self.hidden_size], initializer=xavier)         
		weights = [W, b, c0, q0]
		return weights 
  

	def __compute_projected_hidden_states(self, weights, context, question, question_mask):
		"""
		Computes the projected question hidden states and includes the sentinel vectors.
		
		The projected question hidden states are given by
		
			  q'_j = tanh(W q_j + b) for j in {1, 2, ..., M}
		
		The set of the context and question hidden states are extended to include 
		the sentinel vectors c0 and q0, respectively.
		
		Inputs:
		  weights: a list containing the model parameters:
				W: Tensor shape (hidden_size, hidden_size)
				b: Tensor shape (hidden_size)
				c0: Tensor shape (hidden_size)
				q0: Tensor shape (hidden_size)
		  context: the context hidden states
				Tensor shape (batch_size, num_context, hidden_size)
		  question: the question hidden states
			  Tensor shape (batch_size, num_question, hidden_size)
		  question_mask: mask for the question hidden states
			  Has 1s where there is real input, 0s where there's padding
			  Tensor shape (batch_size, num_question)
		  
		Outputs:
		  c: context hidden states and c0             
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		  q: projected question hidden states and q0
			  Tensor shape (batch_size, num_question + 1, hidden_size) 
		.  q_mask: boolean mask corresponding to the projected question hidden states
			  Tensor shape (batch_size, num_question + 1)  
		"""
		# Batch size is the same for context, question, and question_mask
		batch_size = tf.shape(context)[0]
		W, b, c0, q0 = weights

		# Append c0 to the set of context hidden states for each example in the batch
		c0_tile = tf.reshape(tf.tile(c0, multiples=[batch_size]), 
							 shape=[batch_size, -1, self.hidden_size])    # Tensor shape (batch_size,               1, hidden_size)
		c = tf.concat([context, c0_tile], axis=1)                         # Tensor shape (batch_size, num_context + 1, hidden_size)

		# Append q0 to the set of question hidden states for example in the batch
		q0_tile = tf.reshape(tf.tile(q0, multiples=[batch_size]), 
							 shape=[batch_size, -1, self.hidden_size])    # Tensor shape (batch_size,                1, hidden_size)
		qp = tf.tanh(tf.tensordot(question, W, axes=[[2], [0]]) + b)      # Tensor shape (batch_size,     num_question, hidden_size)
		q = tf.concat([qp, q0_tile], axis=1)                              # Tensor shape (batch_size, num_question + 1, hidden_size)

		# Append 1 to the question hidden state mask
		one_tile = tf.reshape(tf.tile([1], multiples=[batch_size]), 
							  shape=[batch_size, -1])                     # Tensor shape (batch_size,                1)
		q_mask = tf.concat([question_mask, one_tile], axis=1)             # Tensor shape (batch_size, num_question + 1)
		return c, q, q_mask 


	def __compute_affinity_scores(self, c, q):
		"""
		Computes the matrix of affinity scores. 
		
		The affinity score for some given context and quesiton hidden states is
		
			  L_ij = c_i q_j^T
		
		Note: c_i and q_j are row vectors, not column vectors.
		
		Inputs:
		  c: the context hidden states + sentinel context hidden state
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		  q: the projected question hidden states + sentinel question hidden state
			  Tensor shape (batch_size, num_question + 1, hidden_size)
		
		Outputs:
		  L: the matrix of affinity scores
			  Tensor shape (batch_size, num_context + 1, num_question + 1)
		"""
		q_t = tf.transpose(q, perm=[0, 2, 1])    # Tensor shape (batch_size,     hidden_size, num_question + 1)
		L = tf.matmul(c, q_t)                    # Tensor shape (batch_size, num_context + 1, num_question + 1)
		return L


	def __compute_c2q_outputs(self, L, q, q_mask):
		"""
		Computes the Context-to-Question (C2Q) attention outputs.  
		
		First, the attention distribution is computed as
		
			  alpha^{i} = softmax(L_{i,:}) for i in {1, 2, ..., N+1}
		
		where L is the matrix of affinity scores given by
		
			  L_{ij} = c_{i} q_{j}^T
		
		Note: the context and question hidden states are row vectors, not column vectors
		
		The attention outputs are a sum of the projected question hidden states weighted
		by the attention distribution:
		
			  a_{i} = sum_{j=1}^{M+1} alpha_{j}^{i} q'_{j} for i in {1, 2, ..., N+1}
		
		Inputs:
		  L: the matrix of affinity scores
			  Tensor shape (batch_size, num_context + 1, num_question + 1)
		  q: the projected question hidden states + sentinel question hidden state
			  Tensor shape (batch_size, num_question + 1, hidden_size)
		.  q_mask: boolean mask corresponding to the projected question hidden states
			  Tensor shape (batch_size, num_question + 1)        
		
		Outputs:
		  c2q_output: the C2Q attention outputs
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		"""
		c2q_logits_mask = tf.expand_dims(q_mask, axis=1)            # Tensor shape (batch_size,               1, num_question + 1)
		_, c2q_dist = masked_softmax(L, c2q_logits_mask, dim=2)     # Tensor shape (batch_size, num_context + 1, num_question + 1)
		c2q_output = tf.matmul(c2q_dist, q)                         # Tensor shape (batch_size, num_context + 1,      hidden_size)             
		return c2q_dist, c2q_output


	def __compute_q2c_outputs(self, L, c):
		"""
		Computes the Question-to-Context (Q2C) attention outputs.  

		First, the attention distribution is computed as
		
			  beta^{j} = softmax(L_{:,j}) for j in {1, 2, ..., M+1}
		
		where L is the matrix of affinity scores given by
		
			  L_{ij} = c_{i} q_{j}^{T}
		
		The attention outputs are a sum of the context hidden states weighted
		by the attention distribution:
		
		  b_{j} = sum_{i=1}^{N+1} beta_{i}^{j} c_{i} for j in {1, 2, ..., M+1}
		
		Inputs:
		  L: the matrix of affinity scores
			  Tensor shape (batch_size, num_context + 1, num_question + 1)
		  c: the context hidden states + sentinel context hidden state
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		
		Outputs:
		  q2c_output: The Q2C attention outputs
			  Tensor shape (batch_size, num_question + 1, hidden_size)
		"""
		# Kinda jank, but softmax only works on last axis so need to tranpose
		L_t = tf.transpose(L, perm=[0, 2, 1])         # Tensor shape (batch_size, num_question + 1,  num_context + 1)
		q2c_dist_t = tf.nn.softmax(L_t)               # Tensor shape (batch_size, num_question + 1,  num_context + 1)
		q2c_dist = tf.transpose(L, perm=[0, 2, 1])    # Tensor shape (batch_size,  num_context + 1, num_question + 1)
		q2c_output = tf.matmul(q2c_dist_t, c)         # Tensor shape (batch_size, num_question + 1,      hidden_size)
		return q2c_dist, q2c_output



	def __compute_second_level_outputs(self, alpha, b):
		"""
		Computes the second level attention outputs. 

		The second-level attention outputs are given by
		
		  s_{i} = sum_{j=1}^{M+1} alpha_{j}^{i} b_{j} for i in {1, ..., N}
		
		Inputs:
		  alpha: the C2Q attention distributions
			  Tensor shape (batch_size, num_context + 1, num_question + 1)
		  b: the Q2C attention outputs
			  Tensor shape (batch_size, num_question + 1, hidden_size)
		
		Outputs:
		  s: The second-level attention outputs
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		"""
		s = tf.matmul(alpha, b)             # Tensor shape (batch_size, num_context + 1, hidden_size)
		return s


	def __compute_coattention_encodings(self, a, s, use_dropout):
		"""
		Computes the coattention encoding from the C2Q and second-level
		attention outputs:
		
		  u = {u_{1}, ..., u_{N}} = biLSTM({[s_{1} ; a_{1}] , ..., [S_{N} ; a_{N}]})
		
		Inputs:
		  a: the C2Q attention outputs
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		  s: the second-level attention outputs
			  Tensor shape (batch_size, num_context + 1, hidden_size)
		  use_dropout: boolean flag, adds dropout to the biLSTM if True and does 
			  nothing otherwise
		
		Outputs:
		  encodings: the coattention encodings
			  Tensor shape (batch_size, num_context, 2 * hidden_size)
		"""
		# Remove rows corresponding to sentinel vectors
		s = s[:,:-1,:]                        # Tensor shape (batch_size, num_context,     hidden_size)
		a = a[:,:-1,:]                        # Tensor shape (batch_size, num_context,     hidden_size)
		inputs = tf.concat([s, a], axis=2)    # Tensor shape (batch_size, num_context, 2 * hidden_size)

		if use_dropout:
			self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
			self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
		(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_cell_fw, 
															  cell_bw=self.rnn_cell_bw, 
															  inputs=inputs,
															  dtype=tf.float32)
		encodings = tf.concat([fw_out, bw_out], axis=2)     # Tensor shape (batch_size, num_context, 2 * hidden_size)
		return encodings


	def build_graph(self, context, question, question_mask, use_dropout=True):
		"""
		Adds a coattention layer to the TensorFlow graph.
		
		Inputs:
		  context: the context hidden states 
			  Tensor shape (batch_size, num_context, hidden_size)
		  question: the question hidden states
			  Tensor shape (batch_size, num_question, hidden_size)
		  question_mask: mask for the question hidden states.  Has 1s where
						 there is real input, 0s where there's padding
			  Tensor shape (batch_size, num_question)
		  use_dropout: boolean flag, adds dropout to the BiLSTM used for
			  computing the final coattention encoding if True and does 
			  nothing otherwise
		
		Outputs:
		  final_output: the coattention encoding
			  Tensor shape (batch_size, num_context, 2 * hidden_size)
		"""
		with vs.variable_scope("Coattention"):

			weights = self.__setup_variables()
			c, q, q_mask = self.__compute_projected_hidden_states(weights, context, question, question_mask)
			L = self.__compute_affinity_scores(c, q)
			c2q_dist, c2q_output = self.__compute_c2q_outputs(L, q, q_mask)
			q2c_dist, q2c_output = self.__compute_q2c_outputs(L, c)
			second_output = self.__compute_second_level_outputs(c2q_dist, q2c_output)
			coattn_output = self.__compute_coattention_encodings(c2q_output, second_output, use_dropout)

			return coattn_output
	

class Bidaf(object):
	"""Module for Bidirectional Attention Flow layer

	Implements the Bidirectional Attention Flow layer described in the paper
	"Bi-directional Attention Flow for Machine Comprehension" by Seo et al in
	2017.

	"""

	def __init__(self, keep_prob, hidden_size):
		""" 
		Inputs:
			keep_prob: tensor containing a single scalar that is the keep 
					   probability (for dropout)
			hidden_size: size of the hidden states from the RNN encoder.

		Dropout is added to the output of the BiDAF layer by default, but can be 
		turned off by specifying use_dropout=False in the build_graph method.      
		"""
		self.keep_prob = keep_prob
		self.hidden_size = hidden_size             


	def __setup_variables(self):
		"""
		Initializes the weight variables for the bidirectional attention flow layer.
		
		In the handout, the only weight term appearing in the BiDAF layer is w_sim.
		But for efficiency purposes, we store w_sim as 3 separate weight vectors
		w_1, w_2, and w_3.  These are related to w_sim like so:
		
		  w_sim = [ w_1 ; w_2 ; w_3 ] 
		
		Inputs:
		  None
		
		Outputs:
		  weights: list containing the model parameters w_1, w_2, and w_3
		"""
		xavier = tf.contrib.layers.xavier_initializer()
		w_1 = tf.get_variable(name='w_1', shape=[self.hidden_size], initializer=xavier)
		w_2 = tf.get_variable(name='w_2', shape=[self.hidden_size], initializer=xavier)
		w_3 = tf.get_variable(name='w_3', shape=[self.hidden_size], initializer=xavier)
		weights = [w_1, w_2, w_3]
		return weights


	def __compute_similarity_scores(self, weights, context, question):
		"""
		Computes the similarity scores from the context and question hidden states.  

		Similarity scores are computed as

		  S_ij = w_sim^T [c_i ; q_j ; c_i o q_j] = w_sim^T v 

		where v denotes the concatenation of the vectors c_i, q_j, and c_i o q_j.

		Instead of forming v and taking the dot product with w_sim explicitly, we 
		break the calculation down into 3 parts using 3 different weight vectors: 

		  w_1 multiplies with context hidden states c_i
		  w_2 multiplies with question hidden states q_j
		  w_3 multiplies with the elementwise products c_i o q_j

		The similarity score for is then computed as

		  S_ij = w_1^T c_i + w_2^T q_j + w_3^T (c_i o q_j) = S1 + S2 + S3

		Broadcasting will take care of the dimensions.

		Inputs:
		  weights: a list containing the tf.Variables for w_1, w_2, w_3
		  context: Tensor containing the context hidden states, 
			  Tensor shape (batch_size, num_context, hidden_size)
		  question: Tensor containing the question hidden states, 
			  Tensor shape (batch_size, num_question, hidden_size)

		Returns:
		  S: The matrix of similarity scores, 
			  Tensor shape (batch_size, num_context, num_question)
		"""        
		# Compute elementwise products c_i o q_j
		c = tf.expand_dims(context, axis=2)       # Tensor shape (batch_size, num_context,            1, hidden_size)
		q = tf.expand_dims(question, axis=1)      # Tensor shape (batch_size,           1, num_question, hidden_size)
		e = c * q                                 # Tensor shape (batch_size, num_context, num_question, hidden_size)

		# Compute the similarity matrix
		w_1, w_2, w_3 = weights
		S1 = tf.tensordot(c, w_1, axes=[[3], [0]])     # Tensor shape (batch_size, num_context,            1)
		S2 = tf.tensordot(q, w_2, axes=[[3], [0]])     # Tensor shape (batch_size,           1, num_question)
		S3 = tf.tensordot(e, w_3, axes=[[3], [0]])     # Tensor shape (batch_size, num_context, num_question)
		S = S1 + S2 + S3                               # Tensor shape (batch_size, num_context, num_question)
		return S


	def __compute_c2q_outputs(self, S, question, question_mask):
		"""
		Computes the Context-to-Question (C2Q) attention outputs. 

		The attention distributions are given by
		
		  alpha_i = softmax(S_i,:) for i in {1, 2, ..., N}
		
		The C2Q attention outputs are a sum of the question hidden states weighted
		by the attention distributions:
		
		  a_i = sum_j alpha_j^i * q_j for i in {1, 2, ..., N}
		
		Inputs:
		  S: The matrix of similarity scores, 
			  Tensor shape (batch_size, num_context, num_question)
		  question: Tensor containing the question hidden states, 
			  Tensor shape (batch_size, num_question, hidden_size)
		  question_mask: Tensor shape (batch_size, num_question).
						  1s where there's real input, 0s where there's padding
		
		Returns:
		  c2q_dist: the C2Q attention distribution, 
			  Tensor shape (batch_size, num_context, num_question)
		  c2q_output: the C2Q attention output,     
			  Tensor shape (batch_size, num_context, hidden_size)
		"""                                            
		c2q_logits_mask = tf.expand_dims(question_mask, axis=1)    # Tensor shape (batch_size,           1, num_question)
		_, c2q_dist = masked_softmax(S, c2q_logits_mask, dim=2)    # Tensor shape (batch_size, num_context, num_question)
		c2q_output = tf.matmul(c2q_dist, question)                 # Tensor shape (batch_size, num_context,  hidden_size)
		return c2q_dist, c2q_output


	def __compute_q2c_outputs(self, S, context, context_mask):
		"""
		Compute the Question-to-Context (Q2C) attention outputs.  
		
		The Q2C ttention distribution is given by
		
		  m_i = max_j S_ij for i in {1, 2, ..., N}
		  beta = softmax(m)   
		
		The Q2C attention outputs are a sum of the context hidden states weighted
		by the attention distribution
		
		Inputs:
		  S: Tensor containing the similarity scores,           
			  Tensor shape (batch_size, num_context, num_question)
		  context: Tensor containing the context hidden states, 
			  Tensor shape (batch_size, num_context, hidden_size)
		  context_mask: Tensor shape (batch_size, num_context)
						  1s where there's real input, 0s where there's padding
		
		Returns:
		  q2c_dist: The Q2C attention distribution,   
			  Tensor shape (batch_size, num_context)
		  q2c_output: The Q2C attention output,       
			  Tensor shape (batch_size, 1, hidden_size) 
		"""

		m = tf.reduce_max(S, reduction_indices=[2])                 # Tensor shape (batch_size, num_context)
		_, q2c_dist = masked_softmax(m, context_mask, dim=1)        # Tensor shape (batch_size, num_context)
		q2c_dist = tf.expand_dims(q2c_dist, axis=1)                 # Tensor shape (batch_size, 1, num_context)
		q2c_output = tf.matmul(q2c_dist, context)                   # Tensor shape (batch_size, 1, hidden_size)
		return q2c_dist, q2c_output

		# m = tf.reduce_max(S, reduction_indices=[2])               # Tensor shape (batch_size, num_context)
		# q2c_dist = tf.nn.softmax(m)                               # Tensor shape (batch_size, num_context)  
		# q2c_dist = tf.expand_dims(q2c_dist, axis=1)               # Tensor shape (batch_size, 1, num_context)
		# q2c_output = tf.matmul(q2c_dist, context)                 # Tensor shape (batch_size, 1, hidden_size)
		# return q2c_dist, q2c_output


	def __form_bidaf_output(self, context, a, c, use_dropout):
		"""
		Forms the final output of the Bidirectional Attention Flow Layer
		
		  output_i = [c_i ; a_i ; c_i o a_i ; c_i o c']
		
		where o denotes elementwise multiplication and
		  
		  c_i are the context hidden states 
		  a_i are the C2Q attention outputs
		  c'  are the Q2C attention outputs
		
		Inputs:
		
		  context: the context hidden states, 
			  Tensor shape (batch_size, num_context, hidden_size)
		  a: the Context-to-Question attention outputs,       
			  Tensor shape (batch_size, num_context, hidden_size)
		  c: the Question-to-Context attention outputs,       
			  Tensor shape (batch_size, hidden_size)
		  use_dropout: boolean flag, adds dropout to the BiDAF output if True and does 
			  nothing otherwise
		
		Returns:
		
		  attn_output: The BiDAF output, 
			  Tensor shape (batch_size, num_context, 4 * hidden_size)
		"""
		attn_output = tf.concat([context, a, context * a, context * c], axis=2)  
		if use_dropout:
		  attn_output = tf.nn.dropout(attn_output, self.keep_prob)        
		return attn_output
		

	def build_graph(self, question, question_mask, context, context_mask, use_dropout=True):
		"""
		Adds a bidirectional attention flow layer to the Tensorflow graph.
		
		Inputs:
		  question: The question hidden states
			  Tensor shape (batch_size, num_question, hidden_size)
		  question_mask: Boolean mask for the question hidden states
			  1s where there's real input, 0s where there's padding
			  Tensor shape (batch_size, num_question).
		  context: The context hidden states
			  Tensor shape (batch_size, num_context, hidden_size)
		  use_dropout: boolean flag, adds dropout to the BiDAF output if True and does 
				  nothing otherwise
		
		Outputs:
			attn_output: The final output of the bidirectional attention flow layer
			  Tensor shape (batch_size, num_context, 4 * hidden_size)
		"""
		with vs.variable_scope("BiDAF"):

			weights = self.__setup_variables()
			S = self.__compute_similarity_scores(weights, context, question)           # Tensor shape (batch_size, num_context,    num_question)
			_, c2q_output = self.__compute_c2q_outputs(S, question, question_mask)     # Tensor shape (batch_size, num_context,     hidden_size)
			_, q2c_output = self.__compute_q2c_outputs(S, context, context_mask)       # Tensor shape (batch_size,           1,     hidden_size)
			attn_output = self.__form_bidaf_output(context, c2q_output, 
												 q2c_output, use_dropout)            # Tensor shape (batch_size, num_context, 4 * hidden_size)

			return attn_output


class BiLSTMEncoder(object):
	""" Module for a general Bidirectional LSTM encoder """

	def __init__(self, input_size, output_size):
		"""
		Inputs:
		  input_size: the dimension of the input states
		  output_size: the dimension of the output states for each direction
			of the BiLSTM.
		"""
		self.input_size = input_size
		self.output_size = output_size
		self.rnn_cell_fw = rnn_cell.LSTMCell(num_units=self.input_size, 
											 num_proj=self.output_size)
		self.rnn_cell_bw = rnn_cell.LSTMCell(num_units=self.input_size, 
											 num_proj=self.output_size)


	def build_graph(self, input_states):
		""" Computes an encoding for the input states

		Pushes the input states through a bidirectional LSTM, producing output
		states of size |output_size| in each direction.  These output states
		are concatenated to produce an encoding of the input states.

		Inputs:
		  input_states: the input states to the BiLSTM
			Tensor shape (batch_size, context_len, seq_len)

		Outputs:
		  output_states: the output states of the BiLSTM
			Tensor shape (batch_size, context_len, 2 * hidden_size)

		"""
		(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_cell_fw, 
															  cell_bw=self.rnn_cell_bw, 
															  inputs=input_states,
															  dtype=tf.float32)
		output_states = tf.concat([fw_out, bw_out], axis=2)
		return output_states


class BidafModelingLayer(object):
	""" Module for the Bidrectional Attention Flow modeling layer.

	Implements the modeling layer described in the paper "Bi-directional Attention 
	Flow for Machine Comprehension" by Seo et al in 2017.

	The modeling layer consists of two BiLSTMs.  The output size for each direction
	of the LSTMs is |hidden_size|.  The output returned by the modeling layer
	is concatenated outputs of from the second BiLSTM.
	"""

	def __init__(self, attn_size, hidden_size):
		"""
		Inputs:
		  attn_size: the size of the BiDAF encoder hidden states
		  hidden_size: the size of the hidden states

		"""
		self.attn_size = attn_size
		self.hidden_size = hidden_size

	def build_graph(self, attn_encodings):
		""" Adds the BiDAF modeling layer to the TensorFlow graph.

		Inputs:
		  attn_encodings: the BiDAF encoder hidden states
			Tensor shape (batch_size, context_len, 8 * hidden_size)   

		Outputs:
		  output_states: the output hidden states
			Tensor shape (batch_size, context_len, 2 * hidden_size)
		"""
		with vs.variable_scope("BidafModelingLayer"):
			# Layers must belong to different variable scopes in order to maintain different sets of weights
			with vs.variable_scope("Layer1"):
				layer_1 = BiLSTMEncoder(self.attn_size, self.hidden_size)         
				layer_1_output = layer_1.build_graph(attn_encodings)    # Tensor shape (batch_size, context_len, 2 * hidden_size)
			with vs.variable_scope("Layer2"):
				layer_2 = BiLSTMEncoder(2 * self.hidden_size, self.hidden_size)
				output_states = layer_2.build_graph(layer_1_output)     # Tensor shape (batch_size, context_len, 2 * hidden_size)
				return output_states


class BidafOutputLayer(object):
	""" Module for the Bidirectional Attention Flow output layer.

	Implements the output layer described in the paper "Bi-directional Attention
	Flow for Machine Comprehension" by Seo et al in 2017.
	"""

	def __init__(self, hidden_size):
		"""
		Inputs:
		  hidden_size: the size of the hidden states
		"""
		self.hidden_size = hidden_size


	def __setup_variables(self):
		""" Initializes the weight variables for the output layer.

		The output layer for the Bidaf model uses two weight vectors

		  w_p1: Tensor shape (10 * hidden_size)
		  w_p2: Tensor shape (10 * hidden_size)

		Inputs:
		  None

		Outputs:
		  weights: list containing the Tensor objects for w_p1 and w_p2
		"""
		xavier = tf.contrib.layers.xavier_initializer()
		w_p1 = tf.get_variable(name='w_1', shape=[10 * self.hidden_size], initializer=xavier)
		w_p2 = tf.get_variable(name='w_2', shape=[10 * self.hidden_size], initializer=xavier)
		weights = [w_p1, w_p2]
		return weights


	def build_graph(self, model_outputs):
		""" Adds the output layer for the Bidirectional Attention Flow model to the TensorFlow graph. 

		The output layer uses a softmax layer, followed by a BiLSTM layer, followed by
		another softmax layer.

		The first softmax layer computes the probability distribution for the
		position of the start index from the concatenated input [G; M], where
		|G| is the output from the Bidaf layer and |M| is the output from the modeling
		layer.

		Afterwards, |M| is pushed through the BiLSTM layer to produce another
		encoding |M_2|.

		The second softmax layer computes the probability distribution for the
		position of the end index from the concatenated input [G; M_2].

		Inputs:
		  model_outputs: a list of Tensors output from the modeling layer

		Outputs:
		  results: a dictionary containing the logits and probability distributions
			for the start and end indices.
		"""
		weights = self.__setup_variables()
		w_p1, w_p2 = weights
		G, M = model_outputs
		
		# Use softmax layer to compute probability distribution for start location
		with vs.variable_scope("StartDist"):
			blended_reps = tf.concat([G, M], axis=2)									# Tensor shape (batch_size, num_context, 10 * hidden_size)
			p1_logits = tf.tensordot(blended_reps, w_p1, axes=[[2], [0]])               # Tensor shape (batch_size, num_context)
		  	p1_dist = tf.nn.softmax(p1_logits)             								# Tensor shape (batch_size, num_context)

		# Pass M through another BiLSTM layer
		lstm_layer = BiLSTMEncoder(2 * self.hidden_size, self.hidden_size)
		M_2 = lstm_layer.build_graph(M)

		# Use softmax layer to compute probability distribution for end location
		with vs.variable_scope("EndDist"):
			blended_reps = tf.concat([G, M_2], axis=2)									# Tensor shape (batch_size, num_context, 10 * hidden_size)
		  	p2_logits = tf.tensordot(blended_reps, w_p2, axes=[[2], [0]])               # Tensor shape (batch_size, num_context)
		  	p2_dist = tf.nn.softmax(p2_logits)             								# Tensor shape (batch_size, num_context)

		results = {"logits_start": p1_logits,
				   "probdist_start": p1_dist,
				   "logits_end": p2_logits,
				   "probdist_end": p2_dist}
		return results

	

class BasicAttn(object):
	"""Module for basic attention.

	Note: in this module we use the terminology of "keys" and "values" (see lectures).
	In the terminology of "X attends to Y", "keys attend to values".

	In the baseline model, the keys are the context hidden states
	and the values are the question hidden states.

	We choose to use general terminology of keys and values in this module
	(rather than context and question) to avoid confusion if you reuse this
	module with other inputs.
	"""

	def __init__(self, keep_prob, key_vec_size, value_vec_size):
		"""
		Inputs:
		  keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
		  key_vec_size: size of the key vectors. int
		  value_vec_size: size of the value vectors. int
		"""
		self.keep_prob = keep_prob
		self.key_vec_size = key_vec_size
		self.value_vec_size = value_vec_size


	def build_graph(self, values, values_mask, keys):
		"""
		Keys attend to values.
		For each key, return an attention distribution and an attention output vector.

		Inputs:
		  values: Tensor shape (batch_size, num_values, value_vec_size).
		  values_mask: Tensor shape (batch_size, num_values).
			1s where there's real input, 0s where there's padding
		  keys: Tensor shape (batch_size, num_keys, value_vec_size)

		Outputs:
		  attn_dist: Tensor shape (batch_size, num_keys, num_values).
			For each key, the distribution should sum to 1,
			and should be 0 in the value locations that correspond to padding.
		  output: Tensor shape (batch_size, num_keys, hidden_size).
			This is the attention output; the weighted sum of the values
			(using the attention distribution as weights).
		"""
		with vs.variable_scope("BasicAttn"):

			# Useful notes:
			#   - matrix operations in tensorflow are adapted to also work with batches of matrices

			# Calculate attention distribution
			values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
			attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
			attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
			_, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

			# Use attention distribution to take weighted sum of values
			output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

			# Apply dropout
			output = tf.nn.dropout(output, self.keep_prob)

			return attn_dist, output


def masked_softmax(logits, mask, dim):
	"""
	Takes masked softmax over given dimension of logits.

	Inputs:
	logits: Numpy array. We want to take softmax over dimension dim.
	mask: Numpy array of same shape as logits.
	  Has 1s where there's real data in logits, 0 where there's padding
	dim: int. dimension over which to take softmax

	Returns:
	masked_logits: Numpy array same shape as logits.
	  This is the same as logits, but with 1e30 subtracted
	  (i.e. very large negative number) in the padding locations.
	prob_dist: Numpy array same shape as logits.
	  The result of taking softmax over masked_logits in given dimension.
	  Should be 0 in padding locations.
	  Should sum to 1 over given dimension.
	"""
	exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
	masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
	prob_dist = tf.nn.softmax(masked_logits, dim)
	return masked_logits, prob_dist