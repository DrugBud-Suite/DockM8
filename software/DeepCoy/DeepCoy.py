#!/usr/bin/env/python
"""
Usage:
    DeepCoy.py [options]

Options:
    -h --help                Show this screen
    --dataset NAME           Dataset name: zinc, zinc_phosphorus
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components
    --restrict_data INT      Limit data
"""
from typing import Sequence, Any
from docopt import docopt
from collections import defaultdict, deque
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json
import os
from ..DeepCoy.GGNN_DeepCoy import ChemModel
from ..DeepCoy import utils
from ..DeepCoy.utils import *
import pickle
import random
from numpy import linalg as LA
from rdkit import Chem
from copy import deepcopy
import os
import time
from ..DeepCoy.data_augmentation import *
'''
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edege types (3)
es:     maximum number of BFS transitions in this batch
v:      number of vertices per graph in this batch
h:      GNN hidden size
j:      noise/augmentation vector size
'''


class DenseGGNNChemModel(ChemModel):

    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
                        'task_sample_ratios': {},
                        'use_edge_bias': True,       # whether use edge bias in gnn

                        'clamp_gradient_norm': 1.0,
                        'out_layer_dropout_keep_prob': 1.0,

                        'tie_fwd_bkwd': True,
                        'random_seed': 0,

                        'batch_size': 8,
                        'num_epochs': 10,
                        'epoch_to_generate': 10,
                        'number_of_generation_per_valid': 10,
                        'maximum_distance': 50,
                        "use_argmax_generation": False,    # use random sampling or argmax during generation
                        'residual_connection_on': True,    # whether residual connection is on
                        'residual_connections': {          # For iteration i, specify list of layers whose output is added as an input
                                2: [0],
                                4: [0, 2],
                                6: [0, 2, 4],
                                8: [0, 2, 4, 6],
                                10: [0, 2, 4, 6, 8],
                                12: [0, 2, 4, 6, 8, 10],
                                14: [0, 2, 4, 6, 8, 10, 12],
                            },
                        'num_timesteps': 7,           # gnn propagation step
                        'hidden_size': 100,
                        'encoding_size': 8,
                        "kl_trade_off_lambda": 0.3,    # kl tradeoff
                        'learning_rate': 0.001,
                        'graph_state_dropout_keep_prob': 1,
                        "compensate_num": 0,           # how many atoms to be added during generation

                        'train_file': 'data/molecules_train_zinc.json',
                        'valid_file': 'data/molecules_valid_zinc.json',

                        'try_different_starting': True,
                        "num_different_starting": 1,

                        'generation': False,        # only for generation
                        'use_graph': True,          # use gnn
                        "label_one_hot": False,     # one hot label or not
                        "multi_bfs_path": False,    # whether sample several BFS paths for each molecule
                        "bfs_path_count": 30,
                        "path_random_order": False, # False: canonical order, True: random order
                        "sample_transition": False, # whether to use transition sampling
                        'edge_weight_dropout_keep_prob': 1,
                        'check_overlap_edge': False,
                        "truncate_distance": 10,

                        "output_name": '', # Filename of generated SMILES. If '' then name set based on run ID.
                        "subgraph_freq_file": '', # Location of pkl containing subgraph frequencies. Not necessary for molecule generation.
                        "use_subgraph_freqs": True, #  False: Vanilla cross entropy, True: Reweigh using subgraph frequencies
                        })

        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        out_dim = self.params['encoding_size']
        expanded_h_dim = self.params['hidden_size'] + self.params[
            'hidden_size'] + 1  # 1 for focus bit
        self.placeholders['graph_state_keep_prob'] = tf.compat.v1.placeholder(
            tf.float32, None, name='graph_state_keep_prob')
        self.placeholders[
            'edge_weight_dropout_keep_prob'] = tf.compat.v1.placeholder(
                tf.float32, None, name='edge_weight_dropout_keep_prob')
        # initial graph representation
        self.placeholders[
            'initial_node_representation_in'] = tf.compat.v1.placeholder(
                tf.float32, [None, None, self.params['hidden_size']],
                name='node_features_in')  # padded node symbols
        self.placeholders[
            'initial_node_representation_out'] = tf.compat.v1.placeholder(
                tf.float32, [None, None, self.params['hidden_size']],
                name='node_features_out')  # padded node symbols
        # mask out invalid node
        self.placeholders['node_mask_in'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='node_mask_in')  # [b x v]
        self.placeholders['node_mask_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='node_mask_out')  # [b x v]
        self.placeholders['num_vertices'] = tf.compat.v1.placeholder(
            tf.int32, ())
        # adj for encoder
        self.placeholders['adjacency_matrix_in'] = tf.compat.v1.placeholder(
            tf.float32, [None, self.num_edge_types, None, None],
            name="adjacency_matrix_in")  # [b, e, v, v]
        self.placeholders['adjacency_matrix_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, self.num_edge_types, None, None],
            name="adjacency_matrix_out")  # [b, e, v, v]

        # labels for node symbol prediction
        self.placeholders['node_symbols_in'] = tf.compat.v1.placeholder(
            tf.float32,
            [None, None, self.params['num_symbols']])  # [b, v, edge_type]
        self.placeholders['node_symbols_out'] = tf.compat.v1.placeholder(
            tf.float32,
            [None, None, self.params['num_symbols']])  # [b, v, edge_type]
        # node symbols used to enhance latent representations
        self.placeholders['latent_node_symbols_in'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.params['hidden_size']],
            name='latent_node_symbol_in')  # [b, v, h]
        self.placeholders['latent_node_symbols_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.params['hidden_size']],
            name='latent_node_symbol_out')  # [b, v, h]

        # mask out cross entropies in decoder
        self.placeholders['iteration_mask_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None])  # [b, es]
        # adj matrices used in decoder
        self.placeholders['incre_adj_mat_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.num_edge_types, None, None],
            name='incre_adj_mat_out')  # [b, es, e, v, v]
        # distance
        self.placeholders['distance_to_others_out'] = tf.compat.v1.placeholder(
            tf.int32, [None, None, None],
            name='distance_to_others_out')  # [b, es,v]
        # maximum iteration number of this batch
        self.placeholders['max_iteration_num'] = tf.compat.v1.placeholder(
            tf.int32, [], name='max_iteration_num')  # number
        # node number in focus at each iteration step
        self.placeholders['node_sequence_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, None],
            name='node_sequence_out')  # [b, es, v]
        # mask out invalid edge types at each iteration step
        self.placeholders['edge_type_masks_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.num_edge_types, None],
            name='edge_type_masks_out')  # [b, es, e, v]
        # ground truth edge type labels at each iteration step
        self.placeholders['edge_type_labels_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.num_edge_types, None],
            name='edge_type_labels_out')  # [b, es, e, v]
        # mask out invalid edge at each iteration step
        self.placeholders['edge_masks_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, None], name='edge_masks_out')  # [b, es, v]
        # ground truth edge labels at each iteration step
        self.placeholders['edge_labels_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, None],
            name='edge_labels_out')  # [b, es, v]
        # ground truth labels for whether it stops at each iteration step
        self.placeholders['local_stop_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None], name='local_stop_out')  # [b, es]
        # z_prior sampled from standard normal distribution
        self.placeholders['z_prior'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.params['encoding_size']],
            name='z_prior')  # prior z ~ normal distribution - output molecule
        self.placeholders['z_prior_in'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, self.params['hidden_size']],
            name='z_prior_in')  # prior z ~ normal distribution - input molecule
        # put in front of kl latent loss
        self.placeholders['kl_trade_off_lambda'] = tf.compat.v1.placeholder(
            tf.float32, [], name='kl_trade_off_lambda')  # number
        # overlapped edge features
        self.placeholders[
            'overlapped_edge_features_out'] = tf.compat.v1.placeholder(
                tf.int32, [None, None, None],
                name='overlapped_edge_features_out')  # [b, es, v]
        # weights for loss (based on compound freqs)
        self.placeholders['new_compound_freqs_out'] = tf.compat.v1.placeholder(
            tf.float32, [None, None, None],
            name='new_compound_freqs_out')  # [b, es, v]
        self.placeholders[
            'new_compound_freqs_edge_out'] = tf.compat.v1.placeholder(
                tf.float32, [None, None, None, None],
                name='new_compound_freqs_out')  # [b, es, e, v]

        # weights for encoder and decoder GNN.
        if self.params["residual_connection_on"]:
            # weights for encoder and decoder GNN. Different weights for each iteration
            for scope in ['_encoder', '_decoder']:
                if scope == '_encoder':
                    new_h_dim = h_dim
                else:
                    new_h_dim = expanded_h_dim
                for iter_idx in range(self.params['num_timesteps']):
                    with tf.compat.v1.variable_scope("gru_scope" + scope +
                                                     str(iter_idx),
                                                     reuse=False):
                        self.weights['edge_weights' + scope +
                                     str(iter_idx)] = tf.Variable(
                                         glorot_init([
                                             self.num_edge_types, new_h_dim,
                                             new_h_dim
                                         ]))
                        if self.params['use_edge_bias']:
                            self.weights['edge_biases' + scope +
                                         str(iter_idx)] = tf.Variable(
                                             np.zeros([
                                                 self.num_edge_types, 1,
                                                 new_h_dim
                                             ]).astype(np.float32))

                        cell = tf.compat.v1.nn.rnn_cell.GRUCell(new_h_dim)
                        cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                            cell,
                            state_keep_prob=self.
                            placeholders['graph_state_keep_prob'])
                        self.weights['node_gru' + scope + str(iter_idx)] = cell
        else:
            for scope in ['_encoder', '_decoder']:
                if scope == '_encoder':
                    new_h_dim = h_dim
                else:
                    new_h_dim = expanded_h_dim
                self.weights['edge_weights' + scope] = tf.Variable(
                    glorot_init([self.num_edge_types, new_h_dim, new_h_dim]))
                if self.params['use_edge_bias']:
                    self.weights['edge_biases' + scope] = tf.Variable(
                        np.zeros([self.num_edge_types, 1,
                                  new_h_dim]).astype(np.float32))
                with tf.compat.v1.variable_scope("gru_scope" + scope):
                    cell = tf.compat.v1.nn.rnn_cell.GRUCell(new_h_dim)
                    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        cell,
                        state_keep_prob=self.
                        placeholders['graph_state_keep_prob'])
                    self.weights['node_gru' + scope] = cell

        # The weights for calculating mean and variance
        self.weights['mean_weights'] = tf.Variable(glorot_init([h_dim, h_dim]))
        self.weights['mean_biases'] = tf.Variable(
            np.zeros([1, h_dim]).astype(np.float32))
        self.weights['variance_weights'] = tf.Variable(
            glorot_init([h_dim, h_dim]))
        self.weights['variance_biases'] = tf.Variable(
            np.zeros([1, h_dim]).astype(np.float32))

        self.weights['mean_weights_out'] = tf.Variable(
            glorot_init([h_dim, out_dim]))
        self.weights['mean_biases_out'] = tf.Variable(
            np.zeros([1, out_dim]).astype(np.float32))
        self.weights['variance_weights_out'] = tf.Variable(
            glorot_init([h_dim, out_dim]))
        self.weights['variance_biases_out'] = tf.Variable(
            np.zeros([1, out_dim]).astype(np.float32))

        # The weights for combining means and variances
        self.weights['mean_combine_weights_in'] = tf.Variable(
            glorot_init([out_dim, h_dim]))
        self.weights['variance_combine_weights_in'] = tf.Variable(
            glorot_init([out_dim, h_dim]))
        self.weights['atten_weights_c_in'] = tf.Variable(
            glorot_init([h_dim, h_dim]))
        self.weights['atten_weights_y_in'] = tf.Variable(
            glorot_init([h_dim, h_dim]))

        # The weights for generating nodel symbol logits
        self.weights['node_symbol_weights_in'] = tf.Variable(
            glorot_init([h_dim, self.params['num_symbols']]))
        self.weights['node_symbol_biases_in'] = tf.Variable(
            np.zeros([1, self.params['num_symbols']]).astype(np.float32))

        feature_dimension = 6 * expanded_h_dim
        # record the total number of features
        self.params["feature_dimension"] = 6
        # weights for generating edge type logits
        for direc in ["in", "out"]:
            for i in range(self.num_edge_types):
                self.weights['edge_type_%d_%s' % (i, direc)] = tf.Variable(
                    glorot_init([feature_dimension, feature_dimension]))
                self.weights['edge_type_biases_%d_%s' %
                             (i, direc)] = tf.Variable(
                                 np.zeros([1, feature_dimension
                                          ]).astype(np.float32))
                self.weights['edge_type_output_%d_%s' %
                             (i, direc)] = tf.Variable(
                                 glorot_init([feature_dimension, 1]))
            # weights for generating edge logits
            self.weights['edge_iteration_' + direc] = tf.Variable(
                glorot_init([feature_dimension, feature_dimension]))
            self.weights['edge_iteration_biases_' + direc] = tf.Variable(
                np.zeros([1, feature_dimension]).astype(np.float32))
            self.weights['edge_iteration_output_' + direc] = tf.Variable(
                glorot_init([feature_dimension, 1]))
            # Weights for the stop node
            self.weights["stop_node_" + direc] = tf.Variable(
                glorot_init([1, expanded_h_dim]))
            # Weight for distance embedding
            self.weights['distance_embedding_' + direc] = tf.Variable(
                glorot_init([self.params['maximum_distance'], expanded_h_dim]))
            # Weight for overlapped edge feature
            self.weights["overlapped_edge_weight_" + direc] = tf.Variable(
                glorot_init([2, expanded_h_dim]))

        # use node embeddings
        self.weights["node_embedding"] = tf.Variable(
            glorot_init([self.params["num_symbols"], h_dim]))

        # graph state mask
        self.ops['graph_state_mask_in'] = tf.expand_dims(
            self.placeholders['node_mask_in'], 2)
        self.ops['graph_state_mask_out'] = tf.expand_dims(
            self.placeholders['node_mask_out'], 2)

    # transform one hot vector to dense embedding vectors
    def get_node_embedding_state(self, one_hot_state, source=False):
        node_nums = tf.argmax(one_hot_state, axis=2)
        if source:
            return tf.nn.embedding_lookup(
                self.weights["node_embedding"],
                node_nums) * self.ops['graph_state_mask_in']
        else:
            return tf.nn.embedding_lookup(
                self.weights["node_embedding"],
                node_nums) * self.ops['graph_state_mask_out']

    def compute_final_node_representations_with_residual(
            self, h, adj, scope_name):  # scope_name: _encoder or _decoder
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        # _decoder uses a larger latent space because concat of symbol and latent representation
        if scope_name == "_decoder":
            h_dim = self.params['hidden_size'] + self.params['hidden_size'] + 1
        else:
            h_dim = self.params['hidden_size']
        h = tf.reshape(h, [-1, h_dim])  # [b*v, h]
        # record all hidden states at each iteration
        all_hidden_states = [h]
        for iter_idx in range(self.params['num_timesteps']):
            with tf.compat.v1.variable_scope("gru_scope" + scope_name +
                                             str(iter_idx),
                                             reuse=None) as g_scope:
                for edge_type in range(self.num_edge_types):
                    # the message passed from this vertice to other vertices
                    m = tf.matmul(
                        h, self.weights['edge_weights' + scope_name +
                                        str(iter_idx)][edge_type])  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases' + scope_name +
                                          str(iter_idx)][edge_type]  # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])  # [b, v, h]
                    # collect the messages from other vertices to each vertice
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                # all messages collected for each node
                acts = tf.reshape(acts, [-1, h_dim])  # [b*v, h]
                # add residual connection here
                layer_residual_connections = self.params[
                    'residual_connections'].get(iter_idx)
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [
                        all_hidden_states[residual_layer_idx]
                        for residual_layer_idx in layer_residual_connections
                    ]
                # concat current hidden states with residual states
                acts = tf.concat([acts] + layer_residual_states,
                                 axis=1)  # [b, (1+num residual connection)* h]

                # feed msg inputs and hidden states to GRU
                h = self.weights['node_gru' + scope_name + str(iter_idx)](
                    acts, h)[1]  # [b*v, h]
                # record the new hidden states
                all_hidden_states.append(h)
        last_h = tf.reshape(all_hidden_states[-1], [-1, v, h_dim])
        return last_h

    def compute_final_node_representations_without_residual(
            self, h, adj, edge_weights, edge_biases, node_gru, gru_scope_name):
        # h: initial representation, adj: adjacency matrix, different GNN parameters for encoder and decoder
        v = self.placeholders['num_vertices']
        if gru_scope_name == "gru_scope_decoder":
            h_dim = self.params['hidden_size'] + self.params['hidden_size']
        else:
            h_dim = self.params['hidden_size']
        h = tf.reshape(h, [-1, h_dim])

        with tf.compat.v1.variable_scope(gru_scope_name) as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(
                        h,
                        tf.nn.dropout(
                            edge_weights[edge_type],
                            rate=1 -
                            (self.placeholders['edge_weight_dropout_keep_prob'])
                        ))  # [b*v, h]
                    if self.params['use_edge_bias']:
                        m += edge_biases[edge_type]  # [b, v, h]
                    m = tf.reshape(m, [-1, v, h_dim])  # [b, v, h]
                    if edge_type == 0:
                        acts = tf.matmul(adj[edge_type], m)
                    else:
                        acts += tf.matmul(adj[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])  # [b*v, h]
                h = node_gru(acts, h)[1]  # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h

    def compute_mean_and_logvariance(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        out_dim = self.params['encoding_size']

        # AVERAGE ENCODING - output molecule
        avg_last_h_out = tf.reduce_sum(self.ops['final_node_representations_out'] * self.ops['graph_state_mask_out'], 1) / \
                         tf.reduce_sum(self.ops['graph_state_mask_out'], 1) # [b, 1, h] - mask out unused nodes
        mean_out = tf.matmul(
            avg_last_h_out,
            self.weights['mean_weights_out']) + self.weights['mean_biases_out']
        logvariance_out = tf.matmul(avg_last_h_out,
                                    self.weights['variance_weights_out']
                                   ) + self.weights['variance_biases_out']

        mean_out_ex = tf.reshape(
            tf.tile(tf.expand_dims(mean_out, 1), [1, v, 1]), [-1, out_dim])
        logvariance_out_ex = tf.reshape(
            tf.tile(tf.expand_dims(logvariance_out, 1), [1, v, 1]),
            [-1, out_dim])

        # PER VERTEX ENCODING - input molecule
        reshped_last_h = tf.reshape(self.ops['final_node_representations_in'],
                                    [-1, h_dim])
        mean = tf.matmul(
            reshped_last_h,
            self.weights['mean_weights']) + self.weights['mean_biases']

        logvariance = tf.matmul(
            reshped_last_h,
            self.weights['variance_weights']) + self.weights['variance_biases']

        return mean, logvariance, mean_out_ex, logvariance_out_ex

    def sample_with_mean_and_logvariance(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        out_dim = self.params['encoding_size']
        # Sample from normal distribution
        z_prior = tf.reshape(self.placeholders['z_prior'], [-1, out_dim])
        z_prior_in = tf.reshape(self.placeholders['z_prior_in'], [-1, h_dim])
        # Train: sample from u, Sigma. Generation: sample from 0,1
        z_sampled = tf.cond(
            self.placeholders['is_generative'],
            lambda: z_prior,  # standard normal 
            lambda: tf.add(
                self.ops['mean_out'],
                tf.multiply(tf.sqrt(tf.exp(self.ops['logvariance_out'])),
                            z_prior)))  # non-standard normal
        # Update node representations
        mean = tf.reshape(self.ops['mean'], [-1, v, h_dim])
        mean = mean * self.ops['graph_state_mask_in']
        roll_shift = tf.cast(
            tf.reduce_min(tf.reduce_sum(self.ops['graph_state_mask_in'], 1)),
            tf.int32)  # Shift by smallest number of atoms in smallest mol
        inverted_mask = tf.ones(tf.shape(
            self.ops['graph_state_mask_in'])) - self.ops['graph_state_mask_in']
        update_vals = self.placeholders[
            'z_prior_in'] * inverted_mask  # Extra nodes from N(0,1)
        mean = tf.reshape(
            tf.add(mean, update_vals),
            [-1, h_dim])  # Fill in extra vertices with random noise from N(0,1)

        # Combine input molecule node embeddings with output molecule embedding
        # Attention mechanism over in_mol encodings to determine combination with z_sampled
        atten_masks_c = tf.tile(
            tf.expand_dims(self.ops['graph_state_mask_out'], 2),
            [1, 1, v, 1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_masks_yi = tf.tile(
            tf.expand_dims(self.ops['graph_state_mask_out'], 1),
            [1, v, 1, 1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_masks = atten_masks_c + atten_masks_yi

        atten_c = tf.tile(tf.expand_dims(tf.reshape(mean, [-1, v, h_dim]), 2),
                          [1, 1, v, 1])  # [b, v, v, h]
        atten_yi = tf.tile(tf.expand_dims(tf.reshape(mean, [-1, v, h_dim]), 1),
                           [1, v, 1, 1])  # [b, v, v, h]
        atten_c = tf.reshape(
            tf.matmul(tf.reshape(atten_c, [-1, h_dim]),
                      self.weights['atten_weights_c_in']), [-1, v, v, h_dim])
        atten_yi = tf.reshape(
            tf.matmul(tf.reshape(atten_yi, [-1, h_dim]),
                      self.weights['atten_weights_y_in']), [-1, v, v, h_dim])
        atten_mi = tf.nn.sigmoid(tf.add(atten_c, atten_yi) + atten_masks)
        atten_mi = tf.reduce_sum(atten_mi, 2) / tf.tile(
            tf.expand_dims(tf.reduce_sum(self.ops['graph_state_mask_out'], 1),
                           1), [1, v, 1])

        z_sampled = tf.reshape(
            tf.matmul(z_sampled, self.weights['mean_combine_weights_in']),
            [-1, v, h_dim])
        mean_sampled = tf.reshape(
            mean, [-1, v, h_dim
                  ]) * self.ops['graph_state_mask_out'] + atten_mi * z_sampled

        return mean_sampled

    def fully_connected(self, input, hidden_weight, hidden_bias, output_weight):
        output = tf.nn.relu(tf.matmul(input, hidden_weight) + hidden_bias)
        output = tf.matmul(output, output_weight)
        return output

    def generate_cross_entropy(self, idx, cross_entropy_losses,
                               edge_predictions, edge_type_predictions):
        direc = "out"
        direc_r = "in"

        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        num_symbols = self.params['num_symbols']
        batch_size = tf.shape(self.placeholders['initial_node_representation_' +
                                                direc])[0]
        # Use latent representation as decoder GNN input
        filtered_z_sampled = self.ops["initial_repre_for_decoder_" +
                                      direc_r]  # [b, v, h+h]
        # data needed in this iteration
        incre_adj_mat = self.placeholders['incre_adj_mat_' +
                                          direc][:,
                                                 idx, :, :, :]  # [b, e, v, v]
        distance_to_others = self.placeholders['distance_to_others_' +
                                               direc][:, idx, :]  # [b,v]
        overlapped_edge_features = self.placeholders['overlapped_edge_features_'
                                                     + direc][:,
                                                              idx, :]  # [b,v]
        subgraph_freqs = self.placeholders['new_compound_freqs_' +
                                           direc][:, idx, :]  # [b, v]
        subgraph_edge_freqs = self.placeholders['new_compound_freqs_edge_' +
                                                direc][:,
                                                       idx, :, :]  # [b, e, v]
        node_sequence = self.placeholders['node_sequence_' +
                                          direc][:, idx, :]  # [b, v]
        node_sequence = tf.expand_dims(node_sequence, axis=2)  # [b, v, 1]
        edge_type_masks = self.placeholders['edge_type_masks_' +
                                            direc][:, idx, :, :]  # [b, e, v]
        # make invalid locations to be very small before using softmax function
        edge_type_masks = edge_type_masks * LARGE_NUMBER - LARGE_NUMBER
        edge_type_labels = self.placeholders['edge_type_labels_' +
                                             direc][:, idx, :, :]  # [b, e, v]
        edge_masks = self.placeholders['edge_masks_' + direc][:,
                                                              idx, :]  # [b, v]
        # make invalid locations to be very small before using softmax function
        edge_masks = edge_masks * LARGE_NUMBER - LARGE_NUMBER
        edge_labels = self.placeholders['edge_labels_' +
                                        direc][:, idx, :]  # [b, v]
        local_stop = self.placeholders['local_stop_' + direc][:, idx]  # [b]
        # concat the hidden states with the node in focus
        filtered_z_sampled = tf.concat([filtered_z_sampled, node_sequence],
                                       axis=2)  # [b, v, h + h + 1]
        # Decoder GNN
        if self.params["use_graph"]:
            if self.params["residual_connection_on"]:
                new_filtered_z_sampled = self.compute_final_node_representations_with_residual(
                    filtered_z_sampled, tf.transpose(incre_adj_mat,
                                                     [1, 0, 2, 3]),
                    "_decoder")  # [b, v, h + h]
            else:
                new_filtered_z_sampled = self.compute_final_node_representations_without_residual(
                    filtered_z_sampled, tf.transpose(incre_adj_mat,
                                                     [1, 0, 2, 3]),
                    self.weights['edge_weights_decoder'],
                    self.weights['edge_biases_decoder'],
                    self.weights['node_gru_decoder'],
                    "gru_scope_decoder")  # [b, v, h + h]
        else:
            new_filtered_z_sampled = filtered_z_sampled
        # Filter nonexist nodes
        new_filtered_z_sampled = new_filtered_z_sampled * self.ops[
            'graph_state_mask_' + direc]
        # Take out the node in focus
        node_in_focus = tf.reduce_sum(node_sequence * new_filtered_z_sampled,
                                      axis=1)  # [b, h + h]
        # edge pair representation
        edge_repr=tf.concat(\
            [tf.tile(tf.expand_dims(node_in_focus, 1), [1,v,1]), new_filtered_z_sampled], axis=2) # [b, v, 2*(h+h)]
        # combine edge repre with local and global repr
        local_graph_repr_before_expansion = tf.reduce_sum(new_filtered_z_sampled, axis=1) /  \
                                            tf.reduce_sum(self.placeholders['node_mask_'+direc], axis=1, keepdims=True) # [b, h + h]
        local_graph_repr = tf.expand_dims(local_graph_repr_before_expansion, 1)
        local_graph_repr = tf.tile(local_graph_repr, [1, v, 1])  # [b, v, h+h]
        global_graph_repr_before_expansion = tf.reduce_sum(filtered_z_sampled, axis=1) / \
                                            tf.reduce_sum(self.placeholders['node_mask_'+direc], axis=1, keepdims=True)
        global_graph_repr = tf.expand_dims(global_graph_repr_before_expansion,
                                           1)
        global_graph_repr = tf.tile(global_graph_repr, [1, v, 1])  # [b, v, h+h]
        # distance representation
        distance_repr = tf.nn.embedding_lookup(
            self.weights['distance_embedding_' + direc_r],
            distance_to_others)  # [b, v, h+h]
        # overlapped edge feature representation
        overlapped_edge_repr = tf.nn.embedding_lookup(
            self.weights['overlapped_edge_weight_' + direc_r],
            overlapped_edge_features)  # [b, v, h+h]
        # concat and reshape.
        combined_edge_repr = tf.concat([
            edge_repr, local_graph_repr, global_graph_repr, distance_repr,
            overlapped_edge_repr
        ],
                                       axis=2)

        combined_edge_repr = tf.reshape(
            combined_edge_repr,
            [-1, self.params["feature_dimension"] * (h_dim + h_dim + 1)])
        # Calculate edge logits
        edge_logits = self.fully_connected(
            combined_edge_repr, self.weights['edge_iteration_' + direc_r],
            self.weights['edge_iteration_biases_' + direc_r],
            self.weights['edge_iteration_output_' + direc_r])
        edge_logits = tf.reshape(edge_logits, [-1, v])  # [b, v]
        # filter invalid terms
        edge_logits = edge_logits + edge_masks
        # Calculate whether it will stop at this step
        # prepare the data
        expanded_stop_node = tf.tile(self.weights['stop_node_' + direc_r],
                                     [batch_size, 1])  # [b, h + h]
        distance_to_stop_node = tf.nn.embedding_lookup(
            self.weights['distance_embedding_' + direc_r],
            tf.tile([0], [batch_size]))  # [b, h + h]
        overlap_edge_stop_node = tf.nn.embedding_lookup(
            self.weights['overlapped_edge_weight_' + direc_r],
            tf.tile([0], [batch_size]))  # [b, h + h]

        combined_stop_node_repr = tf.concat([
            node_in_focus, expanded_stop_node,
            local_graph_repr_before_expansion,
            global_graph_repr_before_expansion, distance_to_stop_node,
            overlap_edge_stop_node
        ],
                                            axis=1)  # [b, 6 * (h + h)]
        # logits for stop node
        stop_logits = self.fully_connected(
            combined_stop_node_repr, self.weights['edge_iteration_' + direc_r],
            self.weights['edge_iteration_biases_' + direc_r],
            self.weights['edge_iteration_output_' + direc_r])  #[b, 1]
        edge_logits = tf.concat([edge_logits, stop_logits],
                                axis=1)  # [b, v + 1]

        # Calculate edge type logits
        edge_type_logits = []
        for i in range(self.num_edge_types):
            edge_type_logit = self.fully_connected(
                combined_edge_repr,
                self.weights['edge_type_%d_%s' % (i, direc_r)],
                self.weights['edge_type_biases_%d_%s' % (i, direc_r)],
                self.weights['edge_type_output_%d_%s' %
                             (i, direc_r)])  #[b * v, 1]
            edge_type_logits.append(tf.reshape(edge_type_logit,
                                               [-1, 1, v]))  # [b, 1, v]

        edge_type_logits = tf.concat(edge_type_logits, axis=1)  # [b, e, v]
        # filter invalid items
        edge_type_logits = edge_type_logits + edge_type_masks  # [b, e, v]
        # softmax over edge type axis
        edge_type_probs = tf.nn.softmax(edge_type_logits, 1)  # [b, e, v]

        # edge labels
        edge_labels = tf.concat(
            [edge_labels, tf.expand_dims(local_stop, 1)], axis=1)  # [b, v + 1]
        # Reweigh edge probabilities by subgraph frequencies
        if self.params['use_subgraph_freqs']:
            subgraph_freqs_mask = self.placeholders['edge_masks_' +
                                                    direc][:, idx, :]
            subgraph_freqs = tf.concat([subgraph_freqs,
                                        tf.reduce_sum(subgraph_freqs*subgraph_freqs_mask+SMALL_NUMBER, axis=1, keepdims=True) \
                                         / tf.reduce_sum(subgraph_freqs_mask+SMALL_NUMBER, axis=1, keepdims=True)], axis=1) # [b, v+1] # add stop node multiplier
            edge_softmax = tf.nn.softmax(
                edge_logits
            ) * subgraph_freqs  # reweight probabilities by subgraph occurrence
            edge_softmax = edge_softmax / tf.reduce_sum(
                edge_softmax, axis=1, keepdims=True)  # Normalise

            # Reweight edge loss by subgraph frequencies
            subgraph_freqs_mask = tf.concat(
                [subgraph_freqs_mask,
                 np.ones((self.params['batch_size'], 1))],
                axis=1)  # [b, v+1]
            edge_loss_multi = tf.tile(tf.reduce_sum(subgraph_freqs*subgraph_freqs_mask+SMALL_NUMBER, axis=1, keepdims=True) \
                                      /(tf.reduce_sum(subgraph_freqs_mask+SMALL_NUMBER, axis=1, keepdims=True)), [1, v+1]) # [b, v+1]

            # Edge loss
            edge_loss = -tf.reduce_sum(tf.math.log(edge_softmax + SMALL_NUMBER)
                                       * edge_labels * edge_loss_multi,
                                       axis=1)

            # Reweigh edge type probabilityies by subgraph frequencies
            edge_type_probs = edge_type_probs * subgraph_edge_freqs
            edge_type_probs = edge_type_probs / tf.reduce_sum(
                edge_type_probs, axis=1, keepdims=True)  # Normalise

            # Reweigh edge type loss by subgraph frequencies
            subgraph_edge_freqs_mask = self.placeholders['edge_type_masks_' +
                                                         direc][:, idx, :, :]
            edge_type_loss_multi = tf.tile(tf.reduce_mean(subgraph_edge_freqs*subgraph_edge_freqs_mask+SMALL_NUMBER, axis=1, keepdims=True) \
                                           /(tf.reduce_sum(subgraph_edge_freqs_mask+SMALL_NUMBER, axis=1, keepdims=True)+1.), [1, self.num_edge_types, 1]) # [b, e, v]

            # softmax for edge type
            edge_type_loss = -edge_type_labels * tf.math.log(
                edge_type_probs +
                SMALL_NUMBER) * edge_type_loss_multi  # [b, e, v]
            edge_type_loss = tf.reduce_sum(edge_type_loss, axis=[1, 2])  # [b]
        # Don't reweigh - vanilla cross entropy
        else:
            # edge
            edge_softmax = tf.nn.softmax(edge_logits)
            edge_loss = -tf.reduce_sum(
                tf.compat.v1.log(edge_softmax + SMALL_NUMBER) * edge_labels,
                axis=1)

            # edge type
            edge_type_loss = -edge_type_labels * tf.compat.v1.log(
                edge_type_probs + SMALL_NUMBER)  # [b, e, v]
            edge_type_loss = tf.reduce_sum(edge_type_loss, axis=[1, 2])  # [b]

        # total loss
        iteration_loss = edge_loss + edge_type_loss
        cross_entropy_losses = cross_entropy_losses.write(idx, iteration_loss)
        edge_predictions = edge_predictions.write(idx,
                                                  tf.nn.softmax(edge_logits))
        edge_type_predictions = edge_type_predictions.write(
            idx, edge_type_probs)
        return (idx + 1, cross_entropy_losses, edge_predictions,
                edge_type_predictions)

    def construct_logit_matrices(self):
        v = self.placeholders['num_vertices']
        batch_size = tf.shape(
            self.placeholders['initial_node_representation_out'])[0]
        h_dim = self.params['hidden_size']

        in_direc = "in"
        out_direc = "out"

        # Initial state: embedding
        latent_node_state = self.get_node_embedding_state(
            self.placeholders["latent_node_symbols_" + out_direc])
        # Concat z_sampled with node symbols
        filtered_z_sampled = tf.concat(
            [self.ops['z_sampled_' + in_direc], latent_node_state],
            axis=2)  # [b, v, h + h]
        self.ops["initial_repre_for_decoder_" + in_direc] = filtered_z_sampled
        # The tensor array used to collect the cross entropy losses at each step
        cross_entropy_losses = tf.TensorArray(
            dtype=tf.float32, size=self.placeholders['max_iteration_num'])
        edge_predictions = tf.TensorArray(
            dtype=tf.float32, size=self.placeholders['max_iteration_num'])
        edge_type_predictions = tf.TensorArray(
            dtype=tf.float32, size=self.placeholders['max_iteration_num'])
        idx_final, cross_entropy_losses_final, edge_predictions_final,edge_type_predictions_final=\
                tf.while_loop(cond=lambda idx, cross_entropy_losses,edge_predictions,edge_type_predictions: idx < self.placeholders['max_iteration_num'],
                body=self.generate_cross_entropy,
                loop_vars=(tf.constant(0), cross_entropy_losses,edge_predictions,edge_type_predictions,))
        # Record the predictions for generation
        self.ops['edge_predictions_' +
                 in_direc] = edge_predictions_final.read(0)
        self.ops['edge_type_predictions_' +
                 in_direc] = edge_type_predictions_final.read(0)

        # Final cross entropy losses
        cross_entropy_losses_final = cross_entropy_losses_final.stack()
        self.ops['cross_entropy_losses_' + in_direc] = tf.transpose(
            cross_entropy_losses_final, [1, 0])  # [b, es]

        # Logits for node symbols
        self.ops['node_symbol_logits_' + in_direc] = tf.reshape(
            tf.matmul(
                tf.reshape(self.ops['z_sampled_' + in_direc], [-1, h_dim]),
                self.weights['node_symbol_weights_' + in_direc]) +
            self.weights['node_symbol_biases_' + in_direc],
            [-1, v, self.params['num_symbols']])

    def construct_loss(self):
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        out_dim = self.params['encoding_size']
        kl_trade_off_lambda = self.placeholders['kl_trade_off_lambda']

        in_direc = "in"
        out_direc = "out"

        # Edge loss
        self.ops["edge_loss_" + in_direc] = tf.reduce_sum(
            self.ops['cross_entropy_losses_' + in_direc] *
            self.placeholders['iteration_mask_' + out_direc],
            axis=1)

        # KL loss
        # Node embeddings of input molecule
        kl_loss_in = 1 + self.ops['logvariance'] - tf.square(
            self.ops['mean']) - tf.exp(self.ops['logvariance'])
        kl_loss_in = tf.reshape(kl_loss_in, [-1, v, h_dim]) * self.ops[
            'graph_state_mask_' + in_direc]  # Only penalise for nodes in graph
        # Node embeddings of output molecule
        kl_loss_noise = 1 + self.ops['logvariance_out'] - tf.square(
            self.ops['mean_out']) - tf.exp(self.ops['logvariance_out'])
        kl_loss_noise = tf.reshape(kl_loss_noise, [-1, v, out_dim]) * self.ops[
            'graph_state_mask_' + out_direc]  # Only penalise for nodes in graph
        self.ops['kl_loss_' + in_direc] = -0.5 * tf.reduce_sum(
            kl_loss_in, [1, 2]) - 0.5 * tf.reduce_sum(kl_loss_noise, [1, 2])

        # Node symbol loss
        self.ops['node_symbol_prob_' + in_direc] = tf.nn.softmax(
            self.ops['node_symbol_logits_' + in_direc])
        self.ops['node_symbol_loss_' + in_direc] = -tf.reduce_sum(
            tf.math.log(self.ops['node_symbol_prob_' + in_direc] + SMALL_NUMBER)
            * self.placeholders['node_symbols_' + out_direc],
            axis=[1, 2])

        # Overall losses
        self.ops['mean_edge_loss_' + in_direc] = tf.reduce_mean(
            self.ops["edge_loss_" + in_direc])
        self.ops['mean_node_symbol_loss_' + in_direc] = tf.reduce_mean(
            self.ops["node_symbol_loss_" + in_direc])
        self.ops['mean_kl_loss_' + in_direc] = tf.reduce_mean(
            kl_trade_off_lambda * self.ops['kl_loss_' + in_direc])

        return tf.reduce_mean(self.ops["edge_loss_in"] + self.ops['node_symbol_loss_in'] +\
                              kl_trade_off_lambda *self.ops['kl_loss_in'])

    def gated_regression(self, last_h, regression_gate, regression_transform,
                         hidden_size, projection_weight, projection_bias, v,
                         mask):
        # last_h: [b x v x h]
        last_h = tf.reshape(last_h, [-1, hidden_size])  # [b*v, h]
        # linear projection on last_h
        last_h = tf.nn.relu(
            tf.matmul(last_h, projection_weight) + projection_bias)  # [b*v, h]
        # same as last_h
        gate_input = last_h
        # linear projection and combine
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * tf.nn.tanh(
            regression_transform(last_h))  # [b*v, 1]
        gated_outputs = tf.reshape(gated_outputs, [-1, v])  # [b, v]
        masked_gated_outputs = gated_outputs * mask  # [b x v]
        output = tf.reduce_sum(masked_gated_outputs, axis=1)  # [b]
        output = tf.sigmoid(output)
        return output

    def calculate_incremental_results(self, raw_data, bucket_sizes, file_name,
                                      is_training_data):
        incremental_results = [[], []]
        # copy the raw_data if more than 1 BFS path is added
        new_raw_data = []
        for idx, d in enumerate(raw_data):
            out_direc = "out"
            res_idx = 1

            # Use canonical order or random order here. canonical order starts from index 0. random order starts from random nodes
            if not self.params["path_random_order"]:
                # Use several different starting index if using multi BFS path
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx = list(
                        range(self.params["bfs_path_count"]))
                else:
                    list_of_starting_idx = [0]  # the index 0
            else:
                # get the node length for this output molecule
                node_length = len(d["node_features_" + out_direc])
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx = np.random.choice(
                        node_length,
                        self.params["bfs_path_count"],
                        replace=True)  # randomly choose several
                else:
                    list_of_starting_idx = [
                        random.choice(list(range(node_length)))
                    ]  # randomly choose one
            for list_idx, starting_idx in enumerate(list_of_starting_idx):
                # Choose a bucket
                chosen_bucket_idx = np.argmax(bucket_sizes > max(
                    max([v for e in d['graph_out'] for v in [e[0], e[2]]]),
                    max([v for e in d['graph_in'] for v in [e[0], e[2]]])))
                chosen_bucket_size = bucket_sizes[chosen_bucket_idx]

                nodes_no_master = d['node_features_' + out_direc]
                edges_no_master = d['graph_' + out_direc]
                incremental_adj_mat,distance_to_others,node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features, new_compound_frequencies, new_compound_frequencies_edge=\
                        construct_incremental_graph_freqs(self.params["dataset"], edges_no_master, chosen_bucket_size,
                                            len(nodes_no_master), nodes_no_master, self.params, is_training_data, self.freq_dict, initial_idx=starting_idx)
                if self.params["sample_transition"] and list_idx > 0:
                    incremental_results[res_idx][-1] = [
                        x + y
                        for x, y in zip(incremental_results[res_idx][-1], [
                            incremental_adj_mat, distance_to_others,
                            node_sequence, edge_type_masks, edge_type_labels,
                            local_stop, edge_masks, edge_labels,
                            overlapped_edge_features, new_compound_frequencies,
                            new_compound_frequencies_edge
                        ])
                    ]
                else:
                    incremental_results[res_idx].append([
                        incremental_adj_mat, distance_to_others, node_sequence,
                        edge_type_masks, edge_type_labels, local_stop,
                        edge_masks, edge_labels, overlapped_edge_features,
                        new_compound_frequencies, new_compound_frequencies_edge
                    ])
                    new_raw_data.append(d)
            # Progress
            if idx % 50 == 0:
                print('Finished calculating %d incremental matrices' % idx,
                      end="\r")
        return incremental_results, new_raw_data

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self,
                           raw_data,
                           is_training_data,
                           file_name,
                           bucket_sizes=None):
        if bucket_sizes is None:
            bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
        incremental_results, raw_data = self.calculate_incremental_results(
            raw_data, bucket_sizes, file_name, is_training_data)
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features_out"][0])

        for d, incremental_result_1 in zip(raw_data, incremental_results[1]):
            # choose a bucket
            chosen_bucket_idx = np.argmax(bucket_sizes > max(
                max([v for e in d['graph_in'] for v in [e[0], e[2]]]),
                max([v for e in d['graph_out'] for v in [e[0], e[2]]])))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            # total number of nodes in this data point out
            n_active_nodes_in = len(d["node_features_in"])
            n_active_nodes_out = len(d["node_features_out"])
            bucketed[chosen_bucket_idx].append({
                'adj_mat_in':
                    graph_to_adj_mat(d['graph_in'], chosen_bucket_size,
                                     self.num_edge_types,
                                     self.params['tie_fwd_bkwd']),
                'adj_mat_out':
                    graph_to_adj_mat(d['graph_out'], chosen_bucket_size,
                                     self.num_edge_types,
                                     self.params['tie_fwd_bkwd']),
                'incre_adj_mat_out':
                    incremental_result_1[0],
                'distance_to_others_out':
                    incremental_result_1[1],
                'overlapped_edge_features_out':
                    incremental_result_1[8],
                'new_compound_freqs_out':
                    incremental_result_1[9],
                'new_compound_freqs_edge_out':
                    incremental_result_1[10],
                'node_sequence_out':
                    incremental_result_1[2],
                'edge_type_masks_out':
                    incremental_result_1[3],
                'edge_type_labels_out':
                    incremental_result_1[4],
                'edge_masks_out':
                    incremental_result_1[6],
                'edge_labels_out':
                    incremental_result_1[7],
                'local_stop_out':
                    incremental_result_1[5],
                'number_iteration_out':
                    len(incremental_result_1[5]),
                'init_in':
                    d["node_features_in"] +
                    [[0
                      for _ in range(x_dim)]
                     for __ in range(chosen_bucket_size - n_active_nodes_in)],
                'init_out':
                    d["node_features_out"] +
                    [[0
                      for _ in range(x_dim)]
                     for __ in range(chosen_bucket_size - n_active_nodes_out)],
                'mask_in':
                    [1. for _ in range(n_active_nodes_in)] +
                    [0. for _ in range(chosen_bucket_size - n_active_nodes_in)],
                'mask_out': [1. for _ in range(n_active_nodes_out)] + [
                    0. for _ in range(chosen_bucket_size - n_active_nodes_out)
                ],
                'smiles_in':
                    d['smiles_in'],
                'smiles_out':
                    d['smiles_out'],
            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)

        bucket_at_step = [[
            bucket_idx
            for _ in range(len(bucket_data) // self.params['batch_size'])
        ]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return (bucketed, bucket_sizes, bucket_at_step)

    def pad_annotations(self, annotations):
        return np.pad(annotations,
                      pad_width=[[0, 0], [0, 0],
                                 [
                                     0, self.params['hidden_size'] -
                                     self.params["num_symbols"]
                                 ]],
                      mode='constant')

    def make_batch(self, elements, maximum_vertice_num):
        # get maximum number of iterations in this batch. used to control while_loop
        max_iteration_num = -1
        for d in elements:
            max_iteration_num = max(d['number_iteration_out'],
                                    max_iteration_num)
        batch_data = {
            'adj_mat_in': [],
            'adj_mat_out': [],
            'init_in': [],
            'init_out': [],
            'edge_type_masks_out': [],
            'edge_type_labels_out': [],
            'edge_masks_out': [],
            'edge_labels_out': [],
            'node_mask_in': [],
            'node_mask_out': [],
            'node_sequence_out': [],
            'iteration_mask_out': [],
            'local_stop_out': [],
            'incre_adj_mat_out': [],
            'distance_to_others_out': [],
            'max_iteration_num': max_iteration_num,
            'overlapped_edge_features_out': [],
            'new_compound_freqs_out': [],
            'new_compound_freqs_edge_out': []
        }
        for d in elements:
            batch_data['adj_mat_in'].append(d['adj_mat_in'])
            batch_data['adj_mat_out'].append(d['adj_mat_out'])
            batch_data['init_in'].append(d['init_in'])
            batch_data['init_out'].append(d['init_out'])
            batch_data['node_mask_in'].append(d['mask_in'])
            batch_data['node_mask_out'].append(d['mask_out'])

            for direc in ['_out']:
                # sparse to dense for saving memory
                incre_adj_mat = incre_adj_mat_to_dense(
                    d['incre_adj_mat' + direc], self.num_edge_types,
                    maximum_vertice_num)
                distance_to_others = distance_to_others_dense(
                    d['distance_to_others' + direc], maximum_vertice_num)
                overlapped_edge_features = overlapped_edge_features_to_dense(
                    d['overlapped_edge_features' + direc], maximum_vertice_num)
                new_compound_freqs = transition_freqs_to_dense(
                    d['new_compound_freqs' + direc], maximum_vertice_num)
                new_compound_freqs_edge = transition_freqs_edge_to_dense(
                    d['new_compound_freqs_edge' + direc],
                    maximum_vertice_num,
                    num_edge_types=self.num_edge_types)
                node_sequence = node_sequence_to_dense(
                    d['node_sequence' + direc], maximum_vertice_num)
                edge_type_masks = edge_type_masks_to_dense(
                    d['edge_type_masks' + direc], maximum_vertice_num,
                    self.num_edge_types)
                edge_type_labels = edge_type_labels_to_dense(
                    d['edge_type_labels' + direc], maximum_vertice_num,
                    self.num_edge_types)
                edge_masks = edge_masks_to_dense(d['edge_masks' + direc],
                                                 maximum_vertice_num)
                edge_labels = edge_labels_to_dense(d['edge_labels' + direc],
                                                   maximum_vertice_num)

                batch_data['incre_adj_mat' + direc].append(incre_adj_mat + [
                    np.zeros((self.num_edge_types, maximum_vertice_num,
                              maximum_vertice_num))
                    for _ in range(max_iteration_num -
                                   d['number_iteration' + direc])
                ])
                batch_data['distance_to_others' +
                           direc].append(distance_to_others + [
                               np.zeros((maximum_vertice_num))
                               for _ in range(max_iteration_num -
                                              d['number_iteration' + direc])
                           ])
                batch_data['overlapped_edge_features' +
                           direc].append(overlapped_edge_features + [
                               np.zeros((maximum_vertice_num))
                               for _ in range(max_iteration_num -
                                              d['number_iteration' + direc])
                           ])
                batch_data['new_compound_freqs' +
                           direc].append(new_compound_freqs + [
                               np.ones((maximum_vertice_num))
                               for _ in range(max_iteration_num -
                                              d['number_iteration' + direc])
                           ])
                batch_data['new_compound_freqs_edge' +
                           direc].append(new_compound_freqs_edge + [
                               np.ones((self.num_edge_types,
                                        maximum_vertice_num))
                               for _ in range(max_iteration_num -
                                              d['number_iteration' + direc])
                           ])
                batch_data['node_sequence' + direc].append(node_sequence + [
                    np.zeros((maximum_vertice_num))
                    for _ in range(max_iteration_num -
                                   d['number_iteration' + direc])
                ])
                batch_data['edge_type_masks' + direc].append(edge_type_masks + [
                    np.zeros((self.num_edge_types, maximum_vertice_num))
                    for _ in range(max_iteration_num -
                                   d['number_iteration' + direc])
                ])
                batch_data['edge_masks' + direc].append(edge_masks + [
                    np.zeros((maximum_vertice_num))
                    for _ in range(max_iteration_num -
                                   d['number_iteration' + direc])
                ])
                batch_data['edge_type_labels' +
                           direc].append(edge_type_labels + [
                               np.zeros((self.num_edge_types,
                                         maximum_vertice_num))
                               for _ in range(max_iteration_num -
                                              d['number_iteration' + direc])
                           ])
                batch_data['edge_labels' + direc].append(edge_labels + [
                    np.zeros((maximum_vertice_num))
                    for _ in range(max_iteration_num -
                                   d['number_iteration' + direc])
                ])
                batch_data['iteration_mask' + direc].append(
                    [1 for _ in range(d['number_iteration' + direc])] + [
                        0 for _ in range(max_iteration_num -
                                         d['number_iteration' + direc])
                    ])
                batch_data['local_stop' + direc].append(
                    [int(s) for s in d['local_stop' + direc]] + [
                        0 for _ in range(max_iteration_num -
                                         d['number_iteration' + direc])
                    ])

        return batch_data

    def get_dynamic_feed_dict(self, elements, latent_node_symbol, incre_adj_mat,
                              num_vertices, distance_to_others,
                              overlapped_edge_dense, new_compound_freqs_dense,
                              new_compound_freqs_edge_dense, node_sequence,
                              edge_type_masks, edge_masks, random_normal_states,
                              random_normal_states_in):
        if incre_adj_mat is None:
            incre_adj_mat = np.zeros((1, 1, self.num_edge_types, 1, 1))
            distance_to_others = np.zeros((1, 1, 1))
            overlapped_edge_dense = np.zeros((1, 1, 1))
            new_compound_freqs_dense = np.ones((1, 1, 1))
            new_compound_freqs_edge_dense = np.ones((1, 1, 1, 1))
            node_sequence = np.zeros((1, 1, 1))
            edge_type_masks = np.zeros((1, 1, self.num_edge_types, 1))
            edge_masks = np.zeros((1, 1, 1))
            latent_node_symbol = np.zeros((1, 1, self.params["num_symbols"]))
        return {
            self.placeholders['z_prior']:
                random_normal_states,  # [1, v, j]
            self.placeholders['z_prior_in']:
                random_normal_states_in,  # [1, v, h]
            self.placeholders['incre_adj_mat_out']:
                incre_adj_mat,  # [1, 1, e, v, v]
            self.placeholders['num_vertices']:
                num_vertices,  # v
            self.placeholders['initial_node_representation_in']:
                self.pad_annotations([elements['init_in']]),
            self.placeholders['initial_node_representation_out']:
                self.pad_annotations([elements['init_out']]),
            self.placeholders['node_symbols_out']: [elements['init_out']],
            self.placeholders['node_symbols_in']: [elements['init_in']],
            self.placeholders['latent_node_symbols_in']:
                self.pad_annotations(latent_node_symbol),
            self.placeholders['latent_node_symbols_out']:
                self.pad_annotations(latent_node_symbol),
            self.placeholders['adjacency_matrix_in']: [elements['adj_mat_in']],
            self.placeholders['adjacency_matrix_out']: [
                elements['adj_mat_out']
            ],
            self.placeholders['node_mask_in']: [elements['mask_in']],
            self.placeholders['node_mask_out']: [elements['mask_out']],
            self.placeholders['graph_state_keep_prob']:
                1,
            self.placeholders['edge_weight_dropout_keep_prob']:
                1,
            self.placeholders['iteration_mask_out']: [[1]],
            self.placeholders['is_generative']:
                True,
            self.placeholders['out_layer_dropout_keep_prob']:
                1.0,
            self.placeholders['distance_to_others_out']:
                distance_to_others,  # [1, 1,v]
            self.placeholders['overlapped_edge_features_out']:
                overlapped_edge_dense,
            self.placeholders['new_compound_freqs_out']:
                new_compound_freqs_dense,
            self.placeholders['new_compound_freqs_edge_out']:
                new_compound_freqs_edge_dense,
            self.placeholders['max_iteration_num']:
                1,
            self.placeholders['node_sequence_out']:
                node_sequence,  #[1, 1, v]
            self.placeholders['edge_type_masks_out']:
                edge_type_masks,  #[1, 1, e, v]
            self.placeholders['edge_masks_out']:
                edge_masks,  # [1, 1, v]
        }

    def get_node_symbol(self, batch_feed_dict):
        fetch_list = [self.ops['node_symbol_prob_in']]
        result = self.sess.run(fetch_list, feed_dict=batch_feed_dict)
        return result[0]

    def node_symbol_one_hot(self, sampled_node_symbol, real_n_vertices,
                            max_n_vertices):
        one_hot_representations = []
        for idx in range(max_n_vertices):
            representation = [0] * self.params["num_symbols"]
            if idx < real_n_vertices:
                atom_type = sampled_node_symbol[idx]
                representation[atom_type] = 1
            one_hot_representations.append(representation)
        return one_hot_representations

    def search_and_generate_molecule(self, initial_idx, valences,
                                     sampled_node_symbol, real_n_vertices,
                                     random_normal_states,
                                     random_normal_states_in, elements,
                                     max_n_vertices):
        # New molecule
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)

        # Add atoms
        add_atoms(new_mol, sampled_node_symbol, self.params["dataset"])
        # Initalise queue
        queue = deque([initial_idx])

        # color 0: have not found 1: in the queue 2: searched already
        color = [0] * max_n_vertices
        # Empty adj list at the beginning
        incre_adj_list = defaultdict(list)

        count_bonds = 0
        # Record the log probabilities at each step
        total_log_prob = 0

        while len(queue) > 0:
            node_in_focus = queue.popleft()
            # iterate until the stop node is selected
            while True:
                # Prepare data for one iteration based on the graph state
                edge_type_mask_sparse, edge_mask_sparse = generate_mask(
                    valences, incre_adj_list, color, real_n_vertices,
                    node_in_focus, self.params["check_overlap_edge"], new_mol)
                edge_type_mask = edge_type_masks_to_dense(
                    [edge_type_mask_sparse], max_n_vertices,
                    self.num_edge_types)  # [1, e, v]
                edge_mask = edge_masks_to_dense([edge_mask_sparse],
                                                max_n_vertices)  # [1, v]
                node_sequence = node_sequence_to_dense([node_in_focus],
                                                       max_n_vertices)  # [1, v]
                distance_to_others_sparse = bfs_distance(
                    node_in_focus, incre_adj_list)
                distance_to_others = distance_to_others_dense(
                    [distance_to_others_sparse], max_n_vertices)  # [1, v]
                overlapped_edge_sparse = get_overlapped_edge_feature(
                    edge_mask_sparse, color, new_mol)

                overlapped_edge_dense = overlapped_edge_features_to_dense(
                    [overlapped_edge_sparse], max_n_vertices)  # [1, v]
                new_compound_freqs_dense = np.ones(
                    (1, max_n_vertices))  # [1, v]
                new_compound_freqs_edge_dense = np.ones(
                    (1, self.num_edge_types, max_n_vertices))  # [1, e, v]
                incre_adj_mat = incre_adj_mat_to_dense(
                    [incre_adj_list], self.num_edge_types,
                    max_n_vertices)  # [1, e, v, v]
                sampled_node_symbol_one_hot = self.node_symbol_one_hot(
                    sampled_node_symbol, real_n_vertices, max_n_vertices)

                # get feed_dict
                feed_dict = self.get_dynamic_feed_dict(
                    elements, [sampled_node_symbol_one_hot], [incre_adj_mat],
                    max_n_vertices, [distance_to_others],
                    [overlapped_edge_dense], [new_compound_freqs_dense],
                    [new_compound_freqs_edge_dense], [node_sequence],
                    [edge_type_mask], [edge_mask], random_normal_states,
                    random_normal_states_in)

                # fetch nn predictions
                fetch_list = [
                    self.ops['edge_predictions_in'],
                    self.ops['edge_type_predictions_in']
                ]
                edge_probs, edge_type_probs = self.sess.run(fetch_list,
                                                            feed_dict=feed_dict)
                # select an edge
                if not self.params["use_argmax_generation"]:
                    neighbor = np.random.choice(np.arange(max_n_vertices + 1),
                                                p=edge_probs[0])
                else:
                    neighbor = np.argmax(edge_probs[0])
                # update log prob
                total_log_prob += np.log(edge_probs[0][neighbor] + SMALL_NUMBER)
                # stop it if stop node is picked
                if neighbor == max_n_vertices:
                    break
                # or choose an edge type
                if not self.params["use_argmax_generation"]:
                    bond = np.random.choice(np.arange(self.num_edge_types),
                                            p=edge_type_probs[0, :, neighbor])
                else:
                    bond = np.argmax(edge_type_probs[0, :, neighbor])
                # update log prob
                total_log_prob += np.log(edge_type_probs[0, :, neighbor][bond] +
                                         SMALL_NUMBER)
                #update valences
                valences[node_in_focus] -= (bond + 1)
                valences[neighbor] -= (bond + 1)
                #add the bond
                new_mol.AddBond(int(node_in_focus), int(neighbor),
                                number_to_bond[bond])
                # add the edge to increment adj list
                incre_adj_list[node_in_focus].append((neighbor, bond))
                incre_adj_list[neighbor].append((node_in_focus, bond))
                # Explore neighbor nodes
                if color[neighbor] == 0:
                    queue.append(neighbor)
                    color[neighbor] = 1
            color[node_in_focus] = 2  # explored
        # Remove unconnected node
        remove_extra_nodes(new_mol)
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        return new_mol, total_log_prob

    def generate_graph_with_state(self, random_normal_states,
                                  random_normal_states_in, num_vertices,
                                  generated_all_smiles, elements, count):
        # Get back node symbol predictions
        # Prepare dict
        node_symbol_batch_feed_dict = self.get_dynamic_feed_dict(
            elements, None, None, num_vertices, None, None, None, None, None,
            None, None, random_normal_states, random_normal_states_in)

        # Get predicted node probs (symbol and keep)
        predicted_node_symbol_prob = self.get_node_symbol(
            node_symbol_batch_feed_dict)
        # Node numbers for each graph
        real_length = get_graph_length([elements['mask_out']
                                       ])[0]  # [valid_node_number]
        # Sample node symbols
        sampled_node_symbol = sample_node_symbol(
            predicted_node_symbol_prob, [real_length],
            self.params["dataset"])[0]  # [v]

        # Maximum valences for each node
        valences = get_initial_valence(sampled_node_symbol,
                                       self.params["dataset"])  # [v]
        # Randomly pick the starting point or use zero
        if not self.params["path_random_order"]:
            # Try different starting points
            if self.params["try_different_starting"]:
                #starting_point=list(range(self.params["num_different_starting"]))
                starting_point = random.sample(
                    range(real_length),
                    min(self.params["num_different_starting"], real_length))
            else:
                starting_point = [0]
        else:
            if self.params["try_different_starting"]:
                starting_point = random.sample(
                    range(real_length),
                    min(self.params["num_different_starting"], real_length))
            else:
                starting_point = [random.choice(list(range(real_length)))
                                 ]  # randomly choose one
        # Record all molecules from different starting points
        all_mol = []
        for idx in starting_point:
            # Generate a new molecule
            new_mol, total_log_prob = self.search_and_generate_molecule(
                idx, np.copy(valences), sampled_node_symbol, real_length,
                random_normal_states, random_normal_states_in, elements,
                num_vertices)
            # If multiple starting points, select best only by total_log_prob
            if (self.params["dataset"] == 'zinc' or self.params["dataset"]
                    == 'zinc_phosphorus') and new_mol is not None:
                all_mol.append((0, total_log_prob, new_mol))
        # Select one out
        best_mol = select_best(all_mol)
        # Nothing generated
        if best_mol is None:
            return
        # Record the best molecule
        generated_all_smiles.append(elements['smiles_in'] + " " +
                                    Chem.MolToSmiles(best_mol))
        dump('%s_generated_smiles_%s' % (self.run_id, self.params["dataset"]),
             generated_all_smiles)
        # Progress
        if count % 100 == 0:
            print("Generated mol %d" % (count))

    def compensate_node_length(self, elements, bucket_size):
        maximum_length = bucket_size + self.params["compensate_num"]
        real_length = get_graph_length([elements['mask_in']])[0]
        real_length_out = get_graph_length([elements['mask_in']
                                           ])[0] + self.params["compensate_num"]
        elements['mask_out'] = [1] * real_length_out + [0] * (maximum_length -
                                                              real_length_out)
        elements['init_out'] = np.zeros(
            (maximum_length, self.params["num_symbols"]))
        elements['adj_mat_out'] = np.zeros(
            (self.num_edge_types, maximum_length, maximum_length))

        elements['mask_in'] = [1] * real_length + [0] * (
            maximum_length - real_length)  # [v] -> [(v+comp)]
        elements['init_in'] = np.pad(
            elements['init_in'],
            pad_width=[[0, self.params["compensate_num"]], [0, 0]],
            mode='constant')  # [v, h] -> [v+comp, h]
        elements['adj_mat_in'] = np.pad(
            elements['adj_mat_in'],
            pad_width=[[0, 0], [0, self.params["compensate_num"]],
                       [0, self.params["compensate_num"]]],
            mode='constant')  # [e, v, v] -> [e, v+comp, v+comp]
        return maximum_length

    def generate_new_graphs(self, data):
        # bucketed: data organized by bucket
        (bucketed, bucket_sizes, bucket_at_step) = data
        bucket_counters = defaultdict(int)
        # All generated SMILES
        generated_all_smiles = []
        # Counter
        count = 0
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]  # bucket number
            # data index
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            # batch data
            elements_batch = bucketed[bucket][start_idx:end_idx]
            for elements in elements_batch:
                # Allow control over number of additional atoms during generation
                maximum_length = self.compensate_node_length(
                    elements, bucket_sizes[bucket])
                # Generate multiple outputs per mol in valid/test set
                for _ in range(self.params['number_of_generation_per_valid']):
                    # initial state
                    random_normal_states=generate_std_normal(1, maximum_length,\
                                                         self.params['encoding_size']) # [1, v, j]
                    random_normal_states_in = generate_std_normal(1, maximum_length,\
                                                         self.params['hidden_size']) # [1, v, h]
                    self.generate_graph_with_state(random_normal_states,
                                                   random_normal_states_in,
                                                   maximum_length,
                                                   generated_all_smiles,
                                                   elements, count)
                    count += 1
            bucket_counters[bucket] += 1
        # Terminate when loop finished
        print("Generation done")
        # Save output in non-pickle format
        print("Number of generated SMILES: %d" % len(generated_all_smiles))
        if self.params['output_name'] != '':
            file_name = self.params['output_name']
        else:
            file_name = '%s_generated_smiles_%s.smi' % (self.run_id,
                                                        self.params["dataset"])
        with open(file_name, 'w') as out_file:
            for line in generated_all_smiles:
                out_file.write(line + '\n')

    def make_minibatch_iterator(self, data, is_training: bool):
        (bucketed, bucket_sizes, bucket_at_step) = data
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)
        bucket_counters = defaultdict(int)
        dropout_keep_prob = self.params[
            'graph_state_dropout_keep_prob'] if is_training else 1.
        edge_dropout_keep_prob = self.params[
            'edge_weight_dropout_keep_prob'] if is_training else 1.
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements, bucket_sizes[bucket])

            num_graphs = len(batch_data['init_in'])
            initial_representations_in = batch_data['init_in']
            initial_representations_in = self.pad_annotations(
                initial_representations_in)
            initial_representations_out = batch_data['init_out']
            initial_representations_out = self.pad_annotations(
                initial_representations_out)
            batch_feed_dict = {
                self.placeholders['initial_node_representation_in']:
                    initial_representations_in,
                self.placeholders['initial_node_representation_out']:
                    initial_representations_out,
                self.placeholders['node_symbols_out']:
                    batch_data['init_out'],
                self.placeholders['node_symbols_in']:
                    batch_data['init_in'],
                self.placeholders['latent_node_symbols_in']:
                    initial_representations_in,
                self.placeholders['latent_node_symbols_out']:
                    initial_representations_out,
                self.placeholders['num_graphs']:
                    num_graphs,
                self.placeholders['num_vertices']:
                    bucket_sizes[bucket],
                self.placeholders['adjacency_matrix_in']:
                    batch_data['adj_mat_in'],
                self.placeholders['adjacency_matrix_out']:
                    batch_data['adj_mat_out'],
                self.placeholders['node_mask_in']:
                    batch_data['node_mask_in'],
                self.placeholders['node_mask_out']:
                    batch_data['node_mask_out'],
                self.placeholders['graph_state_keep_prob']:
                    dropout_keep_prob,
                self.placeholders['edge_weight_dropout_keep_prob']:
                    edge_dropout_keep_prob,
                self.placeholders['iteration_mask_out']:
                    batch_data['iteration_mask_out'],
                self.placeholders['incre_adj_mat_out']:
                    batch_data['incre_adj_mat_out'],
                self.placeholders['distance_to_others_out']:
                    batch_data['distance_to_others_out'],
                self.placeholders['node_sequence_out']:
                    batch_data['node_sequence_out'],
                self.placeholders['edge_type_masks_out']:
                    batch_data['edge_type_masks_out'],
                self.placeholders['edge_type_labels_out']:
                    batch_data['edge_type_labels_out'],
                self.placeholders['edge_masks_out']:
                    batch_data['edge_masks_out'],
                self.placeholders['edge_labels_out']:
                    batch_data['edge_labels_out'],
                self.placeholders['local_stop_out']:
                    batch_data['local_stop_out'],
                self.placeholders['max_iteration_num']:
                    batch_data['max_iteration_num'],
                self.placeholders['kl_trade_off_lambda']:
                    self.params['kl_trade_off_lambda'],
                self.placeholders['overlapped_edge_features_out']:
                    batch_data['overlapped_edge_features_out'],
                self.placeholders['new_compound_freqs_out']:
                    batch_data['new_compound_freqs_out'],
                self.placeholders['new_compound_freqs_edge_out']:
                    batch_data['new_compound_freqs_edge_out'],
            }
            bucket_counters[bucket] += 1
            yield batch_feed_dict


if __name__ == "__main__":
    args = docopt(__doc__)
    dataset = args.get('--dataset')
    try:
        model = DenseGGNNChemModel(args)
        evaluation = False
        if evaluation:
            model.example_evaluation()
        else:
            model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
