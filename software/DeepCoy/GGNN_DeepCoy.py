#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence
import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random
from ..DeepCoy import utils
from ..DeepCoy.utils import MLP, dataset_info, ThreadedIterator, graph_to_adj_mat, SMALL_NUMBER, LARGE_NUMBER, graph_to_adj_mat


class ChemModel(object):

    @classmethod
    def default_params(cls):
        return {}

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params

        # Get which dataset in use
        self.params['dataset'] = dataset = args.get('--dataset')
        # Number of atom types of this dataset
        self.params['num_symbols'] = len(dataset_info(dataset)["atom_types"])

        self.run_id = "_".join(
            [time.strftime("%Y-%m-%d-%H-%M-%S"),
             str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir,
                                     "%s_log_%s.json" % (self.run_id, dataset))
        self.best_model_file = os.path.join(log_dir,
                                            "%s_model.pickle" % self.run_id)

        with open(
                os.path.join(log_dir,
                             "%s_params_%s.json" % (self.run_id, dataset)),
                "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" %
              (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        # Load subgraph frequency dictionary
        if params['subgraph_freq_file']:
            self.freq_dict = pickle.load(
                open(params['subgraph_freq_file'], 'rb'))
            print("Loaded subgraph frequency dictionary from %s" %
                  params['subgraph_freq_file'])
        else:
            self.freq_dict = {}
            print("Subgraph frequency dictionary not used.")

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data(params['train_file'],
                                         is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'],
                                         is_training_data=False)

        # Build the actual model
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and int(restrict) > 0:
            data = data[:int(restrict)]

        # Get some common data out:
        num_fwd_edge_types = len(utils.bond_dict) - 1
        for g in data:
            self.max_num_vertices = max(
                self.max_num_vertices,
                max([v for e in g['graph_in'] for v in [e[0], e[2]]]),
                max([v for e in g['graph_out'] for v in [e[0], e[2]]]))

        self.num_edge_types = max(
            self.num_edge_types,
            num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size,
                                   len(data[0]["node_features_in"][0]))

        return self.process_raw_graphs(data, is_training_data, file_name)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')] for s in graph_string.split('\n')
               ]

    def process_raw_graphs(self,
                           raw_data,
                           is_training_data,
                           file_name,
                           bucket_sizes=None):
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['num_graphs'] = tf.compat.v1.placeholder(
            tf.int64, [], name='num_graphs')
        self.placeholders[
            'out_layer_dropout_keep_prob'] = tf.compat.v1.placeholder(
                tf.float32, [], name='out_layer_dropout_keep_prob')
        # whether this session is for generating new graphs or not
        self.placeholders['is_generative'] = tf.compat.v1.placeholder(
            tf.bool, [], name='is_generative')

        with tf.compat.v1.variable_scope("graph_model"):
            self.prepare_specific_graph_model()

            # Initial state: embedding
            initial_state_in = self.get_node_embedding_state(
                self.placeholders['initial_node_representation_in'],
                source=True)
            initial_state_out = self.get_node_embedding_state(
                self.placeholders['initial_node_representation_out'],
                source=False)

            # This does the actual graph work:
            if self.params['use_graph']:
                if self.params["residual_connection_on"]:
                    self.ops[
                        'final_node_representations_in'] = self.compute_final_node_representations_with_residual(
                            initial_state_in,
                            tf.transpose(
                                self.placeholders['adjacency_matrix_in'],
                                [1, 0, 2, 3]), "_encoder")
                    self.ops[
                        'final_node_representations_out'] = self.compute_final_node_representations_with_residual(
                            initial_state_out,
                            tf.transpose(
                                self.placeholders['adjacency_matrix_out'],
                                [1, 0, 2, 3]), "_encoder")

                else:
                    self.ops[
                        'final_node_representations_in'] = self.compute_final_node_representations_without_residual(
                            initial_state_in,
                            tf.transpose(
                                self.placeholders['adjacency_matrix_in'],
                                [1, 0, 2, 3]),
                            self.weights['edge_weights_encoder'],
                            self.weights['edge_biases_encoder'],
                            self.weights['node_gru_encoder'],
                            "gru_scope_encoder")
                    self.ops[
                        'final_node_representations_out'] = self.compute_final_node_representations_without_residual(
                            initial_state_out,
                            tf.transpose(
                                self.placeholders['adjacency_matrix_out'],
                                [1, 0, 2, 3]),
                            self.weights['edge_weights_encoder'],
                            self.weights['edge_biases_encoder'],
                            self.weights['node_gru_encoder'],
                            "gru_scope_encoder")

            else:
                self.ops['final_node_representations_in'] = initial_state_in
                self.ops['final_node_representations_out'] = initial_state_out

        # Calculate p(z|x)'s mean and log variance
        self.ops['mean'], self.ops['logvariance'], self.ops[
            'mean_out'], self.ops[
                'logvariance_out'] = self.compute_mean_and_logvariance()
        # Sample from a gaussian distribution according to the mean and log variance
        self.ops['z_sampled_in'] = self.sample_with_mean_and_logvariance()
        # Construct logit matrices for both edges and edge types
        self.construct_logit_matrices()

        # Obtain losses
        self.ops['loss'] = self.construct_loss()

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(
                self.sess.graph.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                    scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars

        optimizer = tf.compat.v1.train.AdamOptimizer(
            self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'],
                                                     var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append(
                    (tf.clip_by_norm(grad,
                                     self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        grads_for_display = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grads_for_display.append(
                    (tf.clip_by_norm(grad,
                                     self.params['clamp_gradient_norm']), var))
        self.ops['grads'] = grads_for_display
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.compat.v1.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception(
            "Models have to implement prepare_specific_graph_model!")

    def compute_mean_and_logvariance(self):
        raise Exception(
            "Models have to implement compute_mean_and_logvariance!")

    def sample_with_mean_and_logvariance(self):
        raise Exception(
            "Models have to implement sample_with_mean_and_logvariance!")

    def construct_logit_matrices(self):
        raise Exception("Models have to implement construct_logit_matrices!")

    def construct_loss(self):
        raise Exception("Models have to implement construct_loss!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def save_probs(self, all_results):
        with open('epoch_prob_matices_%s' % self.params["dataset"],
                  'wb') as out_file:
            pickle.dump([all_results], out_file, pickle.HIGHEST_PROTOCOL)

    def run_epoch(self, epoch_name: str, epoch_num, data, is_training: bool):
        loss = 0
        edge_loss, kl_loss, node_symbol_loss = 0, 0, 0
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(
            data, is_training),
                                          max_queue_size=5)

        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            batch_data[self.placeholders['is_generative']] = False
            # Randomly sample from normal distribution
            batch_data[self.placeholders['z_prior']] = utils.generate_std_normal(\
                self.params['batch_size'], batch_data[self.placeholders['num_vertices']],self.params['encoding_size'])
            batch_data[self.placeholders['z_prior_in']] = utils.generate_std_normal(\
                self.params['batch_size'], batch_data[self.placeholders['num_vertices']],self.params['hidden_size'])

            if is_training:
                batch_data[self.placeholders[
                    'out_layer_dropout_keep_prob']] = self.params[
                        'out_layer_dropout_keep_prob']
                fetch_list = [
                    self.ops['loss'], self.ops['mean_edge_loss_in'],
                    self.ops['mean_kl_loss_in'],
                    self.ops['mean_node_symbol_loss_in'], self.ops['train_step']
                ]
            else:
                batch_data[
                    self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [
                    self.ops['loss'], self.ops['mean_edge_loss_in'],
                    self.ops['mean_kl_loss_in'],
                    self.ops['mean_node_symbol_loss_in']
                ]

            result = self.sess.run(fetch_list, feed_dict=batch_data)

            batch_loss = result[0]
            loss += batch_loss * num_graphs

            edge_loss += result[1] * num_graphs
            kl_loss += result[2] * num_graphs
            node_symbol_loss += result[3] * num_graphs

            print(
                "Running %s, batch %i (has %i graphs). Loss so far: %.4f. Edge loss: %.4f, KL loss: %.4f, Node symbol loss: %.4f"
                % (epoch_name, step, num_graphs, loss / processed_graphs,
                   edge_loss / processed_graphs, kl_loss / processed_graphs,
                   node_symbol_loss / processed_graphs),
                end='\r')

        loss = loss / processed_graphs
        edge_loss = edge_loss / processed_graphs
        kl_loss = kl_loss / processed_graphs
        node_symbol_loss = node_symbol_loss / processed_graphs
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return (loss, edge_loss, kl_loss, node_symbol_loss), instance_per_sec

    def generate_new_graphs(self, data):
        raise Exception("Models have to implement generate_new_graphs!")

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            for epoch in range(1, self.params['num_epochs'] + 1):
                if not self.params['generation']:
                    print("== Epoch %i" % epoch)

                    train_losses, train_speed = self.run_epoch(
                        "epoch %i (training)" % epoch, epoch, self.train_data,
                        True)
                    print(
                        "\r\x1b[K Train: Total loss: %.4f | Edge loss: %.2f, KL loss: %.2f, Node symbol loss: %.2f | instances/sec: %.2f"
                        % (train_losses[0], train_losses[1], train_losses[2],
                           train_losses[3], train_speed))

                    valid_losses, valid_speed = self.run_epoch(
                        "epoch %i (validation)" % epoch, epoch, self.valid_data,
                        False)

                    print(
                        "\r\x1b[K Valid: Total loss: %.4f | Edge loss: %.2f, KL loss: %.2f, Node symbol loss: %.2f | instances/sec: %.2f"
                        % (valid_losses[0], valid_losses[1], valid_losses[2],
                           valid_losses[3], valid_speed))

                    epoch_time = time.time() - total_time_start

                    log_entry = {
                        'epoch':
                            epoch,
                        'time':
                            epoch_time,
                        'results_format':
                            "Total loss, Edge loss, KL loss, Node symbol loss, instances/sec",
                        'train_results': (train_losses, train_speed),
                        'valid_results': (valid_losses, valid_speed),
                    }
                    log_to_save.append(log_entry)
                    with open(self.log_file, 'w') as f:
                        json.dump(log_to_save, f, indent=4)
                    self.save_model(self.run_id + "_" + str(epoch) +
                                    ("_%s.pickle" % (self.params["dataset"])))
                # Run epoches for graph generation
                if epoch >= self.params['epoch_to_generate']:
                    self.generate_new_graphs(self.valid_data)
                    break

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {"params": self.params, "weights": weights_to_save}

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        variables_to_initialize = []
        with tf.compat.v1.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(
                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(
                        variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print(
                        'Freshly initializing %s since no saved value was found.'
                        % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(
                tf.compat.v1.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)
