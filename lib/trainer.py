from __future__ import print_function
import sys
import json
import importlib
import pdb
import traceback
import string
import numpy as np
from collections import defaultdict
import inspect
import tensorflow as tf
from data import Data
import datetime as dt
import time
from pprint import pprint

class Trainer(object):
    """
    In this class, hyperparameters refer to training hyperparameters.
    Hyperparameters from the model specification (like num_timesteps)
    will be called "model_hypers".
    """

    BUILD_MODEL_PREFIX = "build_model_"

    def __init__(self, model, model_module, model_hypers, train_hypers, data, seed=0, do_debug=False, make_log=False, logdir="logs"):
        self.set_seed(seed)
        self.model = model
        self.model_hypers = model_hypers
        self.train_hypers = train_hypers
        self.data = data
        self.do_debug = do_debug
        self.make_log = make_log
        self.logdir = logdir

        self.load_terpret_runtime(model_module)
        self.initialize_tensorflow()
        self.map_model_hypers_to_build_model_methods()
        self.build_computation_graphs()

    @classmethod
    def default_train_hypers(cls):
        return {
            "optimizer": "rmsprop",
            "num_epochs": 2000,
            "stop_below_loss": 0.00001,
            "learning_rate": 0.01,
            "learning_rate_decay": 0.9,
            "momentum": 0.0,
            "minibatch_size": 10,
            "print_frequency": 100,
            "validate_frequency": 100
        }

    def set_seed(self, seed):
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def initialize_tensorflow(self):
        config_proto = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        self.sess = tf.Session(config=config_proto)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.num_updates_applied = 0
        self.train_loss_values = []  # stores losses from do_one_update
        self.eval_loss_values = defaultdict(list)  # stores losses from evaluate_loss
        self.check_ops = []
        self.summary_nodes = {}

    def set_tf_session(self, session):
        self.sess = session

    def map_model_hypers_to_build_model_methods(self):
        members = inspect.getmembers(self.model.__class__, predicate=inspect.ismethod)
        self.model_hypers_to_build_graph = {}
        for method_name, method in members:
            if method_name.startswith(Trainer.BUILD_MODEL_PREFIX):
                model_hypers_name = method_name[len(Trainer.BUILD_MODEL_PREFIX):]
                self.model_hypers_to_build_graph[model_hypers_name] = method

    def construct_output_loss_nodes(self, output_nodes):
        output_datas = {}
        output_masks = {}
        loss_nodes = {}
        for var_name in output_nodes.keys():
            data_var_name = "%s_data" % var_name
            mask_var_name = "%s_mask" % var_name
            data_node = tf.placeholder(tf.int32, shape=[None], name=data_var_name)
            mask_node = tf.to_float(tf.placeholder(tf.int32, shape=[None]), name=mask_var_name)
            output_node = output_nodes[var_name]
            output_rank = output_node.get_shape().ndims
            if output_rank == 1:
                output_node = tf.tile(tf.expand_dims(output_node, 0), [tf.shape(data_node)[0], 1])
                mask_node = tf.tile(tf.expand_dims(mask_node, 0), [tf.shape(data_node)[0], 1])
            loss_node = self.tpt.observe(output_node, data_node, mask_node,
                                         scope="%s_observe" % var_name)
            output_datas[var_name] = data_node
            output_masks[mask_var_name] = mask_node
            loss_nodes[var_name] = loss_node
        return (output_datas, output_masks, loss_nodes)

    def construct_loss(self, output_nodes):
        """
        This gets the nodes declared as Output() as output_nodes. Per
        default, loss is constructed as the sum of cross-entropy
        losses for the outputs, averaged over the considered
        instances.  For this, it uses construct_output_data_nodes to
        build placeholders for the observed data.  It returns a loss,
        a loss to be shown during training, and a dict mapping names
        to generated placeholder nodes (which will have to be filled
        by data instances).

        """
        (output_datas, output_masks, loss_nodes) = self.construct_output_loss_nodes(output_nodes)

        avg_losses = []
        for name, loss_node in loss_nodes.iteritems():
            avg_losses.append(tf.reduce_mean(loss_node))  # average across instances

        # sum across variables
        loss = tf.add_n(avg_losses)
        display_loss = loss  # loss to be displayed
        return loss, display_loss, output_datas, output_masks, loss_nodes

    def make_optimizer(self):
        if self.train_hypers["optimizer"] == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.train_hypers["learning_rate"])

        elif self.train_hypers["optimizer"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.train_hypers["learning_rate"],
                                                  decay=self.train_hypers["learning_rate_decay"],
                                                  momentum=self.train_hypers["momentum"])

        elif self.train_hypers["optimizer"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.train_hypers["learning_rate"])

        elif self.train_hypers["optimizer"] == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.train_hypers["learning_rate"],
                                                   momentum=self.train_hypers["momentum"])

        else:
            assert False, "Unexpected optimizer: %s" % self.train_hypers["optimizer"]
        return optimizer

    def log_dirichlet(self, size, scale=1.0):
        mu = tf.random_gamma([1], scale * np.ones(size).astype(np.float32))
        mu = tf.log(mu / tf.reduce_sum(mu))
        return mu

    def param_init_function(self, size):
        return self.log_dirichlet(size)

    def build_computation_graphs(self):
        self.model.declare_params(self.param_init_function)

        self.tf_nodes = {}
        to_build = {k:v for k, v in self.model_hypers_to_build_graph.iteritems() 
                        if k in self.data.get_hypers_names()}

        for model_hypers, build_graph in to_build.iteritems():
            print ("Construct forward graph... ", end="")

            forward_time_start = time.time()
            inputs, outputs = build_graph(self.model)
            loss, display_loss, output_placeholders, mask_placeholders, loss_nodes = \
                self.construct_loss(outputs)
            print ("done in %.2fs." % (time.time() - forward_time_start))

            optimizer = self.make_optimizer()

            gradient_time_start = time.time()
            print ("Construct gradient graph... ", end="")
            grads_and_vars = self.compute_gradients(optimizer, loss)
            print ("done in %.2fs." % (time.time() - gradient_time_start))

            gradient_apply_time_start = time.time()
            print ("Construct apply gradient graph... ", end="")
            train_op = self.apply_update(optimizer, grads_and_vars)
            print ("done in %.2fs." % (time.time() - gradient_apply_time_start))

            if self.do_debug:
                check_time_start = time.time()
                print ("Construct check numerics graph... ", end="")
                self.check_ops.append(tf.add_check_numerics_ops())
                print ("done in %.2fs." % (time.time() - check_time_start))

            if self.make_log:
                self.summary_nodes["train"] = tf.scalar_summary('train_loss', display_loss)
                self.summary_nodes["validate"] = tf.scalar_summary('validate_loss', display_loss)
                self.summary_nodes["params"] = []
                for p_name, p_node in self.model.params.iteritems():
                    n_elements = p_node.get_shape()[0].value
                    for i in range(n_elements):
                        self.summary_nodes["params"].append(
                            tf.scalar_summary('%s/%i' % (p_name, i), p_node[i]))


            placeholders = {}
            placeholders.update(inputs)
            placeholders.update(output_placeholders)
            placeholders.update(mask_placeholders)
            self.tf_nodes[model_hypers] = {
                "inputs": inputs,
                "outputs": outputs,
                "placeholders": placeholders,
                "loss_nodes": loss_nodes,
                "loss": loss,
                "display_loss": display_loss,
                "grads_and_vars": grads_and_vars,
                "train_op": train_op
            }

    def initialize_training(self):
        """
        Call this once at the start of training.
        """
        init_time_start = time.time()
        print("Initializing variables... ", end="")
        self.num_updates_applied = 0
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.global_step.assign(0))
        print("done in %.2fs." % (time.time() - init_time_start))

    def initialize_summary_writers(self, run_id=None):
        if run_id is None:
            random_id = ''.join(np.random.choice(list(string.lowercase), 4))
            run_id = "%s_%s" % (dt.datetime.now().strftime('%Y%m%d_%H%M%S'), random_id)
        self.make_summary_writers(run_id)

    def compute_gradients(self, optimizer, loss):
        return optimizer.compute_gradients(loss)

    def apply_update(self, optimizer, grads_and_vars):
        """
        Override this to, e.g., apply gradient noise.
        """
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op

    def get_minibatch(self, batch_name, minibatch_size):
        hypers_name, minibatch = self.data.get_random_minibatch(batch_name, minibatch_size)

        observed_vals = {}
        for var_name, val_mask_dict in minibatch.iteritems():
            node = self.tf_nodes[hypers_name]["placeholders"][var_name]
            observed_vals[node] = val_mask_dict["values"]

            mask_node_name = "%s_mask" % var_name
            if mask_node_name in self.tf_nodes[hypers_name]["placeholders"]:
                mask_node = self.tf_nodes[hypers_name]["placeholders"][mask_node_name]
                observed_vals[mask_node] = val_mask_dict["masks"]

        return hypers_name, observed_vals

    def transform_observed_vals(self, observed_vals):
        """
        By default this is an identity op, but when overriding loss function
        construction, it can be useful to transform the observation data
        before it is passed in to tensorflow. This method allows this without
        having to override do_one_update.
        """
        return observed_vals

    def do_one_update(self, batch_name):
        hypers_name, observed_vals = self.get_minibatch(batch_name,
                                                        self.train_hypers["minibatch_size"])
        observed_vals = self.transform_observed_vals(observed_vals)

        nodes = self.tf_nodes[hypers_name]
        train_op, display_loss = nodes["train_op"], nodes["display_loss"]
        ops_to_execute = [train_op, display_loss]
        if self.do_debug:
            ops_to_execute += self.check_ops
        if self.make_log:
            ops_to_execute.append(self.summary_nodes["train"])
            summary_idx = len(ops_to_execute) - 1

        results = self.sess.run(ops_to_execute, observed_vals)

        loss_val = results[1]
        self.train_loss_values.append(loss_val)
        if self.make_log:
            self.summary_writers["train"].add_summary(
                results[summary_idx], self.num_updates_applied)

        self.num_updates_applied += 1
        if self.num_updates_applied % self.train_hypers["print_frequency"] == 0:
            print( "epoch %i, loss %s, needed %.2fs" % (self.num_updates_applied, loss_val, time.time() - self.last_print_time))
            self.last_print_time = time.time()
            if self.make_log:
                self.summary_writers["train"].flush()

    def record_params(self, writer):
        if self.summary_nodes["params"]:
            param_values = self.sess.run(self.summary_nodes["params"])
            for s in param_values:
                writer.add_summary(s, self.num_updates_applied)

    def evaluate_loss(self, batch_name, show_loss_breakdown=False):
        hypers_name, observed_vals = self.get_minibatch(batch_name, -1)  # get all instances
        display_loss = self.tf_nodes[hypers_name]["display_loss"]
        ops_to_execute = [display_loss]
        if self.make_log:
            ops_to_execute.append(self.summary_nodes["validate"])
            summary_idx = len(ops_to_execute) - 1

        results = self.sess.run(ops_to_execute, observed_vals)
        loss_val = results[0]
        self.eval_loss_values[batch_name].append(loss_val)
        if self.make_log:
            self.summary_writers["validate"].add_summary(
                results[summary_idx], self.num_updates_applied)
            self.summary_writers["validate"].flush()

        print("*** eval ***", batch_name, loss_val)

        if show_loss_breakdown:
            ops_to_execute = self.tf_nodes[hypers_name]["loss_nodes"]
            results = self.sess.run(ops_to_execute, observed_vals)
            print(results)

        return loss_val

    def make_test_node(self, hypers_name):
        outputs = self.tf_nodes[hypers_name]["outputs"]

        deltas = []
        for var_name, output_node in outputs.iteritems():
            data_node = self.tf_nodes[hypers_name]["placeholders"][var_name]
            output_rank = output_node.get_shape().ndims
            if output_rank == 1:
                output_node = tf.tile(tf.expand_dims(output_node, 0), [tf.shape(data_node)[0], 1])
            deltas.append(
                tf.to_int32(tf.argmax(output_node, dimension=1)) - data_node)

        zero_if_correct = tf.reduce_sum(tf.pack(deltas), reduction_indices=0)
        zero_elements = tf.equal(zero_if_correct, tf.zeros_like(zero_if_correct))
        n_correct = tf.reduce_sum(tf.to_int32(zero_elements))
        n_total = tf.shape(zero_if_correct)[0]
        accuracy = tf.truediv(n_correct, n_total)
        self.summary_nodes["test"] = tf.scalar_summary('test_accuracy', accuracy)
        self.tf_nodes[hypers_name]["accuracy"] = accuracy

    def evaluate_accuracy(self, batch_name_list):
        def go(batch_name):
            hypers_name, observed_vals = self.get_minibatch(batch_name, -1)  # get all instances
            if "accuracy" not in self.tf_nodes[hypers_name]:
                self.make_test_node(hypers_name)
            ops_to_execute = [self.tf_nodes[hypers_name]["accuracy"],
                              self.summary_nodes["test"]]
            results = self.sess.run(ops_to_execute, observed_vals)

            if self.make_log:
                self.summary_writers["test"].add_summary(
                    results[1], self.num_updates_applied)
                self.summary_writers["test"].flush()

            print("*** eval ***", batch_name, results[0])
        if isinstance(batch_name_list, list):
            for batch_name in batch_name_list:
                go(batch_name)
        else:
            go(batch_name)

    def make_summary_writers(self, id):
        self.summary_writers = {}
        for stage in ["train", "validate", "test"]:
            summary_file = "%s/%s/%s" % (self.logdir, id, stage)
            self.summary_writers[stage] = tf.train.SummaryWriter(summary_file, None)

    def param_values(self):
        keys, nodes, vals = [], [], []
        for name, logits in self.model.params.iteritems():
            keys.append(name)
            nodes.append(logits)
        if nodes:
            vals = self.sess.run(nodes)
            return dict(zip(keys, vals))
        else:
            return {}

    def var_values(self, batch_name):
        hypers_name, observed_vals = self.get_minibatch(batch_name, -1)
        var_names = self.model.var_nodes.keys()
        var_nodes = self.model.var_nodes.values()
        var_values = self.sess.run(var_nodes, observed_vals)

        param_var_names = self.model.param_var_nodes.keys()
        param_var_nodes = self.model.param_var_nodes.values()
        param_var_values = self.sess.run(param_var_nodes, observed_vals)

        res = dict(zip(var_names, var_values))
        res.update(dict(zip(param_var_names, param_var_values)))
        return res

    def discretized_param_values(self):
        param_vals = self.param_values()
        discretized_param_vals = {}
        for key, vals in param_vals.iteritems():
            if len(vals) == 0:
                discretized_param_vals[key] = None
            else:
                discretized_param_vals[key] = np.argmax(vals)
        return discretized_param_vals

    def pretty_print_dict(self, dict_to_print):
        for name in sorted(dict_to_print):
            vals = dict_to_print[name]
            print ('%s\t: ' % name, end="")
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print (np.exp(vals) / np.sum(np.exp(vals), axis=vals.ndim-1, keepdims=True))

    def pretty_print_params(self, params, discretized_params):
        print ("** PARAMS **")
        self.pretty_print_dict(params)

        print ("** DISCRETIZED PARAMS **")
        for name in sorted(discretized_params):
            v = discretized_params[name]
            print ('%s: %i' % (name, v))

    def train(  
            self, 
            train_batch_name, 
            validation_batch_name=None, 
            test_batch_name=None, 
            print_params=False, 
            print_final_loss_breakdown=False, 
            do_initialization=True,
            run_id = None
            ):
        self.do_train = train_batch_name is not None
        self.do_validate = (validation_batch_name is not None) and \
                           (self.train_hypers.get("validate_frequency", -1) is not -1)
        self.do_test = (test_batch_name is not None) and \
                       (self.train_hypers.get("test_frequency", -1) is not -1)

        if do_initialization:
            self.initialize_training()

        if self.make_log:
            self.initialize_summary_writers(run_id)

        pprint(self.train_hypers)

        stop_below_loss = self.train_hypers.get('stop_below_loss', None)
        self.last_print_time = time.time()
        for i in xrange(self.train_hypers["num_epochs"]):
            # train
            if self.do_train:
                self.do_one_update(train_batch_name)

            # validate
            if self.do_validate and (i % self.train_hypers["validate_frequency"] is 0):
                self.evaluate_loss(validation_batch_name, show_loss_breakdown=True)
                if self.make_log: self.record_params(self.summary_writers["validate"])

            # test
            if self.do_test and (i % self.train_hypers["test_frequency"] is 0):
                self.evaluate_accuracy(test_batch_name)

            if stop_below_loss is not None and self.train_loss_values[-1] < float(stop_below_loss):
                print ("  Early training termination after %i epochs because loss %f is below bound %f." % (i, self.train_loss_values[-1], float(stop_below_loss)))
                break

        # Always run test after finishing training:
        if test_batch_name is not None:
            self.evaluate_accuracy(test_batch_name)

        if print_final_loss_breakdown:
            for name, loss_val in self.get_loss_breakdown(train_batch_name).iteritems():
                print ("  ", name, loss_val)

        params = self.param_values()
        discretized_params = self.discretized_param_values()
        if print_params:
            self.pretty_print_params(params, discretized_params)

        if self.make_log:
            for stage in ["train", "validate", "test"]:
                self.summary_writers[stage].flush()
                self.summary_writers[stage].close()

        return params, discretized_params




    def get_loss_breakdown(self, batch_name):
        """
        Report the individual contributions to the loss.
        """
        # get all instances
        hypers_name, observed_vals = self.get_minibatch(batch_name, -1)

        loss_nodes_dict = self.tf_nodes[hypers_name]["loss_nodes"]
        loss_names = loss_nodes_dict.keys()
        loss_nodes = loss_nodes_dict.values()

        results = self.sess.run(loss_nodes, observed_vals)

        return dict(zip(loss_names, results))

    def load_terpret_runtime(self, module):
        if module.runtime == 'logspace':
            self.tpt = importlib.import_module("terpret_tf_log_runtime")
        else:
            self.tpt = importlib.import_module("terpret_tf_runtime")
