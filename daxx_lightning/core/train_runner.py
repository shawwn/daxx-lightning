# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bypass TPUEstimator for ResNet-50 Train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import threading
import time
import os

from absl import flags
import tensorflow as tf
from . import tflex
Session = tf.Session

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7


def device_for_tpu_core(task=0, core=0):
  job_name = FLAGS.tpu_job_name or "tpu_worker"
  return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job_name, task,
                                                            core)


def wrap_computation_in_while_loop(op_fn, n, parallel_iterations=1):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    ops = op_fn()
    if not isinstance(ops, list):
      ops = [ops]
    with tf.control_dependencies(ops):
      return i + 1

  return tf.while_loop(
      lambda i: tf.less(i, n),
      computation, [tf.constant(0)],
      parallel_iterations=parallel_iterations)


def tpu_ordinal_fn(shard_index_in_host):
  """Return the TPU ordinal associated with a shard.

  Required because the enqueue ops are placed on CPU.

  Args:
    shard_index_in_host: the shard index

  Returns:
    The ordinal of the TPU device the shard's infeed should be placed on.
  """
  return shard_index_in_host % FLAGS.tpu_cores_per_host


class TrainRunner(object):
  """Remove init overheads in TPU Estimator via direct session.run calls."""

  def __init__(self, iterations, train_steps):
    tf.logging.info("TrainRunner: constructor")
    self.feature_structure = {}
    self.loss = None
    self.infeed_queue = []
    self.enqueue_ops = []
    self.dataset_initializer = []
    self.iterations = iterations
    self.sess = None
    self.input_sess = None
    self.infeed_thread = None
    if train_steps % iterations != 0:
      train_steps = iterations * int(math.ceil(train_steps / iterations))
    self.train_steps = train_steps
    self.init_graph = tf.Graph()
    self.input_graph = tf.Graph()
    self.cluster_resolver = tflex.TPUClusterResolver(
        FLAGS.tpu or FLAGS.master,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    self.config = tf.ConfigProto(operation_timeout_in_ms=600 * 60 * 1000,
                                 graph_options=tf.GraphOptions(
                                     rewrite_options=rewriter_config_pb2.RewriterConfig(
                                         disable_meta_optimizer=True)),
                                 isolate_session_state=True)
    cluster_spec = self.cluster_resolver.cluster_spec()
    if cluster_spec:
      self.config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    with self.init_graph.as_default():
      self.tpu_init = tpu.initialize_system()
      self.tpu_shutdown = tpu.shutdown_system()
    self.init_sess = Session(self.cluster_resolver.get_master(), graph=self.init_graph, config=self.config)
    if 'NO_TPU_INIT' not in os.environ:
      tf.logging.info("initializing TPU...")
      self.init_sess.run(self.tpu_init)
      if 'EXIT_AFTER_TPU_INIT' in os.environ:
        import posix
        posix._exit(0)

  def device_for_host(self, task=0, cpu=0):
    job_name = FLAGS.tpu_job_name or "tpu_worker"
    return "/job:%s/task:%d/device:CPU:%d" % (job_name, task, cpu)

  def build_enqueue_ops(self, input_fn, params, host_id):
    """Build enqueue operations for the input pipeline in a given host.

    Args:
      input_fn: dataset input graph generation function
      params:  input function parameters
      host_id:  host identifier
    """

    iparams = {}
    iparams["batch_size"] = params["batch_size"] // FLAGS.num_cores
    iparams["dataset_num_shards"] = FLAGS.num_cores // FLAGS.tpu_cores_per_host

    def get_enqueue_ops_fn():
      """Generate the enqueue ops graph function."""

      iparams["dataset_index"] = host_id
      dataset = input_fn(iparams)
      iterator = dataset.make_initializable_iterator()
      self.dataset_initializer.append(iterator.initializer)

      def enqueue_ops_fn():
        """Generate the infeed enqueue ops graph."""

        per_host_sharded_inputs = []
        control_deps = []
        with tf.device(self.device_for_host(task=host_id)):
          for _ in range(FLAGS.tpu_cores_per_host):
            with tf.control_dependencies(control_deps):
              features, labels = iterator.get_next()
            self.feature_structure["features"] = features
            self.feature_structure["labels"] = labels
            flattened_inputs = data_nest.flatten(self.feature_structure)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          infeed = tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_fn)

      return enqueue_ops_fn

    with self.input_graph.as_default():
      with tf.device(self.device_for_host(host_id)):
        self.enqueue_ops.append(
            wrap_computation_in_while_loop(
                get_enqueue_ops_fn(),
                n=self.train_steps,
                parallel_iterations=1))

  def initialize(self, input_fn, model_fn, params):
    """Build graphs for the TPU device and the input pipelines.

    Args:
      input_fn: Dataset input graph generation function
      model_fn: Model definition function
      params:  Parameters to input and model functions
    """

    tf.logging.info("TrainRunner: initialize method")

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      i = 1
      while i < FLAGS.num_cores // FLAGS.tpu_cores_per_host:
        self.build_enqueue_ops(input_fn, params, i)
        i += 1
      # Build infeed sesssion
      self.input_sess = Session(
          self.cluster_resolver.get_master(),
          graph=self.input_graph,
          config=self.config)
      self.input_sess.run(self.dataset_initializer)
      # Run infeed session.run calls
      self.input_sess.run([self.enqueue_ops])

    self.build_enqueue_ops(input_fn, params, 0)

    def get_tpu_step(mparams):
      """Get the TPU graph generation function."""

      def tpu_step(loss):
        """Generate the TPU graph."""
        del loss
        values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
        unflattened_inputs = data_nest.pack_sequence_as(self.feature_structure,
                                                        values)
        features = unflattened_inputs["features"]
        labels = unflattened_inputs["labels"]
        estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN,
                                  mparams)
        loss, train_op = estimator_spec.loss, estimator_spec.train_op
        with tf.device(device_for_tpu_core()):
          with tf.control_dependencies([train_op]):
            return tf.identity(loss)

      return tpu_step

    tpu_step = get_tpu_step(params)

    @tpu_function.on_device_training_loop
    def tpu_loop():
      return tpu.repeat(self.iterations, tpu_step, [_INITIAL_LOSS])

    (self.loss,) = tpu.shard(
        tpu_loop,
        inputs=[],
        num_shards=FLAGS.num_cores,
        outputs_from_all_shards=False,
    )
    initializer = tf.global_variables_initializer()
    self.saver = tf.train.Saver()
    graph_io.write_graph(tf.Graph().as_graph_def(add_shapes=True),
                         FLAGS.model_dir, "graph.pbtxt")

    # Build tpu train model session and initialize graph
    self.sess = Session(
        self.cluster_resolver.get_master(),
        config=self.config)
    self.sess.run(initializer)

    # Complete infeed graph generation and session.run calls
    self.infeed_thread = threading.Thread(target=infeed_thread_fn, daemon=True)
    self.infeed_thread.start()

  def train(self, num_threads=2):
    """Run the Train steps on the TPU device.

    Args:
      num_threads: number of outstanding checkpointing threads

    """

    def checkpoint_thread_fn(saver, sess):
      saver.save(sess, FLAGS.model_dir + "/model.ckpt-%d" % (cur_step))

    cur_step = 0
    thread_id = 0
    checkpoint_threads = []
    tf.logging.info("TrainRunner: step %d", cur_step)
    for i in range(num_threads):
      checkpoint_threads.append(None)
    while cur_step < self.train_steps:
      start = time.time()
      tf.logging.info("TrainRunner: start next %d steps", self.iterations)
      cur_step += self.iterations
      loss = self.sess.run([self.loss])
      if checkpoint_threads[thread_id] is not None:
        checkpoint_threads[thread_id].join()
      checkpoint_threads[thread_id] = threading.Thread(
          target=checkpoint_thread_fn, args=(self.saver, self.sess), daemon=True)
      checkpoint_threads[thread_id].start()
      thread_id += 1
      if thread_id >= num_threads:
        thread_id = 0
      end = time.time()
      tf.logging.info(
          "TrainRunner: step {} loss {} step time {} sec {} examples/sec"
          .format(cur_step, loss, end - start,
                  self.iterations * FLAGS.train_batch_size / (end - start)))

    self.infeed_thread.join()
    for i in range(num_threads):
      if checkpoint_threads[i] is not None:
        checkpoint_threads[i].join()
        checkpoint_threads[i] = None

  def shutdown(self):
    if 'NO_TPU_INIT' not in os.environ:
      tf.logging.info("Shutting down TPU...")
      self.init_sess.run(self.tpu_shutdown)
