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

import functools
import math
import os
import threading
import time
from absl import flags
from six.moves import queue as Queue
import tensorflow as tf
from . import tflex
#from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver
from .tflex import TPUClusterResolver
import tqdm

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io
from mlp_log import mlp_log

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7
_STOP = -1


def trainer_variables(self, index, variables=None):
  if variables is None:
    variables = self.fetch_vars
  else:
    variables = list(tflex.split_by_params(variables))
  return variables[index % len(variables)]

tflex.trainer_variables = trainer_variables

def trainer_slices(self, variables=None):
  if variables is None:
    variables = self.fetch_vars
  else:
    variables = list(tflex.split_by_params(variables))
  return len(variables)

tflex.trainer_slices = trainer_slices

# Decorator function for tpu computation func that was passed to tpu.rewrite()
# if there are embedded train and eval loops in this func, trace tools will
# generate step markers for each iteration.
def on_device_train_and_eval_loops(func):
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP")
  return func


def device_for_tpu_core(host_name, core=0):
  return host_name + "/device:TPU_REPLICATED_CORE:%d" % core


def device_for_host(host_name):
  return host_name + "/device:CPU:0"


def wrap_computation_in_while_loop(op_fn, n, host_name, parallel_iterations=1):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    ops = op_fn()
    if not isinstance(ops, list):
      ops = [ops]
    with tf.control_dependencies(ops):
      return i + 1

  with tf.device(device_for_host(host_name)):
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


def _profiler_callback(comment, session_id):
  if session_id is None:
    tf.logging.info("Profiling failed for %s", comment)
  else:
    tf.logging.info("Profiling succeeded for %s. Overview page url:", comment)

def dispatch(xs, thunk, *args, **kws):
  n = len(xs)
  results = [None] * n
  with tqdm.tqdm(total=n) as pbar:
    def fn(i):
      results[i] = thunk(i, *args, **kws)
    for thread in tflex.parallelize(list(range(n)), fn):
      thread.join()
      pbar.update(1)
    return results

def dispatch_sync(xs, thunk, *args, **kws):
  n = len(xs)
  results = [None] * n
  for i in tqdm.trange(n):
    results[i] = thunk(i, *args, **kws)
  return results

class VariableAccumulator(object):
  pass

from collections import defaultdict
import numpy as np

def variable_accumulator_new():
  self = VariableAccumulator()
  self.accum = {}
  self.accumcount = defaultdict(int)
  self.lock = threading.Lock()
  return self

tflex.variable_accumulator_new = variable_accumulator_new

def variable_accumulator_add(self, variable, value):
  if np.isnan(value).any():
    return False
  if np.isinf(value).any():
    return False
  if variable.name in self.accum:
    self.accum[variable.name] = self.accum[variable.name] + value
  else:
    self.accum[variable.name] = value
  self.accumcount[variable.name] += 1
  return True

tflex.variable_accumulator_add = variable_accumulator_add

from tensorflow.core.protobuf import config_pb2
tflex.read_deadline = 20000
tflex.write_deadline = 20000
tflex.reset_deadline = 20000

def trainer_slice_read(trainer, accumulator, variables):
  values = trainer.sess.run(tflex.cast_variables(variables, graph=trainer.sess.graph), options=config_pb2.RunOptions(timeout_in_ms=tflex.read_deadline))
  with accumulator.lock:
    for variable, value in zip(variables, values):
      tflex.variable_accumulator_add(accumulator, variable, value)

tflex.trainer_slice_read = trainer_slice_read

def trainer_assign_values(self, variables, values, timeout_in_ms=tflex.write_deadline):
  tflex.assign_values(variables, values, session=self.sess, timeout_in_ms=timeout_in_ms)
  #tflex.trainer_reset_variables(self, variables, timeout_in_ms=tflex.write_deadline)

tflex.trainer_assign_values = trainer_assign_values

def trainer_slice_write(trainer, accumulator, variables):
  values = []
  for variable in variables:
    with accumulator.lock:
      assert(variable.name in accumulator.accum)
      value = accumulator.accum[variable.name]
      n = accumulator.accumcount[variable.name]
    assert(n > 0)
    values.append(value / n)
  tflex.trainer_assign_values(trainer, variables, values)

tflex.trainer_slice_write = trainer_slice_write


tflex.update_trainers_read_timeout = 60
tflex.update_trainers_write_timeout = 60
tflex.update_trainers_write_threads = []

def update_trainers(trainers, i, sync_all=False):
  trainers = [x for x in trainers]
  if len(trainers) <= 0:
    return
  accumulator = tflex.variable_accumulator_new()
  threads = []
  for trainer in trainers:
    #if tflex.trainer_fresh(trainer):
    #  continue
    def thunk(trainer, accumulator, index):
      for variables in ([trainer.variables(index=index)] if not sync_all else tqdm.tqdm(list(tflex.split_by_params(trainer.global_vars)))):
        tflex.trainer_slice_read(trainer, accumulator, variables)
    thread = threading.Thread(target=thunk, args=(trainer,accumulator,i,))
    thread.start()
    threads.append(thread)
  start_time = time.time()
  for thread in threads:
    elapsed = (time.time() - start_time)
    waiting = tflex.update_trainers_read_timeout - elapsed
    if waiting > 0:
      thread.join(timeout=waiting)
  start_time = time.time()
  for thread in tflex.update_trainers_write_threads:
    elapsed = (time.time() - start_time)
    waiting = tflex.update_trainers_write_timeout - elapsed
    if waiting > 0:
      thread.join(timeout=waiting)
  tflex.update_trainers_write_threads = []
  for trainer in trainers:
    def thunk(trainer, accumulator, index):
      for variables in ([trainer.variables(index=index)] if not sync_all else tqdm.tqdm(list(tflex.split_by_params(trainer.global_vars)))):
        tflex.trainer_slice_write(trainer, accumulator, variables)
    thread = threading.Thread(target=thunk, args=(trainer,accumulator,i,))
    thread.start()
    tflex.update_trainers_write_threads.append(thread)

tflex.update_trainers = update_trainers

import json

class TrainAndEvalRunner(object):
  def __init__(self, *args, **kws):
    tf.logging.info("TrainAndEvalRunner: constructor")
    self._lock = threading.RLock()
    self._cur_step = 0
    tpus = FLAGS.tpu or FLAGS.master
    self.tpus = []
    for part in tpus.split(','):
      name, cores = part.split(':')
      cores = int(cores)
      with open('configs/tpu-v3-%d.json' % cores) as f:
        config = json.load(f)
      self.tpus.append([name, cores, config])
    self.shards = dispatch(self.tpus, lambda i: SwarmRunner(self, i, *self.tpus[i], *args, **kws))
    for i, shard in enumerate(self.shards):
      tf.logging.info("Checking shard %d", i)
      assert shard is not None

  def initialize(self, train_input_fn, eval_input_fn, model_fn, params, logger_fn=None):
    """Build graphs for the TPU device and the input pipelines.

    Args:
      train_input_fn: Dataset input graph generation function for training.
      eval_input_fn: Dataset input graph generation function for training.
      model_fn: Model definition function
      params:  Parameters to input and model functions
    """
    tf.logging.info("TrainAndEvalRunner: initialize()...")
    dispatch(self.tpus, lambda i: self.shards[i].initialize(train_input_fn, eval_input_fn, model_fn, params, logger_fn))

  def train_and_eval(self, output_summaries=False, enable_tracing=True):
    """Run the Train steps on the TPU device."""
    tf.logging.info("TrainAndEvalRunner: train_and_eval()...")
    threads = tflex.parallelize(list(range(len(self.tpus))),
                                lambda i: self.shards[i].train_and_eval(
                                  output_summaries=output_summaries,
                                  enable_tracing=enable_tracing))
    for i, thread in enumerate(threads):
      self.shards[i].thread = thread
    while True:
      time.sleep(1.0)
      trainers = [x for x in self.shards if x.thread.is_alive()]
      if len(trainers) <= 0:
        break
      n = len(trainers[0].fetch_vars)
      for j in tqdm.trange(n):
        tflex.update_trainers(trainers, j)

  def shutdown(self):
    tf.logging.info("TrainAndEvalRunner: shutdown()...")
    dispatch(self.tpus, lambda i: self.shards[i].shutdown())

  def claim(self, n):
    with self._lock:
      self._cur_step += n
      return self._cur_step

class SwarmRunner(object):
  """Remove init overheads in TPU Estimator via direct session.run calls."""

  def __init__(self, coordinator, index, tpu_name, num_cores, cfg, iterations, train_steps, eval_steps):
    tf.logging.info("SwarmRunner: constructor")
    iterations = cfg['iterations_per_loop']
    train_steps = cfg['train_steps']
    eval_steps = cfg['steps_per_eval']
    self.coordinator = coordinator
    self.index = index
    self.cfg = cfg
    self.tpu_name = tpu_name
    self.feature_structure = {}
    self.eval_feature_structure = {}
    self.loss = None
    self.eval_loss = None
    self.infeed_queue = []
    self.eval_infeed_queue = []
    self.enqueue_ops = []
    self.num_cores = num_cores
    self.num_hosts = num_cores // FLAGS.tpu_cores_per_host
    self.dequeue_ops = []
    self.queue = Queue.Queue()
    self.eval_enqueue_ops = []
    self.dataset_initializer = []
    self.eval_dataset_initializer = []
    self.iterations = iterations
    self.steps_per_epoch = FLAGS.num_train_images // self.cfg['train_batch_size']
    self.iterator = None
    self.sess = None
    self.saver = None
    self.checkpoint_thread = None
    self.input_sess = None
    self.eval_input_sess = None
    self.eval_output_sess = None
    self.log_sess = None
    self.infeed_thread = None
    self.train_eval_thread = None
    self.graph = tf.Graph()
    self.init_graph = tf.Graph()
    self.input_graph = tf.Graph()
    self.eval_input_graph = tf.Graph()
    self.eval_output_graph = tf.Graph()
    self.log_graph = tf.Graph()
    if train_steps % iterations != 0:
      train_steps = iterations * int(math.ceil(train_steps / iterations))
    self.train_steps = train_steps
    self.max_train_iterations = self.train_steps // iterations
    self.eval_steps = int(eval_steps)
    self.eval_batch_size = self.cfg['eval_batch_size']
    with self.init_graph.as_default():
      tpu_init = [tpu.initialize_system()]
      self.tpu_shutdown = tpu.shutdown_system()
    self.tpu_cluster_resolver = TPUClusterResolver(
        tpu_name,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    self.config = tf.ConfigProto(
        operation_timeout_in_ms=600 * 60 * 1000,
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.master = self.tpu_cluster_resolver.get_master()
    self.init_sess = tflex.Session(self.master, config=self.config, graph=self.init_graph)
    self.init_sess.run(tpu_init)

  def get_host(self, host_id):
    if self.master in ("", "local"):
      return "/replica:0/task:0"
    job_name = self.tpu_cluster_resolver.get_job_name() or "tpu_worker"
    result = "/job:%s/replica:0/task:%d" % (job_name, host_id)
    print(result)
    return result

  def build_enqueue_ops(self, input_fn, params, host_id, is_training=True):
    """Build enqueue operations for the input pipeline in a given host.

    Args:
      input_fn: dataset input graph generation function
      params:  input function parameters
      host_id:  host identifier
      is_training: boolean indicates if it is training
    """

    num_shards = len(self.coordinator.shards)
    index = self.index
    iparams = {}
    iparams["batch_size"] = params["batch_size"] // self.num_cores
    iparams["dataset_num_shards"] = self.num_hosts * num_shards

    def get_enqueue_ops_fn():
      """Generate the enqueue ops graph function."""

      iparams["dataset_index"] = num_shards * index + host_id
      with tf.device(device_for_host(self.get_host(host_id))):
        dataset = input_fn(iparams)
        if not is_training:
          dataset = dataset.cache()
          dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        if is_training:
          self.dataset_initializer.append(iterator.initializer)
        else:
          self.eval_dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Generate the infeed enqueue ops graph."""

          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(FLAGS.tpu_cores_per_host):
            with tf.control_dependencies(control_deps):
              features, labels = iterator.get_next()
            if is_training:
              self.feature_structure["features"] = features
              self.feature_structure["labels"] = labels
              flattened_inputs = data_nest.flatten(self.feature_structure)
            else:
              self.eval_feature_structure["features"] = features
              self.eval_feature_structure["labels"] = labels
              flattened_inputs = data_nest.flatten(self.eval_feature_structure)

            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          infeed = tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          if is_training:
            self.infeed_queue.append(infeed)
          else:
            self.eval_infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_fn)

      return enqueue_ops_fn

    if is_training:
      with self.input_graph.as_default():
        self.enqueue_ops.append(
            wrap_computation_in_while_loop(
                get_enqueue_ops_fn(),
                n=self.iterations,
                host_name=self.get_host(host_id),
                parallel_iterations=1))
    else:
      with self.eval_input_graph.as_default():
        self.eval_enqueue_ops.append(
            wrap_computation_in_while_loop(
                get_enqueue_ops_fn(),
                host_name=self.get_host(host_id),
                n=self.eval_steps,
                parallel_iterations=1))

  def get_tpu_step(self, mparams, model_fn, is_training=True):
    """Get the TPU graph generation function."""

    def tpu_step(loss):
      """Generate the TPU graph."""
      del loss
      if is_training:
        values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
        unflattened_inputs = data_nest.pack_sequence_as(self.feature_structure,
                                                        values)
      else:
        values = self.eval_infeed_queue[0].generate_dequeue_op(tpu_device=0)
        unflattened_inputs = data_nest.pack_sequence_as(
            self.eval_feature_structure, values)

      features = unflattened_inputs["features"]
      labels = unflattened_inputs["labels"]
      if is_training:
        estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN,
                                  mparams)
        loss, train_op = estimator_spec.loss, estimator_spec.train_op
        with tf.device(device_for_tpu_core(self.get_host(0))):
          with tf.control_dependencies([train_op]):
            return tf.identity(loss)
      else:
        estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.EVAL,
                                  mparams)
        loss = estimator_spec.loss
        self.eval_metrics = estimator_spec.eval_metrics
        self.eval_tensors = estimator_spec.eval_metrics[1]
        for _ in self.eval_tensors:
          self.dequeue_ops.append([])
        with tf.device(device_for_tpu_core(self.get_host(0))):
          outfeed_enqueue_ops = tpu.outfeed_enqueue_tuple(self.eval_tensors)
          with tf.control_dependencies([outfeed_enqueue_ops]):
            return tf.identity(loss)

    return tpu_step

  def launch_profiler(self):
    """Launches a profiling session to collect a trace from worker-0."""
    if result == profiler_client.PROFILED_IN_NEW_THREAD:
      tf.logging.info("A profiler session launched in a new thread.")
    else:
      tf.logging.info("profiler.collect() failed.")

  def initialize(self, train_input_fn, eval_input_fn, model_fn, params, logger_fn=None):
    """Build graphs for the TPU device and the input pipelines.

    Args:
      train_input_fn: Dataset input graph generation function for training.
      eval_input_fn: Dataset input graph generation function for training.
      model_fn: Model definition function
      params:  Parameters to input and model functions
    """

    tf.logging.info("TrainAndEvalRunner: initialize method")

    params = dict(params)
    params['batch_size'] = self.cfg['train_batch_size']

    self.build_enqueue_ops(train_input_fn, params, 0)

    # Start the build of the model
    tpu_step = self.get_tpu_step(params, model_fn)
    self.log_ops = None
    if logger_fn:
      with self.log_graph.as_default():
        self.log_step = tf.train.get_or_create_global_step()
        self.log_ops = logger_fn(self.log_step)
        self.log_initializer = tf.global_variables_initializer()
        self.log_step_in = tf.placeholder(tf.int64, [])
        self.log_step_init = tf.assign(self.log_step, self.log_step_in)

    @tpu_function.on_device_training_loop
    def train_loop():
      with tf.variable_scope("resnet", reuse=tf.AUTO_REUSE):
        return tpu.repeat(self.iterations, tpu_step, [_INITIAL_LOSS])

    self.train_loop = train_loop

    # Build tpu train model session and initialize graph
    self.initialize_eval(params, eval_input_fn, model_fn)

    # Build the infeed graph
    i = 1
    while i < self.num_hosts:
      self.build_enqueue_ops(train_input_fn, params, i)
      i = i + 1

    self.sess = tflex.Session(self.master, graph=self.graph, config=self.config)

    self.input_sess = tflex.Session(
        self.master, graph=self.input_graph, config=self.config)

    self.input_sess.run(self.dataset_initializer)

    self.eval_input_sess = tflex.Session(
        self.master, graph=self.eval_input_graph, config=self.config)

    self.eval_input_sess.run(self.eval_dataset_initializer)

    self.eval_output_sess = tflex.Session(
        self.master, graph=self.eval_output_graph, config=self.config)

    self.log_sess = tflex.Session(
      self.master, graph=self.log_graph, config=self.config)

    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())
      self.global_step = tf.train.get_or_create_global_step()
      self.global_step_in = tf.placeholder(tf.int64, [])
      self.global_step_init = tf.assign(self.global_step, self.global_step_in)
      self.sess.run(self.global_step_init, {self.global_step_in: 0})
      self.train_vars = tf.trainable_variables()
      self.fetch_vars = list(tflex.split_by_params(self.train_vars))
      self.saver = tf.train.Saver()
      n = len(self.fetch_vars)
      with tqdm.tqdm(total=n) as pbar:
        def thunk(i):
          variables = self.variables(i)
          values = self.sess.run(variables)
          tflex.assign_values(variables, values, session=self.sess)
        for thread in tflex.parallelize(list(range(n)), thunk):
          thread.join()
          pbar.update(1)

    with self.log_graph.as_default():
      self.log_sess.run(self.log_initializer)
      self.log_sess.run(self.log_step_init, {self.log_step_in: 0})

    def train_eval_thread_fn(sess, train_eval_op):
      sess.run([train_eval_op])

    # Start the just in time compilation of the model function
    tf.logging.info("Starting JIT compilation (sleeping for 60 sec)...")
    self.train_eval_thread = threading.Thread(
        target=train_eval_thread_fn, args=(self.sess, self.train_eval_op))
    self.train_eval_thread.start()

    # Sleep for JTC to finish
    time.sleep(60.0)

  def initialize_eval(self, params, eval_input_fn, model_fn):
    """Initialize eval."""

    self.eval_infeed_queue = []

    for i in range(0, self.num_hosts):
      self.build_enqueue_ops(
          eval_input_fn, params, host_id=i, is_training=False)

    eval_step = self.get_tpu_step(params, model_fn, is_training=False)

    @tpu_function.on_device_training_loop
    def eval_loop():
      with tf.variable_scope("resnet", reuse=tf.AUTO_REUSE):
        return tpu.repeat(int(self.eval_steps), eval_step, [_INITIAL_LOSS])

    def train_eval_step(loss):
      del loss
      with tf.control_dependencies(self.train_loop()):
        return eval_loop()

    @on_device_train_and_eval_loops
    def train_eval_loop():
      return tpu.repeat(self.max_train_iterations, train_eval_step,
                        [_INITIAL_LOSS])

    def create_dequeue_ops(host_id):
      """Create deque ops graph function."""
      dequeue_ops = []
      tensor_dtypes = []
      tensor_shapes = []
      for v in self.eval_tensors:
        dequeue_ops.append([])
        tensor_dtypes.append(v.dtype)
        tensor_shapes.append(v.shape)
      for i in range(FLAGS.tpu_cores_per_host):
        with tf.device(device_for_host(self.get_host(host_id))):
          outfeed_tensors = tpu.outfeed_dequeue_tuple(
              dtypes=tensor_dtypes, shapes=tensor_shapes, device_ordinal=i)
          for j, item in enumerate(outfeed_tensors):
            dequeue_ops[j].append(item)
      for j in range(len(outfeed_tensors)):
        dequeue_ops[j] = tf.concat(dequeue_ops[j], axis=0)
      return dequeue_ops

    with self.graph.as_default():
      with tf.variable_scope("resnet", reuse=True):
        (self.train_eval_op,) = tpu.shard(
            train_eval_loop,
            inputs=[],
            num_shards=self.num_cores,
            outputs_from_all_shards=False)

        graph_io.write_graph(tf.Graph().as_graph_def(add_shapes=True),
                             FLAGS.model_dir, "graph.pbtxt")

    with self.eval_output_graph.as_default():
      with tf.variable_scope("resnet", reuse=True):
        for i in range(0, self.num_hosts):
          host_dequeue_ops = create_dequeue_ops(i)
          for j, dequeue_tenor in enumerate(host_dequeue_ops):
            self.dequeue_ops[j].append(dequeue_tenor)

        for j, _ in enumerate(self.eval_tensors):
          self.dequeue_ops[j] = tf.concat(self.dequeue_ops[j], axis=0)

        with tf.device(device_for_host(self.get_host(0))):
          metrics = self.eval_metrics[0](*self.dequeue_ops)
        metric_update_ops = []
        metric_value_ops = {}
        for (k, v) in metrics.items():
          metric_update_ops.append(v[1])
          metric_value_ops[k] = v[0]
        self.metric_update_ops = metric_update_ops
        self.metric_value_ops = metric_value_ops

        self.metric_initializer = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

  def train_and_eval(self, output_summaries=False, enable_tracing=True):
    """Run the Train steps on the TPU device."""
    if output_summaries:
      output_dir = os.path.join(FLAGS.model_dir, "eval", self.tpu_name)
      tf.gfile.MakeDirs(output_dir)
      # Summary writer writes out eval metrics.
      summary_writer = tf.summary.FileWriter(output_dir)
      if FLAGS.save_graphs:
        summary_writer.add_graph(self.graph)
        summary_writer.add_graph(self.input_graph)
        summary_writer.add_graph(self.eval_input_graph)
        summary_writer.add_graph(self.eval_output_graph)

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      # Build infeed sesssion
      # Run infeed session.run calls
      tf.logging.info("Start infeed thread")
      for _ in range(self.train_steps // self.iterations):
        self.input_sess.run([self.enqueue_ops])
        self.eval_input_sess.run([self.eval_enqueue_ops])

    self.infeed_thread = threading.Thread(target=infeed_thread_fn)
    self.infeed_thread.start()

    # Gather trace for the first few steps.
    if enable_tracing:
      self.launch_profiler()

    self.cur_step = 0
    success = False
    while self.cur_step < self.train_steps or True:
      start = time.time()
      tf.logging.info("TrainAndEvalRunner: start next %d steps",
                      self.iterations)
      self.cur_step = self.coordinator.claim(self.iterations)
      self.sess.run(self.global_step_init, {self.global_step_in: self.cur_step})
      epoch = self.cur_step // self.steps_per_epoch - 1
      mlp_log.mlperf_print(
          "block_start", None, metadata={"first_epoch_num": epoch + 1,
                                         "epoch_count": 4})
      eval_results = self.eval(self.eval_steps)
      end = time.time()
      tf.logging.info(
          "TrainAndEvalRunner ({}): step {} step time {} sec {} examples/sec".format(
              self.tpu_name,
              self.cur_step, end - start,
              self.iterations * self.cfg['train_batch_size'] / (end - start)))
      # Run eval.
      # Write out summary to tensorboard.
      if output_summaries:
        with tf.Graph().as_default():
          summaries = []
          for metric in eval_results:
            summaries.append(
                tf.Summary.Value(tag=metric, simple_value=eval_results[metric]))
            tf_summary = tf.Summary(value=list(summaries))
            summary_writer.add_summary(tf_summary, self.cur_step)
      # MLPerf logging for eval results.
      mlp_log.mlperf_print(
          "eval_accuracy",
          float(eval_results["top_1_accuracy"]),
          metadata={"epoch_num": epoch + 1})

      mlp_log.mlperf_print(
          "block_stop", None, metadata={"first_epoch_num": epoch + 1})
      tf.logging.info("Eval results at step %d: %s", self.cur_step, eval_results)
      if eval_results["top_1_accuracy"] >= FLAGS.stop_threshold:
        success = True
        if FLAGS.export_dir is not None:
          def checkpoint_thread_fn(tpu_name, saver, sess, step):
            name = FLAGS.export_dir + "/model-%s.ckpt-%d" % (tpu_name, step)
            tf.logging.info("Saving model %d: %s", step, name)
            saver.save(sess, name)
          self.checkpoint_thread = threading.Thread(
            target=checkpoint_thread_fn, args=(self.tpu_name, self.saver, self.sess, self.cur_step))
          self.checkpoint_thread.start()
        mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
        break

      if enable_tracing and self.cur_step > self.train_steps // 4:
        self.launch_profiler()
        enable_tracing = False

    if not success:
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "abort"})

    mlp_log.mlperf_print("run_final", None)

    if output_summaries:
      summary_writer.close()

  def eval(self, num_steps):
    """Run the Eval steps on the TPU device.

    Args:
      num_steps: number of steps to run eval

    Returns:
      A dictionary of evaluation results.
    """

    self.eval_output_sess.run(self.metric_initializer)

    eval_results = {}
    tf.logging.info("Starting Eval on %d steps batch size %d" %
                    (num_steps, self.eval_batch_size))

    for _ in range(num_steps):
      _ = self.eval_output_sess.run(self.metric_update_ops)
    # Compute eval metrics
    session_out = self.eval_output_sess.run(self.metric_value_ops)
    for k, v in session_out.items():
      eval_results[k] = v
    self.log_sess.run(self.log_step_init, {self.log_step_in: self.cur_step})
    if self.log_ops:
      log_ops = {}
      for k, v in self.log_ops.items():
        if isinstance(v, int) or isinstance(v, float):
          eval_results[k] = v
        else:
          log_ops[k] = v
      session_out = self.log_sess.run(log_ops)
      for k, v in session_out.items():
        eval_results[k] = v
      for k, v in self.cfg.items():
        eval_results[k] = v
      if False:
        for i in tqdm.trange(len(self.fetch_vars)):
          variables = self.variables(i)
          self.sess.run(variables)
    return eval_results

  def shutdown(self):
    self.queue.put(_STOP)
    self.train_eval_thread.join()
    self.infeed_thread.join()
    if self.checkpoint_thread is not None:
      self.checkpoint_thread.join()
    self.sess.close()
    tf.logging.info("Shutting down TPU...")
    self.init_sess.run(self.tpu_shutdown)

  def variables(self, index):
    return tflex.trainer_variables(self, index)

