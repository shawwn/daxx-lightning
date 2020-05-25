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
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.training import moving_averages


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

FLAGS = flags.FLAGS


def cross_replica_average(inputs, num_shards, distributed_group_size):
  """Calculates the average value of inputs tensor across TPU replicas."""
  group_assignment = None
  if num_shards is not None and distributed_group_size != num_shards:
    group_size = distributed_group_size
    group_assignment = []
    for g in range(num_shards // group_size):
      replica_ids = [g * group_size + i for i in range(group_size)]
      group_assignment.append(replica_ids)

  outputs = tpu_ops.cross_replica_sum(inputs, group_assignment) / tf.cast(
      distributed_group_size, inputs.dtype)
  return outputs


def distributed_batch_norm(inputs,
                           decay=BATCH_NORM_DECAY,
                           epsilon=BATCH_NORM_EPSILON,
                           is_training=True,
                           gamma_initializer=None,
                           num_shards=None,
                           distributed_group_size=2,
                           scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
    epsilon: Small float added to variance to avoid dividing by zero.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    gamma_initializer:  Initializers for gamma.
    num_shards: Number of shards that participate in the global reduction.
      Default is set to None, that will skip the cross replica sum in and
      normalize across local examples only.
    distributed_group_size: Number of replicas to normalize across in the
      distributed batch normalization.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.
  """

  with tf.variable_scope(scope, 'batch_normalization', [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta = tf.get_variable(
        'beta',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)
    gamma = tf.get_variable(
        'gamma',
        dtype=tf.float32,
        shape=params_shape,
        initializer=gamma_initializer,
        trainable=True)
    # Disable partition setting for moving_mean and moving_variance
    # as assign_moving_average op below doesn't support partitioned variable.
    scope = tf.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf.get_variable(
        'moving_mean',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    moving_variance = tf.get_variable(
        'moving_variance',
        shape=params_shape,
        initializer=tf.ones_initializer(),
        trainable=False)
    # Restore scope's partitioner setting.
    scope.set_partitioner(partitioner)

    # Add cross replica sum to do subset mean and variance calculation
    # First compute mean and variance
    if is_training:
      # Execute a distributed batch normalization
      axis = 3
      inputs_dtype = inputs.dtype
      inputs = tf.cast(inputs, tf.float32)
      ndims = len(inputs_shape)
      reduction_axes = [i for i in range(ndims) if i != axis]
      counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
          inputs, reduction_axes, keep_dims=False)
      mean_ss = cross_replica_average(mean_ss, num_shards,
                                      distributed_group_size)
      variance_ss = cross_replica_average(variance_ss, num_shards,
                                          distributed_group_size)
      mean, variance = tf.nn.normalize_moments(
          counts, mean_ss, variance_ss, shift=None)
      outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                          epsilon)
      outputs = tf.cast(outputs, inputs_dtype)
    else:
      outputs, mean, variance = tf.nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=epsilon,
          is_training=False,
          data_format='NHWC')

    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean,
          tf.cast(mean, moving_mean.dtype),
          decay,
          zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance,
          tf.cast(variance, moving_variance.dtype),
          decay,
          zero_debias=False)
      tf.add_to_collection('update_ops', update_moving_mean)
      tf.add_to_collection('update_ops', update_moving_variance)

    outputs.set_shape(inputs_shape)
    return outputs

def spectral_norm(inputs, epsilon=1e-12, singular_value="left"):
  """Performs Spectral Normalization on a weight tensor.

  Details of why this is helpful for GAN's can be found in "Spectral
  Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
  [https://arxiv.org/abs/1802.05957].

  Args:
    inputs: The weight tensor to normalize.
    epsilon: Epsilon for L2 normalization.
    singular_value: Which first singular value to store (left or right). Use
      "auto" to automatically choose the one that has fewer dimensions.

  Returns:
    The normalized weight tensor.
  """
  if len(inputs.shape) < 2:
    raise ValueError(
        "Spectral norm can only be applied to multi-dimensional tensors")

  # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
  # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
  # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
  # layers that put output channels as last dimension. This implies that w
  # here is equivalent to w.T in the paper.
  w = tf.reshape(inputs, (-1, inputs.shape[-1]))

  # Choose whether to persist the first left or first right singular vector.
  # As the underlying matrix is PSD, this should be equivalent, but in practice
  # the shape of the persisted vector is different. Here one can choose whether
  # to maintain the left or right one, or pick the one which has the smaller
  # dimension. We use the same variable for the singular vector if we switch
  # from normal weights to EMA weights.
  var_name = inputs.name.replace("/ExponentialMovingAverage", "").split("/")[-1]
  var_name = var_name.split(":")[0] + "/u_var"
  if singular_value == "auto":
    singular_value = "left" if w.shape[0] <= w.shape[1] else "right"
  u_shape = (w.shape[0], 1) if singular_value == "left" else (1, w.shape[-1])
  u_var = tf.get_variable(
      var_name,
      shape=u_shape,
      dtype=w.dtype,
      initializer=tf.random_normal_initializer(),
      collections=[tf.GraphKeys.LOCAL_VARIABLES],
      trainable=False)
  u = u_var

  # Use power iteration method to approximate the spectral norm.
  # The authors suggest that one round of power iteration was sufficient in the
  # actual experiment to achieve satisfactory performance.
  power_iteration_rounds = 1
  for _ in range(power_iteration_rounds):
    if singular_value == "left":
      # `v` approximates the first right singular vector of matrix `w`.
      v = tf.math.l2_normalize(
          tf.matmul(tf.transpose(w), u), axis=None, epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(w, v), axis=None, epsilon=epsilon)
    else:
      v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True),
                               epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(v, w), epsilon=epsilon)

  # Update the approximation.
  with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
    u = tf.identity(u)

  # The authors of SN-GAN chose to stop gradient propagating through u and v
  # and we maintain that option.
  u = tf.stop_gradient(u)
  v = tf.stop_gradient(v)

  if singular_value == "left":
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
  else:
    norm_value = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
  norm_value.shape.assert_is_fully_defined()
  norm_value.shape.assert_is_compatible_with([1, 1])

  w_normalized = w / norm_value

  # Deflate normalized weights to match the unnormalized tensor.
  w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)
  return w_tensor_normalized, norm_value[0][0]


def weight_initializer(initializer='normal', stddev=0.02):
  """Returns the initializer for the given name.

  Args:
    initializer: Name of the initalizer. Use one in consts.INITIALIZERS.
    stddev: Standard deviation passed to initalizer.

  Returns:
    Initializer from `tf.initializers`.
  """
  if initializer == 'normal':
    return tf.initializers.random_normal(stddev=stddev)
  if initializer == 'truncated':
    return tf.initializers.truncated_normal(stddev=stddev)
  if initializer == 'orthogonal':
    return tf.initializers.orthogonal()
  raise ValueError("Unknown weight initializer {}.".format(initializer))

def linear(inputs, output_size, scope=None, stddev=0.02, bias_start=0.0,
           use_sn=False, use_bias=True, dtype=tf.float32):
  """Linear layer without the non-linear activation applied."""
  shape = inputs.get_shape().as_list()
  with tf.variable_scope(scope or "linear"):
    kernel = tf.get_variable(
        "kernel",
        [shape[1], output_size],
        dtype=dtype,
        initializer=weight_initializer(stddev=stddev))
    # kernel = graph_spectral_norm(kernel)
    if use_sn:
      kernel, norm = spectral_norm(kernel)
    outputs = tf.matmul(inputs, kernel)
    if use_bias:
      bias = tf.get_variable(
          "bias",
          [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start))
      outputs += bias
    return outputs


def evonorm_s0(inputs,
              training=True,
              nonlinearity=True,
              name="batch_normalization",
              #name="evonorm_s0",
              scale=True,
              center=True,
              gamma_initializer=None,
              scope=None):
  with tf.variable_scope(scope, name, [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_dtype = inputs.dtype
    num_channels = inputs.shape[-1].value
    if num_channels is None:
      raise ValueError("`C` dimension must be known but is None")

    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError("Inputs %s has undefined rank" % inputs.name)
    elif inputs_rank not in [4]: #[2, 4]:
      raise ValueError(
          "Inputs %s has unsupported rank."
          " Expected 4 but got %d" % (inputs.name, inputs_rank))
          #" Expected 2 or 4 but got %d" % (inputs.name, inputs_rank))

    # if inputs_rank == 2:
    #   new_shape = [-1, 1, 1, num_channels]
    #   if data_format == "NCHW":
    #     new_shape = [-1, num_channels, 1, 1]
    #   inputs = tf.reshape(inputs, new_shape)

    inputs = tf.cast(inputs, tf.float32)
    if nonlinearity:
      groups = num_channels // 2
      #groups = 8
      assert num_channels % groups == 0
      assert groups <= num_channels
      assert groups > 0
      v = trainable_variable_ones(shape=[1, 1, 1, num_channels])
      num = inputs * tf.sigmoid(v * inputs)
      outputs = num / group_std(inputs, groups=groups)
    else:
      outputs = inputs

    if scale:
      if gamma_initializer is None:
        gamma_initializer = tf.ones_initializer()
      gamma = tf.get_variable(
        "gamma",
        [1, 1, 1, num_channels],
        dtype=tf.float32,
        initializer=gamma_initializer,
        trainable=True)
      outputs *= gamma

    if center:
      beta = tf.get_variable(
        "beta",
        [1, 1, 1, num_channels],
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)
      outputs += beta

    outputs = tf.cast(outputs, inputs_dtype)
  return outputs

def evonorm_q0(inputs, z=None, z_dim=128,
               training=True,
               nonlinearity=True,
               #name="batch_normalization",
               name="evonorm_q0",
               scale=True,
               center=True,
               scope=None,
               num_hidden=128,
               use_sn=True):
  with tf.variable_scope(scope, name, [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_dtype = inputs.dtype
    num_channels = inputs.shape[-1].value
    if num_channels is None:
      raise ValueError("`C` dimension must be known but is None")

    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError("Inputs %s has undefined rank" % inputs.name)
    elif inputs_rank not in [4]: #[2, 4]:
      raise ValueError(
        "Inputs %s has unsupported rank."
        " Expected 4 but got %d" % (inputs.name, inputs_rank))
      #" Expected 2 or 4 but got %d" % (inputs.name, inputs_rank))

    # if inputs_rank == 2:
    #   new_shape = [-1, 1, 1, num_channels]
    #   if data_format == "NCHW":
    #     new_shape = [-1, num_channels, 1, 1]
    #   inputs = tf.reshape(inputs, new_shape)

    inputs = tf.cast(inputs, tf.float32)
    outputs = evonorm_s0(inputs, scale=False, center=False)
    num_channels = inputs.shape[-1].value

    with tf.variable_scope("sbn", values=[inputs]):
      if z is None:
        z = tf.random.uniform(minval=-1.0, maxval=1.0, dtype=tf.float32, shape=[inputs_shape[0], z_dim])
      h = z
      if num_hidden > 0:
        h = linear(h, num_hidden, scope="hidden", use_sn=use_sn)
        h = tf.nn.relu(h)
      if scale:
        gamma = linear(h, num_channels, scope="gamma", bias_start=1.0,
                       use_sn=use_sn)
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels])
        outputs *= gamma
      if center:
        beta = linear(h, num_channels, scope="beta", use_sn=use_sn)
        beta = tf.reshape(beta, [-1, 1, 1, num_channels])
        outputs += beta
      outputs = tf.cast(outputs, inputs_dtype)
      return outputs


# def instance_std(x, eps=1e-5):
#   _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
#   return tf.sqrt(var + eps)
#
# def group_std(x, groups=32, eps=1e-5):
#   N, H, W, C = x.shape
#   x = tf.reshape(x, [N, H, W, groups, C // groups])
#   _, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
#   std = tf.sqrt(var + eps)
#   std = tf.broadcast_to(std, x.shape)
#   return tf.reshape(std, [N, H, W, C])

DEFAULT_EPSILON_VALUE = 1e-5

def instance_std(x, eps=DEFAULT_EPSILON_VALUE):
  _, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
  return tf.sqrt(var + eps)

def group_std(inputs, groups=32, eps=DEFAULT_EPSILON_VALUE, axis=-1):
  groups = min(inputs.shape[axis], groups)

  input_shape = tf.shape(inputs)
  group_shape = [input_shape[i] for i in range(4)]
  group_shape[axis] = input_shape[axis] // groups
  group_shape.insert(axis, groups)
  group_shape = tf.stack(group_shape)
  grouped_inputs = tf.reshape(inputs, group_shape)
  _, var = tf.nn.moments(grouped_inputs, [1, 2, 4], keepdims=True)

  std = tf.sqrt(var + eps)
  std = tf.broadcast_to(std, tf.shape(grouped_inputs))
  return tf.reshape(std, input_shape)

def trainable_variable_ones(shape, name="v", initializer=None):
  if initializer is None:
    initializer = tf.ones_initializer()
  return tf.get_variable(name, shape=shape, initializer=initializer, dtype=tf.float32, trainable=True)

def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  if FLAGS.distributed_group_size > 1:
    assert data_format == 'channels_last'
    tf.logging.info('Using batchnorm distributed_batch_norm')
    inputs = distributed_batch_norm(
        inputs=inputs,
        decay=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        is_training=is_training,
        gamma_initializer=gamma_initializer,
        num_shards=FLAGS.num_cores,
        distributed_group_size=FLAGS.distributed_group_size)
  elif FLAGS.distributed_group_size == 1:
    tf.logging.info('Using batchnorm tf.layers.batch_normalization')
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer)
  elif FLAGS.distributed_group_size == 0:
    tf.logging.info('Using batchnorm evonorm_s0')
    assert data_format == 'channels_last'
    inputs = evonorm_s0(
        inputs=inputs,
        center=True,
        scale=True,
        training=is_training,
        #gamma_initializer=None)
        gamma_initializer=gamma_initializer)
    relu = False
  else:
    tf.logging.info('Using batchnorm evonorm_q0')
    assert data_format == 'channels_last'
    inputs = evonorm_q0(
      inputs=inputs,
      training=is_training)
    relu = False

  if relu:
    inputs = tf.nn.relu(inputs)

  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_first'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_first'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  outputs = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

  return outputs


def residual_block(inputs, filters, is_training, strides,
                   use_projection=False, data_format='channels_first'):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False,
                               data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                           data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs, filters, is_training, strides,
                     use_projection=False, data_format='channels_first'):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False,
                               data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                           data_format=data_format)

  output = tf.nn.relu(inputs + shortcut)

  return output


def block_group(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format='channels_first'):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(inputs, filters, is_training, strides,
                    use_projection=True, data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, 1,
                      data_format=data_format)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, num_classes,
                        data_format='channels_first'):
  """Generator for ResNet v1 models.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """
  def model(inputs, is_training):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

    pooled_inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(pooled_inputs, 'initial_max_pool')

    inputs = block_group(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_group1',
        data_format=data_format)
    inputs = block_group(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_group2',
        data_format=data_format)
    inputs = block_group(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_group3',
        data_format=data_format)
    inputs = block_group(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_group4',
        data_format=data_format)

    # The activation is 7x7 so this is a global average pool.
    # TODO(huangyp): reduce_mean will be faster.
    pool_size = (inputs.shape[1], inputs.shape[2])
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=pool_size, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(
        inputs, [-1, 2048 if block_fn is bottleneck_block else 512])
    inputs = tf.layers.dense(
        inputs=inputs,
        units=num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=.01))
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  model.default_image_size = 224
  return model


def resnet_v1(resnet_depth, num_classes, data_format='channels_first'):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return resnet_v1_generator(
      params['block'], params['layers'], num_classes, data_format)
