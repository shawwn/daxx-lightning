#source "${HOME}/bin/activate-tf1"
set -ex
i="$1"
shift 1
i=`printf '%04d' $i`
#data_dir=gs://mlperf-euw4/garden-imgnet/imagenet/combined
#model_dir=gs://mlsh_test/dev/assets/model_dir-resnet_model_dir_1558673290 
#tpu=TEST_TPU_1558673473.0 

#TPU_CORES="${TPU_CORES:-8}"
#TPU_INDEX="${TPU_INDEX:-68}"
#tpu="${TPU_NAME:-tpu-v3-${TPU_CORES}-euw4a-${TPU_INDEX}}"

TPU_CORES="${TPU_CORES:-8}"
TPU_INDEX="${TPU_INDEX:-0}"
tpu="${TPU_NAME:-tpu-v3-${TPU_CORES}-euw4a-${TPU_INDEX}}"

run_name="${RUN_NAME:-2020dec31_imagenet_0}"

data_dir="gs://mldata-euw4/datasets/imagenet"
model_dir="gs://ml-euw4/benchmarks/daxx/${run_name}/tf-1-15/v3-${TPU_CORES}/results-${i}"
export_dir="gs://ml-euw4/benchmarks/daxx/${run_name}/tf-1-15/v3-${TPU_CORES}/results-${i}"
save_graphs=True

export NOISY=1
export DEBUG=1
#
#exec python3 resnet_main.py --data_dir="$data_dir" \
#--output_summaries=True \
#--enable_lars=True \
#--eval_batch_size=1024 \
#--iterations_per_loop=1252 \
#--label_smoothing=0.1 \
#--lars_base_learning_rate=31.2 \
#--lars_epsilon=1e-05 \
#--lars_warmup_epochs=25 \
#--mode=in_memory_eval \
#--model_dir="$model_dir" \
#--export_dir="$export_dir" \
#--save_graphs="$save_graphs" \
#--num_cores="${TPU_CORES}" \
#--num_prefetch_threads=16 \
#--prefetch_depth_auto_tune=True \
#--resnet_depth=50 \
#--skip_host_call=True \
#--steps_per_eval=157 \
#--stop_threshold=0.759 \
#--tpu=$tpu \
#--train_batch_size=1024 \
#--train_steps=113854 \
#--use_async_checkpointing=True \
#--use_train_runner=True \
#--weight_decay=0.0002 \
#"$@"
#
#
#PYTHONPATH=.:/tmp/code_dir-resnet_code_1558420316/staging/models/rough/transformer/data_generators/:/tmp/code_dir-resnet_code_1558420316/staging/models/rough/:$PYTHONPATH python3 resnet_main.py --cache_decoded_image=True \
#--data_dir=gs://mlperf-euw4/garden-imgnet/imagenet/combined \
#--enable_lars=True \
#--eval_batch_size=4096 \
#--iterations_per_loop=1252 \
#--label_smoothing=0.1 \
#--mode=in_memory_eval \
#--model_dir=gs://mlsh_test/dev/assets/model_dir-resnet_model_dir_1558420316 \
#--num_cores=32 \
#--num_prefetch_threads=16 \
#--prefetch_depth_auto_tune=True \
#--resnet_depth=50 \
#--skip_host_call=True \
#--steps_per_eval=1252 \
#--stop_threshold=0.759 \
#--tpu=TEST_TPU_1558420331.47 \
#--train_batch_size=4096 \
#--train_steps=22536 \
#--use_async_checkpointing=True \
#--use_train_runner=True \
#--weight_decay=0.0002

#exec python3 resnet_main.py --data_dir="$data_dir" \
#--output_summaries=True \
#--distributed_group_size=1 \
#--enable_lars=True \
#--eval_batch_size=32768 \
#--iterations_per_loop=157 \
#--label_smoothing=0.1 \
#--lars_base_learning_rate=31.2 \
#--lars_epsilon=1e-05 \
#--lars_warmup_epochs=25 \
#--mode=in_memory_eval \
#--model_dir="$model_dir" \
#--export_dir="$export_dir" \
#--save_graphs="$save_graphs" \
#--num_cores="${TPU_CORES}" \
#--num_prefetch_threads=16 \
#--prefetch_depth_auto_tune=True \
#--resnet_depth=50 \
#--skip_host_call=True \
#--steps_per_eval=157 \
#--stop_threshold=0.759 \
#--tpu=$tpu \
#--train_batch_size=32768 \
#--train_steps=2983 \
#--use_async_checkpointing=True \
#--use_train_runner=True \
#--weight_decay=0.0001 \
#"$@"
#

#enable_lars=True
enable_lars=False
exec python3pdb resnet_main.py --data_dir="$data_dir" \
--output_summaries=True \
--distributed_group_size=1 \
--enable_lars=$enable_lars \
--eval_batch_size=512 \
--iterations_per_loop=5 \
--label_smoothing=0.1 \
--lars_base_learning_rate=31.2 \
--lars_epsilon=1e-05 \
--lars_warmup_epochs=25 \
--mode=in_memory_eval \
--model_dir="$model_dir" \
--export_dir="$export_dir" \
--save_graphs="$save_graphs" \
--num_cores="${TPU_CORES}" \
--num_prefetch_threads=16 \
--prefetch_depth_auto_tune=True \
--resnet_depth=50 \
--skip_host_call=True \
--steps_per_eval=5 \
--stop_threshold=0.759 \
--tpu=$tpu \
--train_batch_size=512 \
--train_steps=1 \
--use_async_checkpointing=True \
--use_train_runner=True \
--weight_decay=0.0001 \
"$@"
