source "${HOME}/bin/activate-tf1"
set -ex
i="$1"
shift 1
i=`printf '%04d' $i`
#data_dir=gs://mlperf-euw4/garden-imgnet/imagenet/combined
#model_dir=gs://mlsh_test/dev/assets/model_dir-resnet_model_dir_1558673290 
#tpu=TEST_TPU_1558673473.0 
TPU_CORES="${TPU_CORES:-8}"
data_dir="gs://danbooru-euw4a/data/imagenet/out"
model_dir="gs://danbooru-euw4a/mlperf/benchmarks/imagenet/tf-1-14-1-dev20190518/v3-${TPU_CORES}/results-${i}"
export_dir="gs://danbooru-euw4a/mlperf/benchmarks/imagenet/tf-1-14-1-dev20190518/v3-${TPU_CORES}/results-${i}"
TPU_INDEX="${TPU_INDEX:-68}"
#tpu="${TPU_NAME:-tpu-v3-${TPU_CORES}-euw4a-${TPU_INDEX}}"
tpu="${TPU_NAME:-tpu-euw4a-${TPU_INDEX}}"

exec python3 resnet_main.py --data_dir="$data_dir" \
--output_summaries=True \
--distributed_group_size=1 \
--enable_lars=True \
--eval_batch_size=32768 \
--iterations_per_loop=157 \
--label_smoothing=0.1 \
--lars_base_learning_rate=31.2 \
--lars_epsilon=1e-05 \
--lars_warmup_epochs=25 \
--mode=in_memory_eval \
--model_dir="$model_dir" \
--export_dir="$export_dir" \
--num_cores=512 \
--num_prefetch_threads=16 \
--prefetch_depth_auto_tune=True \
--resnet_depth=50 \
--skip_host_call=True \
--steps_per_eval=157 \
--stop_threshold=0.759 \
--tpu=$tpu \
--train_batch_size=32768 \
--train_steps=2983 \
--use_async_checkpointing=True \
--use_train_runner=True \
--weight_decay=0.0001 \
"$@"

