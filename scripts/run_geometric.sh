python run_pretraining.py \
--bert_config_file=configs/bert-base-cased-config.json \
--input_file=gs://span-pretraining/data/geometric_tfrecords/* \
--output_dir=gs://span-pretraining/model_outputs/geometric \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--do_train \
--train_batch_size=256 \
--learning_rate=1e-4 \
--num_train_steps=1000000 \
--num_warmup_steps=10000 \
--save_checkpoints_steps=10000 \
--keep_checkpoint_max=100 \
--use_tpu \
--num_tpu_cores=8 \
--tpu_name=node-v3-8-tf2
