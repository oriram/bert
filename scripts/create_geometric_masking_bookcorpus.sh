python create_pretraining_data.py \
--input_file=../bookcorpus/split_books_*.train \
--output_dir=gs://ori-span-pretraining/data/geometric_bookcorpus_tfrecords \
--vocab_file=vocabs/bert-cased-vocab.txt \
--do_lower_case=False \
--do_whole_word_mask=False \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--num_processes=63 \
--masked_lm_prob=0.15 \
--dupe_factor=5 \
--geometric_masking_p=0.2 \
--max_span_length=10