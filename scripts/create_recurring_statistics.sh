python create_pretraining_data.py \
--input_file=../wiki/enwiki/*/*wiki*,../bookcorpus/split_books_*.train \
--output_dir=gs://span-pretraining/data/no_directory \
--vocab_file=vocabs/bert-cased-vocab.txt \
--only_write_statistics=True \
--ngrams_file=gs://ori-span-pretraining/data/recurring_ngrams_stats.txt \
--do_lower_case=False \
--do_whole_word_mask=False \
--max_seq_length=512 \
--max_predictions_per_seq=60 \
--num_processes=63 \
--masked_lm_prob=0.15 \
--dupe_factor=1 \
--max_span_length=10 \
--recurring_span_masking=True \
--max_recurring_predictions_per_seq=30