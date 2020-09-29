# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

import os
import collections
import random
from multiprocessing import Pool

import numpy as np

import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_dir", None,
    "Output TF example directory.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_bool("recurring_span_masking", False, "Whether to mask recurring spans")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 80,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer("num_processes", 63, "Number of processes")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float("geometric_masking_p", 0.0, "The p for geometric distribution for span masking.")
flags.DEFINE_integer("max_span_length", 10, "Maximum span length to mask")

flags.DEFINE_bool("verbose", False, "verbose")

SPECIAL_TOKENS = {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[SPAN_MASK]'}

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0

    total_written, total_tokens_written = 0, 0
    for (inst_index, instance) in enumerate(instances):
        # if inst_index % 1000 == 0:
        #     tf.logging.info(f"written {inst_index}")
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_len = len(input_ids)
        input_mask = [1] * input_len
        assert input_len <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1
        total_tokens_written += input_len

        if FLAGS.verbose and inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info(f"Wrote {total_written} total instances, average length is {total_tokens_written // total_written}")


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_file, tokenizer, max_seq_length,
                              dupe_factor, masked_lm_prob,
                              max_predictions_per_seq, rng, length_dist=None, lengths=None):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) <doc> starts a new wiki document and </doc> ends it
    # (2) One sentence/paragraph per line.
    with tf.gfile.GFile(input_file, "r") as reader:
        expect_title = False
        for i, line in enumerate(reader):
            # if i % 1000 == 0:
            #     tf.logging.info(f"read {i}")

            line = tokenization.convert_to_unicode(line).strip()

            if (not line) or line.startswith("</doc"):
                continue

            if expect_title:
                expect_title = False
                continue

            # Starting a new document
            if line.startswith("<doc"):
                all_documents.append([])
                expect_title = True
                continue
            tokens = tokenizer.tokenize(line)
            if tokens:
                all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for dupe_idx in range(dupe_factor):
        for document_index in range(len(all_documents)):
            # if document_index % 100 == 0:
            #     tf.logging.info(f"processed {document_index}")
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, dupe_idx, length_dist, lengths))

    rng.shuffle(instances)
    return instances


def create_instance_from_context(segments, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, length_dist=None, lengths=None):
    tokens = ["[CLS]"]
    for segment in segments:
        tokens += segment
    tokens.append("[SEP]")

    if FLAGS.geometric_masking_p > 0:
        tokens, masked_lm_positions, masked_lm_labels = \
            create_geometric_masked_lm_predictions(tokens, masked_lm_prob, length_dist, lengths, [],
                                                   max_predictions_per_seq, vocab_words, rng)
    else:
        tokens, masked_lm_positions, masked_lm_labels = \
            create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

    return TrainingInstance(tokens=tokens,
                            masked_lm_positions=masked_lm_positions,
                            masked_lm_labels=masked_lm_labels)


def create_instances_from_document(
        all_documents, document_index, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
        dupe_idx, length_dist=None, lengths=None):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    instances = []
    current_chunk = []
    current_length = 0
    for i, segment in enumerate(document):
        segment_len = len(segment)

        if current_length + segment_len > max_num_tokens or i % len(document) == dupe_idx:
            if current_chunk:
                instance = create_instance_from_context(current_chunk, masked_lm_prob,
                                                        max_predictions_per_seq, vocab_words, rng, length_dist, lengths)
                instances.append(instance)

            current_chunk, current_length = [], 0
            if segment_len > max_num_tokens:
                # If this segment is too long, take the first max_num_tokens from this segment
                segment = segment[:max_num_tokens]
                # instance = create_instance_from_context([segment], masked_lm_prob,
                #                                         max_predictions_per_seq, vocab_words, rng)
                # instances.append(instance)
                # continue

        current_chunk.append(segment)
        current_length += len(segment)

    if current_chunk:
        instance = create_instance_from_context(current_chunk, masked_lm_prob,
                                                max_predictions_per_seq, vocab_words, rng, length_dist, lengths)
        instances.append(instance)

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def mask_tokens(output_tokens, start_index, end_index, vocab_words, rng):
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
        for idx in range(start_index, end_index+1):
            output_tokens[idx] = "[MASK]"
    else:
        # 10% of the time, replace with random word
        if rng.random() < 0.5:
            for idx in range(start_index, end_index + 1):
                output_tokens[idx] = vocab_words[rng.randint(0, len(vocab_words) - 1)]


def create_geometric_masked_lm_predictions(tokens, masked_lm_prob, length_dist, lengths, already_masked,
                                           max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for geometric objective."""
    output_tokens = list(tokens)

    candidates_for_start, candidates_for_end, candidates_for_mask = \
        [False] * len(output_tokens), [False] * len(output_tokens), [False] * len(output_tokens)
    for i, token in enumerate(output_tokens):
        # if attention_mask[i] and token not in SPECIAL_TOKENS:
        if token not in SPECIAL_TOKENS:
            candidates_for_mask[i] = True
            candidates_for_start[i] = (not token.startswith("##"))
            candidates_for_end[i] = (
                    i == len(output_tokens) - 1 or not output_tokens[i + 1].startswith("##"))
    if sum(candidates_for_start) < 0.5 * len(output_tokens):
        # logger.info("An example with too many OOV words, skipping on geometric masking")
        candidates_for_start = candidates_for_mask
        candidates_for_end = candidates_for_mask

    num_predictions = len(already_masked)
    num_tokens_to_mask = int(masked_lm_prob * sum(candidates_for_mask))
    num_tokens_to_mask = min(max_predictions_per_seq - num_predictions, num_tokens_to_mask)

    len_dist = list(length_dist)
    masked_lms = []
    while num_predictions < num_tokens_to_mask:
        span_len_idx = np.random.choice(range(len(len_dist)), p=len_dist)
        span_len = lengths[span_len_idx]
        if num_predictions + span_len <= num_tokens_to_mask:
            num_attempts = 0
            max_attempts = 30
            while num_attempts < max_attempts:
                start_idx = np.random.randint(len(output_tokens) - span_len + 1)
                end_idx = start_idx + span_len - 1
                if candidates_for_start[start_idx] and candidates_for_end[end_idx] \
                        and all(candidates_for_mask[j] for j in range(start_idx, end_idx + 1)):
                    for j in range(start_idx, end_idx + 1):
                        candidates_for_start[j] = False
                        candidates_for_end[j] = False
                        candidates_for_mask[j] = False
                        masked_lms.append(MaskedLmInstance(index=j, label=output_tokens[j]))

                        num_predictions += 1
                    mask_tokens(output_tokens, start_idx, end_idx, vocab_words, rng)
                    break
                num_attempts += 1
            if num_attempts == max_attempts:
                # print(f"Maximum attempts for span length {span_len}. Skipping geometric masking")
                candidates_for_start = candidates_for_mask
                candidates_for_end = candidates_for_mask

    assert len(masked_lms) <= num_tokens_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    assert 0 < len(tokens) <= 512, str(len(tokens))

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            mask_tokens(output_tokens, index, index, vocab_words, rng)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def process_file(input_file, output_file, tokenizer, rng, length_dist=None, lengths=None):
    tf.logging.info(f"*** Started processing file {input_file} ***")

    instances = create_training_instances(
        input_file, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng, length_dist, lengths)

    tf.logging.info(f"*** Finished processing file {input_file}, writing to output file {output_file} ***")

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, [output_file])

    tf.logging.info(f"*** Finished writing to output file {output_file} ***")


def get_output_file(input_file, output_dir):
    path = os.path.normpath(input_file)
    split = path.split(os.sep)
    dir_and_file = split[-2:]
    return os.path.join(output_dir, '_'.join(dir_and_file) + '.tfrecord')


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    assert not tf.io.gfile.exists(FLAGS.output_dir), "Output directory already exists"
    tf.io.gfile.mkdir(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info(f"*** Reading from {len(input_files)} files ***")

    rng = random.Random(FLAGS.random_seed)

    length_dist, lengths = None, None
    if FLAGS.geometric_masking_p > 0:
        p = FLAGS.geometric_masking_p
        lower, upper = 1, FLAGS.max_span_length
        lengths = list(range(lower, upper + 1))
        length_dist = [p * (1 - p) ** (i - lower) for i in range(lower, upper + 1)] if p >= 0 else None
        length_dist = [x / (sum(length_dist)) for x in length_dist]

    params = [(file, get_output_file(file, FLAGS.output_dir), tokenizer, rng, length_dist, lengths)
              for file in input_files]
    with Pool(FLAGS.num_processes if FLAGS.num_processes else None) as p:
        p.starmap(process_file, params)

    # instances = create_training_instances(
    #     input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
    #     FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
    #     rng)
    #
    # output_files = FLAGS.output_file.split(",")
    # tf.logging.info("*** Writing to output files ***")
    # for output_file in output_files:
    #     tf.logging.info("  %s", output_file)
    #
    # write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
    #                                 FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
