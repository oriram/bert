from collections import namedtuple, defaultdict
import numpy as np
import tensorflow as tf

from tokenization import SPECIAL_TOKENS

STOPWORDS = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

MaskedLmInstance = namedtuple("MaskedLmInstance",
                              ["index", "label"])
MaskedSpanInstance = namedtuple("MaskedSpanInstance",
                              ["index", "begin_label", "end_label"])


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


def create_geometric_masked_lm_predictions(tokens, masked_lm_prob, length_dist, lengths, num_already_masked,
                                           max_predictions_per_seq, vocab_words, rng, input_mask=None):
    """Creates the predictions for geometric objective."""
    output_tokens = list(tokens)

    candidates_for_start, candidates_for_end, candidates_for_mask = \
        [False] * len(output_tokens), [False] * len(output_tokens), [False] * len(output_tokens)
    for i, token in enumerate(output_tokens):
        if (input_mask is None or input_mask[i]) and token not in SPECIAL_TOKENS:
            candidates_for_mask[i] = True
            candidates_for_start[i] = (not token.startswith("##"))
            candidates_for_end[i] = (
                    i == len(output_tokens) - 1 or not output_tokens[i + 1].startswith("##"))
    if sum(candidates_for_start) < 0.5 * len(output_tokens):
        # logger.info("An example with too many OOV words, skipping on geometric masking")
        candidates_for_start = candidates_for_mask
        candidates_for_end = candidates_for_mask

    num_predictions = 0
    num_tokens_to_mask = int(masked_lm_prob * sum(candidates_for_mask))
    num_tokens_to_mask = min(max_predictions_per_seq - num_already_masked, num_tokens_to_mask)

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

    return output_tokens, masked_lm_positions, masked_lm_labels


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, num_already_masked,
                                 vocab_words, rng, do_whole_word_mask=False):
    """Creates the predictions for the masked LM objective."""

    assert 0 < len(tokens) <= 512, str(len(tokens))

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in SPECIAL_TOKENS:
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
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
    num_to_predict -= num_already_masked

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

    return output_tokens, masked_lm_positions, masked_lm_labels


def validate_ngram(tokens, start_index, length):
    # If the vocab at the beginning of the span is a part-of-word (##), we don't want to consider this span.
    # if vocab_word_piece[token_ids[start_index]]:
    if tokens[start_index].startswith("##"):
        return False, None

    # If the token *after* this considered span is a part-of-word (##), we don't want to consider this span.
    if tokens[start_index + length].startswith("##"):
        return False, None

    substring_tokens = tokens[start_index:start_index + length]

    # We filter out n-grams that are all stopwords (e.g. "in the", "with my", ...)
    if all([token.lower() in STOPWORDS for token in substring_tokens]):
        return False, None

    if not all([token.isalnum() or token.startswith("##") for token in substring_tokens]):
        # TODO: Treat spans like " An American in Paris "       (that start and end in " symbol) differently
        # TODO: Treat spans like H&O differently
        # if vocab_alnum[token_ids[start_index]] ==  and vocab_alnum[token_ids[start_index + length - 1]]:
        #   return True, substring_token_ids
        return False, None

    return True, substring_tokens


def find_recurring_ngrams(tokens, max_span_length):
    num_tokens = len(tokens)
    all_valid_substrings = defaultdict(list)
    for l in range(1, max_span_length+1):
        for start_index in range(num_tokens-l):
            is_valid, substring_token_ids = validate_ngram(tokens, start_index=start_index, length=l)
            if is_valid:
                all_valid_substrings[str(substring_token_ids)].append((start_index, start_index+l-1))

    ngrams = [(eval(ngram), spans) for ngram, spans in all_valid_substrings.items() if len(spans) > 1]
    # Decoding the spans back to string (i.e. ["United", "States"] --> "United States" )
    ngrams = [(' '.join(ngram), len(ngram), spans) for ngram, spans in ngrams]

    # We remove any n-gram occurrence that is a substring of another recurring n-gram
    # Note that other occurrences of this n-gram may be valid though (assuming there at least 2)
    filtered_ngrams = []
    for ngram, length, spans in ngrams:
        spans_to_keep = [True] * len(spans)
        for other_ngram, _, other_spans in ngrams:
            if ngram != other_ngram and ngram in other_ngram:
                for i, span in enumerate(spans):
                    for other_span in other_spans:
                        if span[0] >= other_span[0] and span[1] <= other_span[1]:
                            spans_to_keep[i] = False

        # If we have more than one occurrence after the filtering:
        if sum(spans_to_keep) > 1:
            new_spans = [span for i, span in enumerate(spans) if spans_to_keep[i]]
            filtered_ngrams.append(new_spans)
    return filtered_ngrams


def create_recurring_span_mask_predictions(tokens, max_recurring_predictions, max_span_length, masked_lm_prob):
    masked_spans = []
    num_predictions = 0
    input_mask = [1] * len(tokens)
    new_tokens = list(tokens)

    already_masked_tokens = [False] * len(new_tokens)
    span_label_tokens = [False] * len(new_tokens)

    def _iterate_span_indices(span):
        return range(span[0], span[1] + 1)

    num_to_predict = min(max_recurring_predictions,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    spans = find_recurring_ngrams(new_tokens, max_span_length)
    for idx in np.random.permutation(range(len(spans))):
        identical_spans = spans[idx]
        # self._assert_and_return_identical(token_ids, identical_spans)
        num_occurrences = len(identical_spans)

        # Choosing which span to leave unmasked:
        unmasked_span_idx = np.random.randint(num_occurrences)
        unmasked_span = identical_spans[unmasked_span_idx]
        if any([already_masked_tokens[i] for i in _iterate_span_indices(unmasked_span)]):
            # The same token can't be both masked for one pair and unmasked for another pair
            continue

        unmasked_span_beginning, unmasked_span_ending = unmasked_span
        for i, span in enumerate(identical_spans):
            if num_predictions >= num_to_predict:
                # logger.warning(f"Already masked {self.max_predictions} spans.")
                break

            if any([already_masked_tokens[j] for j in _iterate_span_indices(unmasked_span)]):
                break

            if i != unmasked_span_idx:
                if any([already_masked_tokens[j] or span_label_tokens[j] for j in _iterate_span_indices(span)]):
                    # The same token can't be both masked for one pair and unmasked for another pair,
                    # or alternatively masked twice
                    continue

                if any([new_tokens[j] != new_tokens[k] for j, k in
                        zip(_iterate_span_indices(span), _iterate_span_indices(unmasked_span))]):
                    tf.logging.warning(
                        f"Two non-identical spans: unmasked {new_tokens[unmasked_span_beginning:unmasked_span_ending + 1]}, "
                        f"masked:{new_tokens[span[0]:span[1] + 1]}")
                    continue

                is_first_token = True
                for j in _iterate_span_indices(span):
                    if is_first_token:
                        new_tokens[j] = "[SPAN_MASK]"
                        masked_spans.append(MaskedSpanInstance(index=j,
                                                               begin_label=unmasked_span_beginning,
                                                               end_label=unmasked_span_ending))
                        num_predictions += 1
                    else:
                        new_tokens[j] = "[PAD]"
                        input_mask[j] = 0

                    is_first_token = False
                    already_masked_tokens[j] = True

                for j in _iterate_span_indices(unmasked_span):
                    span_label_tokens[j] = True

    assert len(masked_spans) <= num_to_predict
    masked_spans = sorted(masked_spans, key=lambda x: x.index)

    masked_span_positions = []
    span_label_beginnings = []
    span_label_endings = []
    for p in masked_spans:
        masked_span_positions.append(p.index)
        span_label_beginnings.append(p.begin_label)
        span_label_endings.append(p.end_label)

    return new_tokens, masked_span_positions, input_mask, span_label_beginnings, span_label_endings



