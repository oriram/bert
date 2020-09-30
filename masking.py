import collections
import numpy as np

from tokenization import SPECIAL_TOKENS

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
                                 max_predictions_per_seq, vocab_words, rng, do_whole_word_mask=False):
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
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
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
