import re
import math
import torch
import torch.utils.data

import numpy as np

from random import randint, shuffle, choice
from random import random as rand
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch): # 将batch里面的集合按列输出
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x)) # 默认按照dim=0 stack，即相当于按照batch_size stack
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(None)
    return batch_tensors


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


def truncate_tokens_pair(tokens_a, tokens_b, max_len): # tokens_a和tokens_b哪个长就剔除哪个最后一个
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b


def truncate_tokens_signle(tokens_a, max_len):
    if len(tokens_a) > max_len-2:
        tokens_a = tokens_a[:max_len-2]
    return tokens_a


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(
        self,
        file_data,
        batch_size,
        tokenizer,
        max_len,
        bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size

        # read the file into memory
        self.ex_list = []

        threads = min(8, cpu_count())

        with Pool(threads) as p:
            annotate_ = partial(
                self.read_data,
                tokenizer=self.tokenizer)
            
            self.ex_list = list(
                tqdm(
                    p.map(annotate_, file_data, chunksize=32),
                    total=len(file_data),
                    desc="convert squad examples to features"
                )
            )
        
    def read_data(self, line, tokenizer):
        question = line['question']
        answers = [p['answer'] for p in line['passages'] if p['answer']]
        passage = np.random.choice(line['passages'])['passage']
        passage = re.sub(u' |、|；|，', ',', passage)
        final_answer = ''
        for answer in answers:
            if all([a in passage[:self.max_len - 2] for a in answer.split(' ')]):
                final_answer = answer.replace(' ', ',')
                break
        
        src_tk = question
        tgt_tk = final_answer

        return (src_tk, tgt_tk)

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Seq2seq():
    """ Pre-processing steps for pretraining transformer """
    def __init__(
        self, 
        max_pred, 
        mask_prob, 
        vocab_words, 
        indexer, 
        max_len=512, 
        skipgram_prb=0, 
        skipgram_size=0, 
        mask_whole_word=False, 
        mask_source_words=True, 
        tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(
            torch.ones((max_len, max_len), dtype=torch.long)
        )
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, tokens_b = instance[:2]
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = self.tokenizer.tokenize(tokens_b)
        # -3  for special tokens [CLS], [SEP], [SEP]
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [4]*(len(tokens_a)+2) + [5]*(len(tokens_b)+1)
        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(
            tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4BiLM():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        tokens_a, tokens_b = instance[:2]
        if rand() <= 0.5:
            next_sentence_label = 1.0
        else:
            tokens_a, tokens_b = tokens_b, tokens_a
            next_sentence_label = 0.0

        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = self.tokenizer.tokenize(tokens_b)
        # -3  for special tokens [CLS], [SEP], [SEP]
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        # input_mask[:, :len(tokens_a)+2].fill_(1)
        # second_st, second_end = len(
        #     tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        # input_mask[second_st:second_end, second_st:second_end].copy_(
        #     self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4RightLM():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, _ = instance[:2]
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_a = truncate_tokens_signle(tokens_a, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [2]*(len(tokens_a)+2)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = 0
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            # if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
            #     cand_pos.append(i)
            if (tk != '[CLS]') and (tk != '[SEP]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)

        masked_pos = set()

        try:
            max_cand_pos = max(cand_pos)
        except:
            max_cand_pos = 0

        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        # input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = 0, len(tokens_a)+2
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4LeftLM():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.triu(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, _ = instance[:2]

        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_a = truncate_tokens_signle(tokens_a, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']

        segment_ids = [3]*(len(tokens_a)+2)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = 0
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            # if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
            #     cand_pos.append(i)
            if (tk != '[CLS]') and (tk != '[SEP]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)

        masked_pos = set()

        try:
            max_cand_pos = max(cand_pos)
        except:
            max_cand_pos = 0

        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        # input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = 0, len(tokens_a)+2
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4Seq2seqDecode():
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.max_tgt_length = max_tgt_length

    def __call__(self, instance):
        tokens_a, max_a_len = instance

        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        segment_ids = [4]*(len(padded_tokens_a)) + [5]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, segment_ids, position_ids, input_mask)
