import re
import numpy as np
from transformers import AlbertTokenizer


def only_eng_to_kor(text):
    pattern = re.compile("[^"
                         u"\U00000020-\U0000007E"
                         u"\U00001100-\U000011FF"
                         u"\U00003131-\U0000318F"
                         u"\U0000AC00-\U0000D7A3"
                         "]+", flags=re.UNICODE)
    return re.sub(pattern, '', text).lower()


class TrainSetMaker:
    ''' This class makes train data set (masking, padding ...) '''
    def __init__(self, config):
        self.config = config
        self.tokenizer = AlbertTokenizer('./albert_tokenizer')

    @staticmethod
    def _chunk(li: list, n: int):
        for i in range(0, len(li), n):
            yield li[i: i + n]

    @staticmethod
    def _is_start_piece(piece):
        p = re.compile('[^▁가-힣ㄱ-ㅎa-zA-Z]')
        if not p.search(piece) and piece[0] == '▁':
            return True
        else:
            return False

    def _sample_mask(self, seg, n_pred):
        """ n-gram masking SpanBERT(Joshi et al., 2019), reference code: https://github.com/graykode/ALBERT-Pytorch """
        seg_len = len(seg)
        mask = np.array([False] * seg_len, dtype=np.bool_)

        num_predict = 0

        ngrams = np.arange(1, self.config.max_gram + 1)
        pvals = 1. / np.arange(1, self.config.max_gram + 1)
        pvals /= pvals.sum(keepdims=True)

        cur_len = 0

        while cur_len < seg_len:
            if n_pred is not None and num_predict >= n_pred:
                break
            n = np.random.choice(ngrams, p=pvals)
            if n_pred is not None:
                n = min(n, n_pred - num_predict)

            ctx_size = (n * self.config.mask_alpha) // self.config.mask_beta
            l_ctx = np.random.choice(ctx_size)
            r_ctx = ctx_size - l_ctx

            # find the start position of a complete token
            beg = cur_len + l_ctx

            while beg < seg_len and not self._is_start_piece(seg[beg]):
                beg += 1
            if beg >= seg_len:
                break

            end = beg + 1
            cnt_ngram = 1
            while end < seg_len:
                if self._is_start_piece(seg[beg]):
                    cnt_ngram += 1
                    if cnt_ngram > n:
                        break
                end += 1
            if end >= seg_len:
                break
            mask[beg:end] = True
            num_predict += end - beg

            cur_len = end + r_ctx

        while n_pred is not None and num_predict < n_pred:
            i = np.random.randint(seg_len)
            if not mask[i]:
                mask[i] = True
                num_predict += 1

        tokens, masked_tokens, masked_pos = [], [], []
        for i in range(seg_len):
            if mask[i] and (seg[i] != '[CLS]' and seg[i] != '[SEP]'):
                masked_tokens.append(seg[i])
                masked_pos.append(i)
                tokens.append('[MASK]')
            else:
                tokens.append(seg[i])

        return masked_tokens, masked_pos, tokens

    def _padding(self, value: list):
        if self.config.max_position_embeddings > len(value):
            value.extend([0] * (self.config.max_position_embeddings - len(value)))
        return value

    def collate_fn_sop(self, batch):
        if self.tokenizer is None:
            raise NotImplementedError
        doc = []
        for data in batch:
            for key, value in data.items():
                text = ''
                if type(value) == list:
                    for v in value:
                        text += v
                else:
                    text += value
                token = self.tokenizer.tokenize(only_eng_to_kor(text))
                if len(token) < self.config.min_sentence_length:
                    pass
                else:
                    doc.append(token)

        split_doc = []
        max_token_size = self.config.max_position_embeddings - 3
        for token in doc:
            if len(token) > max_token_size:
                for s_token in list(self._chunk(token, max_token_size)):
                    split_doc.append(s_token)
            else:
                split_doc.append(token)

        input_values = {'input_ids': [],
                        'attention_mask': [],
                        'token_type_ids': [],
                        'labels': [],
                        'sentence_order_label': []}

        for i, token in enumerate(split_doc):
            half = len(token) // 2

            # sentence_order_label
            if i % 2 == 0:
                tokens_a, tokens_b = token[:half], token[half:]
                sentence_order_label = 0
            else:
                tokens_b, tokens_a = token[:half], token[half:]
                sentence_order_label = 1

            # tokens
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            n_pred = min(self.config.max_pred, max(1, int(round(len(tokens) * self.config.mask_prob))))
            masked_tokens, masked_pos, input_tokens = self._sample_mask(tokens, n_pred)

            # input_ids
            input_ids = self.tokenizer.encode(input_tokens, add_special_tokens=False)

            # attention_mask
            attention_mask = [1] * len(tokens)
            for pos in masked_pos:
                attention_mask[pos] = 0

            # token_type_ids
            token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            # labels
            labels = self.tokenizer.encode(tokens, add_special_tokens=False)

            input_values['input_ids'].append(self._padding(input_ids))
            input_values['attention_mask'].append(self._padding(attention_mask))
            input_values['token_type_ids'].append(self._padding(token_type_ids))
            input_values['labels'].append(self._padding(labels))
            input_values['sentence_order_label'].append(sentence_order_label)

        return input_values
