"""
.. module:: predictor
    :synopsis: prediction method (for un-annotated text)
 
"""

import torch.autograd as autograd
from tqdm import tqdm

from model.crf import CRFDecode_vb
from model.utils import *


class predict:
    """Base class for prediction, provide method to calculate f1 score and accuracy 

    args: 
        if_cuda: if use cuda to speed up 
        l_map: dictionary for labels 
        batch_size: size of batch in decoding
    """

    def __init__(self, if_cuda, l_map, batch_size=1):
        self.if_cuda = if_cuda
        self.l_map = l_map
        self.r_l_map = revlut(l_map)
        self.batch_size = batch_size
        self.decode_str = self.decode_l

    def decode_l(self, feature, label):
        """
        decode a sentence coupled with label

        args:
            feature (list): words list
            label (list): label list
        """
        return '\n'.join(map(lambda t: t[0] + ' ' + self.r_l_map[t[1]], zip(feature, label)))

    def output_batch(self, ner_model, features, fout):
        """
        decode the whole corpus in the specific format by calling apply_model to fit specific models

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
            fout: output file
        """
        ner_model.eval()

        d_len = len(features)
        if d_len % self.batch_size == 0:
            total_batch = d_len // self.batch_size
        else:
            total_batch = d_len // self.batch_size + 1
        for start in tqdm(range(0, total_batch), desc="-process"):
            end = min(d_len, start + self.batch_size)
            labels = self.apply_model(ner_model, features[start: end])
            labels = labels.transpose(0, 1)
            labels = torch.unbind(labels, 1)
            labels = labels.tolist()
            for feature, label in zip(features[start: end], labels):
                fout.write(self.decode_str(feature, label) + '\n\n\n')

    def apply_model(self, ner_model, features):
        """
        template function for apply_model

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        return None


class predict_w(predict):
    """prediction class for word level model (LSTM-CRF)

    args: 
        if_cuda: if use cuda to speed up 
        f_map: dictionary for words
        l_map: dictionary for labels
        pad_word: word padding
        pad_label: label padding
        start_label: start label 
        batch_size: size of batch in decoding
        caseless: caseless or not
    """

    def __init__(self, if_cuda, f_map, l_map, pad_word, pad_label, start_label, batch_size=1):
        predict.__init__(self, if_cuda, l_map, batch_size)
        self.decoder = CRFDecode_vb(len(l_map), start_label, pad_label)
        self.pad_word = pad_word
        self.f_map = f_map
        self.l_map = l_map

    def apply_model(self, ner_model, features):
        """
        apply_model function for LSTM-CRF

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        features = encode_safe(features, self.f_map, self.f_map['<unk>'])
        f_len = max(map(lambda t: len(t) + 1, features))

        masks = torch.ByteTensor(list(map(lambda t: [1] * (len(t) + 1) + [0] * (f_len - len(t) - 1), features)))
        word_features = torch.LongTensor(list(map(lambda t: t + [self.pad_word] * (f_len - len(t)), features)))

        if self.if_cuda:
            fea_v = autograd.Variable(word_features.transpose(0, 1)).cuda()
            mask_v = masks.transpose(0, 1).cuda()
        else:
            fea_v = autograd.Variable(word_features.transpose(0, 1))
            mask_v = masks.transpose(0, 1).contiguous()

        scores, _ = ner_model(fea_v)
        decoded = self.decoder.decode(scores.data, mask_v)

        return decoded


class predict_wc(predict):
    """prediction class for LM-LSTM-CRF

    args: 
        if_cuda: if use cuda to speed up 
        f_map: dictionary for words
        c_map: dictionary for chars
        l_map: dictionary for labels
        pad_word: word padding
        pad_char: word padding
        pad_label: label padding
        start_label: start label 
        batch_size: size of batch in decoding
        caseless: caseless or not
    """

    def __init__(self, if_cuda, f_map, c_map, l_map, pad_word, pad_char, pad_label, start_label, batch_size=50):
        predict.__init__(self, if_cuda, l_map, batch_size)
        self.decoder = CRFDecode_vb(len(l_map), start_label, pad_label)
        self.pad_word = pad_word
        self.pad_char = pad_char
        self.f_map = f_map
        self.c_map = c_map
        self.l_map = l_map

    def apply_model(self, ner_model, features):
        """
        apply_model function for LM-LSTM-CRF

        args:
            ner_model: sequence labeling model
            feature (list): list of words list
        """
        char_features = encode2char_safe(features, self.c_map)
        word_features = encode_safe(features, self.f_map, self.f_map['<unk>'])
        fea_len = [list(map(lambda t: len(t) + 1, f)) for f in char_features]
        forw_features = concatChar(char_features, self.c_map)

        word_len = max(map(lambda t: len(t) + 1, word_features))
        char_len = max(map(lambda t: len(t[0]) + word_len - len(t[1]), zip(forw_features, word_features)))
        forw_t = list(map(lambda t: t + [self.pad_char] * (char_len - len(t)), forw_features))
        back_t = torch.LongTensor(list(map(lambda t: t[::-1], forw_t)))
        forw_t = torch.LongTensor(forw_t)
        forw_p = torch.LongTensor(
            list(map(lambda t: list(itertools.accumulate(t + [1] * (word_len - len(t)))), fea_len)))
        back_p = torch.LongTensor(list(map(lambda t: [char_len - 1] + [char_len - 1 - tup for tup in t[:-1]], forw_p)))

        masks = torch.ByteTensor(list(map(lambda t: [1] * (len(t) + 1) + [0] * (word_len - len(t) - 1), word_features)))
        word_t = torch.LongTensor(list(map(lambda t: t + [self.pad_word] * (word_len - len(t)), word_features)))

        if self.if_cuda:
            f_f = autograd.Variable(forw_t.transpose(0, 1)).cuda()
            f_p = autograd.Variable(forw_p.transpose(0, 1)).cuda()
            b_f = autograd.Variable(back_t.transpose(0, 1)).cuda()
            b_p = autograd.Variable(back_p.transpose(0, 1)).cuda()
            w_f = autograd.Variable(word_t.transpose(0, 1)).cuda()
            mask_v = masks.transpose(0, 1).cuda()
        else:
            f_f = autograd.Variable(forw_t.transpose(0, 1))
            f_p = autograd.Variable(forw_p.transpose(0, 1))
            b_f = autograd.Variable(back_t.transpose(0, 1))
            b_p = autograd.Variable(back_p.transpose(0, 1))
            w_f = autograd.Variable(word_t.transpose(0, 1))
            mask_v = masks.transpose(0, 1)

        scores = ner_model(f_f, f_p, b_f, b_p, w_f)
        decoded = self.decoder.decode(scores.data, mask_v)

        return decoded
