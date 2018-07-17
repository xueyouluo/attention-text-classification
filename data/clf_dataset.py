import time
import json
import random

import numpy as np
from tqdm import tqdm

from utils.data_utils import EOS_ID, SOS_ID, UNK_ID, read_vocab

class CLFDataSet(object):
    """Helper class for processing data
    """
    def __init__(self, vocab_file, label_file, data_file=None, batch_size=None, max_len=300, min_len=0, label_type='multi-class'):
        self.data_file = data_file
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.label_file = label_file
        self.max_len = max_len
        self.min_len = min_len
        self.label_type = label_type

        self.w2i, self.i2w = read_vocab(self.vocab_file)
        self.l2i, self.i2l = read_vocab(self.label_file,check_vocab=False)
        self._raw_data = []

        if self.data_file:
            self._preprocess()

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def label_size(self):
        return len(self.l2i)

    def get_label(self, labels, normalize=False):
        one_hot_labels = np.zeros(len(self.l2i))
        for n in labels:
            one_hot_labels[self.l2i[n]] = 1
        
        if normalize:
            one_hot_labels = one_hot_labels / len(labels)
        return one_hot_labels

    def tokenize(self, content, inference=False):
        tokens = content.strip().split()
        if not inference:
            if len(tokens) <= self.min_len:
                return None
        tokens = [SOS_ID] + [self.w2i.get(t,UNK_ID) for t in tokens]
        if not inference:
            tokens = tokens[:self.max_len]
        return tokens

    def _preprocess(self):
        print("preprocessing data...")
        start = time.time()

        for line in tqdm(open(self.data_file)):
            if not line.strip():
                continue
            item = json.loads(line)
            content = item['content']
            labels = item['labels']
            tokens = self.tokenize(content)
            length = len(tokens)
            one_hot_labels = self.get_label(labels)
            self._raw_data.append((tokens,length,one_hot_labels))

        self.num_batches = len(self._raw_data) // (self.batch_size)
        end = time.time()
        print("processed {0} data within {1} seconds".format(len(self._raw_data),(end-start)))

    def padding(self,tokens_list):
        max_len = max([len(t) for t in tokens_list])
        ret = np.zeros((len(tokens_list),max_len),np.int32)
        for i,t in enumerate(tokens_list):
            t = t + (max_len-len(t)) * [EOS_ID]
            ret[i] = t
        return ret
    
    def get_next(self, shuffle=False):
        if shuffle:
            random.shuffle(self._raw_data)
        for i in range(self.num_batches):
            data = self._raw_data[i*self.batch_size:(i+1)*self.batch_size]
            source,lengths,labels = zip(*data)
            # batch * max_len
            source = self.padding(source)
            # batch * num_label
            labels = np.asarray(labels)
            # batch
            lengths = np.asarray(lengths)

            yield (source,lengths,labels)
        
        # final batch
        if self.num_batches * self.batch_size < len(self._raw_data):
            rest_size = len(self._raw_data) - self.num_batches * self.batch_size
            data = self._raw_data[-rest_size:]
            source,lengths,labels = zip(*data)
            source = self.padding(source)
            labels = np.asarray(labels)
            lengths = np.asarray(lengths)

            yield (source,lengths,labels)
