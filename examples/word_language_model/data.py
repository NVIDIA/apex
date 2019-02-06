import os
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, pad_to_multiple_of=1):
        # Synthetic elements used to pad the dictionary length.
        # It is assumed that these synthetic elements do not appear in the actual data files.
        self.synthetic = ["vvvvvvvv" + str(i) for i in range(pad_to_multiple_of-1)]

        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        # Pad dictionary size to desired multiple.  For example, padding to a multiple of 8
        # is necessary to ensure Tensor Core usage for the decoder.
        pad_elem = pad_to_multiple_of - len(self.dictionary)%pad_to_multiple_of
        if pad_elem != pad_to_multiple_of:
            for i in range(pad_elem):
                self.dictionary.add_word(self.synthetic[i])

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
