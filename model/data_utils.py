import numpy as np
import os
import model.utils as utils



# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, max_iter=None):
        """
        Args:
            filename: path to the file
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0


        with open(self.filename) as f:
            words, tags, enc_tags = [], [], []
            data = {}

            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []

                else:
                    ls = line.split(' ')
                    word, pos, tag = ls[0], ls[1], ls[-1]
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length




def data_to_dict(CoNLLData):
    """Transform tuples from def CoNLLDataset to dictionary with this form:
     data = {'words':[]
             'tags':[]
             'enc_tags:[]}"""

    data={'words':[], 'tags':[], 'enc_tags':[]}
    for i in CoNLLData:
        enc_tags = []
        data['words'].append(i[0])
        data['tags'].append(i[1])
        for k in i[1]:
            t = [k]
            # print("t",t)
            enc_tag = utils.enc_trans.transform(t)
            # print("enc_tag", enc_tag)
            enc_tags.append(enc_tag[0])
        data['enc_tags'].append(enc_tags)
    return data

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f