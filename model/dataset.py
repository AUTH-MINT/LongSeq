from model.config import Config
import torch
import model.utils as utils


class EntityDataset():
    def __init__(self, words, tags, config):
        self.words = words
        self.tags = tags
        self.config = config


    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        word = self.words[item]
        tags = self.tags[item]

        ids =[]
        target_tag =[]

        #Tokenizing, add CLS / SEP, MASKing, TOKEN,TYPE, PADing
        for i,s in enumerate(word):
            inputs = self.config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

            assert len(ids)==len(target_tag), 'len(ids) and len(target_tag) must be equals'

        ids = ids[:self.config.MAX_LEN - 2]
        target_tag = target_tag[:self.config.MAX_LEN - 2]

        #CLS -> 101 // SEP -> 102
        ids = self.config.cls_tok + ids + self.config.sep_tok
        target_tag = self.config.class_X_enc + target_tag + self.config.class_X_enc

        #Make mask anf token type
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        ## Padding
        padding_len = self.config.MAX_LEN - len(ids)

        ids = ids + (self.config.pad_tok * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + (self.config.class_X_enc * padding_len)

        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.tensor(mask, dtype=torch.uint8),
            "token_type_ids": torch.LongTensor(token_type_ids),
            "target_tag": torch.LongTensor(target_tag)
        }


