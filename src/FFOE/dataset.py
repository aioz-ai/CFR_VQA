"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import src.utils as utils
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import torch
from torch.utils.data import Dataset
import itertools
COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering


def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer, entity, teacher_logit):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'entity'      : entity,
        'teacher_logit': teacher_logit}
    return entry

def _load_gqa_dataset(dataroot, args, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'gqa_%s_questions_entities.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    if 'test' != name[:4]:  # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))
        entries = []
        # Train and evaluate on tiny sample
        if args.tiny:
            questions = questions[:30000]
            answers = answers[:30000]

        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            entity = question['entities']

            entries.append(_create_entry(img_id2val[img_id], question, answer, entity, None))
    else:  # test
        entries = []
        for question in questions:
            img_id = question['image_id']
            entity = question['entities']
            entries.append(_create_entry(img_id2val[img_id], question, None, entity, None))

    return entries

class GQAFeatureDataset(Dataset):
    def __init__(self, args, name, dictionary, dataroot='data/gqa', adaptive=False):
        super(GQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.max_boxes = args.max_boxes
        self.question_len = args.question_len

        self.dictionary = dictionary
        self.adaptive = adaptive
        self.teacher_logits = []
        print('Create %s entries' % name)

        # load stat_word
        self.stat_words = json.load(open('data/gqa/%s_%s_stats_words.json' % (name, args.topk)))
        self.stat_skip_imgid = json.load(open('data/gqa/%s_%s_stats_skip_imgid.json' % (name, args.topk)))
        self.stat_features = {}

        # load attribute word
        self.attr_words = json.load(open('data/gqa/%s_attr_words_non_plural_words.json' % name))
        self.attr_skip_imgid = json.load(open('data/gqa/%s_attr_skip_imgid.json' % name))
        self.skip_imgid = []
        self.attr_features = {}

        self.ans_list = []

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        # Load image feature
        h5_path = os.path.join(dataroot, '%s.hdf5' % name)
        print('loading features from h5 file %s ' % h5_path)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            self.pos_boxes = np.array(hf.get('pos_boxes'))

        self.entries = _load_gqa_dataset(dataroot, args, name, self.img_id2idx)
        self.tokenize(self.question_len)
        self.stat_word_tokenize_1(args.num_stat_word)
        self.attr_word_tokenize(15)
        self.ans_tokenize()
        self.entity_tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1)
        self.s_dim = self.spatials.size(1)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def entity_tokenize(self, max_length=7):
        """Tokenizes the instruction word.

        This will add entity_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            entity = entry['entity']
            entity = ' '.join(entity)
            tokens = self.dictionary.tokenize(entity, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            entry['entity_token'] = tokens

    def ans_tokenize(self, max_length=2):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            try:
                ans = self.label2ans[entry['answer']['labels'][0]]
                tokens = self.dictionary.tokenize(ans, False)
            except:
                tokens = []

            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['ans_token'] = tokens

    # Tokenize statistical words 2-gram
    def stat_word_tokenize(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            # words = words.split(',')
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:2]
                if len(tokens) < 2:
                    padding = [self.dictionary.padding_idx] * (2 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(2, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.stat_features[img_id] = token_words

    # Tokenize attribute words
    def attr_word_tokenize(self, max_length=15):
        for img_id in self.attr_words:
            words = self.attr_words[img_id]
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:3]
                if len(tokens) < 3:
                    padding = [self.dictionary.padding_idx] * (3 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(3, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.attr_features[img_id] = token_words

    # Tokenize statistical words
    def stat_word_tokenize_1(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            words = words.split(',')
            words = ' '.join(words)
            tokens = self.dictionary.tokenize(words, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            self.stat_features[img_id] = tokens

    def ans_word_tokenize(self, max_length=2):
        ans_list = []
        for ans in self.label2ans:
            tokens = self.dictionary.tokenize(ans, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            ans_list.append(tokens)
        self.ans_list = ans_list

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            entity = torch.from_numpy(np.array(entry['entity_token']))
            entry['entity_token'] = entity
            ans = torch.from_numpy(np.array(entry['ans_token']))
            entry['ans_token'] = ans

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        features = features[:self.max_boxes]
        spatials = spatials[:self.max_boxes]

        question = entry['q_token']
        sent = entry['question']
        entity = entry['entity_token']
        question_id = entry['question_id']
        answer = entry['answer']
        img_id = str(entry['image_id'])
        stat_features = torch.from_numpy(np.array(self.stat_features[img_id]))
        attr_features = torch.from_numpy(np.array(self.attr_features[img_id]))
        ans = entry['ans_token']

        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, stat_features, entity, attr_features, question, sent, target, ans
        else:
            return features, spatials, stat_features, entity, attr_features, question, sent, question_id, img_id, ans

    def __len__(self):
        return len(self.entries)
