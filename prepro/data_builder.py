import gc
import glob
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger

import gluonnlp as nlp
from kobert.utils import get_tokenizer, tokenizer
from kobert.utils import download as _download
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer

def get_kobert_vocab(cachedir="./tmp/"):
    # Add BOS,EOS vocab
    vocab_info = tokenizer
    vocab_file = _download(
        vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
    )

    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_file, padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )

    return vocab_b_obj

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def greedy_selection(doc_sent_list, abstract, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9가-힣▁ ]', '', s)

    max_rouge = 0.0
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class BertData:
    def __init__(self, args, vocab, tokenizer):
        self.args = args
        self.vocab = vocab
        self.tokenizer = tokenizer
        
        self.pad_idx = self.vocab["[PAD]"]
        self.cls_idx = self.vocab["[CLS]"]
        self.sep_idx = self.vocab["[SEP]"]
        self.mask_idx = self.vocab["[MASK]"]
        self.bos_idx = self.vocab["[BOS]"]
        self.eos_idx = self.vocab["[EOS]"]

    def preprocess(self, src, tgt, sent_labels, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_token_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in src]
        src_subtoken_idxs = [self.add_special_token(lines) for lines in src_token_ids]
        segments_ids = self.get_token_type_ids(src_subtoken_idxs)

        src_subtoken_idxs = [x for sublist in src_subtoken_idxs for x in sublist]
        segments_ids = [x for sublist in segments_ids for x in sublist]

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt)[:self.args.max_tgt_ntokens]
        tgt_subtoken_idxs = self.add_sentence_token(tgt_subtoken_idxs)
    
        if ((not is_test) and len(tgt_subtoken_idxs) < self.args.min_tgt_ntokens):
            return None

        cls_ids = self.get_cls_index(src_subtoken_idxs)
        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src, tgt

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def add_sentence_token(self, token_ids):
        return [self.bos_idx] + token_ids + [self.eos_idx]

    def get_token_type_ids(self, src_token):
        seg = []
        for i, v in enumerate(src_token):
            if i % 2 == 0:
                seg.append([0] * len(v))
            else:
                seg.append([1] * len(v))
        return seg

    def get_cls_index(self, src_doc):
        cls_index = [index for index, value in enumerate(src_doc) if value == self.cls_idx]
        return cls_index

def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        path = glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.jsonl'))[0]
        with open(path, "r", encoding="utf-8") as f:
            jsonl = list(f)
        data = []
        for json_str in jsonl:
            data.append(json.loads(json_str))
            
        os.makedirs(args.json_path, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)
        for i in range(math.ceil(len(data) / 2000)):
            tmp_path = args.json_path + os.path.splitext(path.split('/')[-1])[0] + f'.{i}.jsonl'
            if os.path.exists(tmp_path):
                logger.info('%s exsists, pass.' % tmp_path)
                continue
            with open(tmp_path, 'w', encoding='utf-8') as f:
                for d in data[i * 2000 : (i + 1) * 2000]:
                    json.dump(d, f, ensure_ascii=False)
                    f.write('\n')
        
        vocab = get_kobert_vocab()
        tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        
        a_lst = []
        for json_f in glob.glob(pjoin(args.json_path, '*' + corpus_type + '.*.jsonl')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, vocab, tokenizer, pjoin(args.save_path, real_name.replace('jsonl', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass
        pool.close()
        pool.join()

def _format_to_bert(batch):
    corpus_type, json_file, args, vocab, tokenizer, save_file = batch
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args, vocab, tokenizer)

    logger.info('Processing %s' % json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        jsonl = list(f)
    jobs = []
    for json_str in jsonl:
        jobs.append(json.loads(json_str))

    datasets = []
    for d in jobs:
        source, tgt = [tokenizer(s) for s in d['article_original']], tokenizer(d['abstractive']) 

        if args.use_anno_labels:
            sent_labels = d['extractive']
        else:
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = ' '.join(tgt).lower().split()
        b_data = bert.preprocess(source, tgt, sent_labels, is_test=is_test)

        if (b_data is None):
            continue

        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
