import torch
import pickle
from transformers import BertTokenizer
import gluonnlp as nlp
from kobert.utils import get_tokenizer, tokenizer
from kobert.utils import download as _download
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer

from tqdm import tqdm

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

class Bertsum_Dataset:
    def __init__(
        self,
        data_path,
        ) -> None:

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.src_tokens = data["src_tokens"]
        self.tgt_tokens = data["tgt_tokens"]

        self.src_string = data["src_raw"]
        self.tgt_string = data["tgt_raw"]
        self.ext_labels = data["ext_labels"]
        
        self.vocab = get_kobert_vocab()
        self.tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), self.vocab, lower=False)    

        self.pad_idx = self.vocab["[PAD]"]
        self.cls_idx = self.vocab["[CLS]"]
        self.sep_idx = self.vocab["[SEP]"]
        self.mask_idx = self.vocab["[MASK]"]
        self.bos_idx = self.vocab["[BOS]"]
        self.eos_idx = self.vocab["[EOS]"]

    # for BertSum (extractive , abstractive)
    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def add_sentence_token(self, token_ids):
        # bos = 8003, eos = 8002 in korean
        # return [self.bos_idx] + token_ids + [self.eos_idx]
        return [self.bos_idx] + token_ids + [self.eos_idx]

    # ext
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
    
    def preprocess(self):
        data = []
        for idx in tqdm(range(len(self.src_tokens))):
            src_token_ids = []
            src_document = []
            src_document_types = []

            for lines in self.src_tokens[idx]:
                src_token_ids.extend([self.tokenizer.convert_tokens_to_ids(words) for words in lines])

            src_add_special = [self.add_special_token(lines) for lines in src_token_ids]
            src_token_types = self.get_token_type_ids(src_add_special)

            for lines in src_add_special:
                src_document += lines

            for lines in src_token_types:
                src_document_types += lines
                
            # tgt
            tgt_token_ids = []
            tgt_document = []
            for lines in self.tgt_tokens[idx]:
                tgt_token_ids.extend(
                    [self.tokenizer.convert_tokens_to_ids(words) for words in lines]
                )

            tgt_add_sent = [self.add_sentence_token(lines) for lines in tgt_token_ids]

            for lines in tgt_add_sent:
                tgt_document += lines
                
            cls_index = self.get_cls_index(src_document)
            
            src_sent_labels = [1 if i in self.ext_labels[idx] else 0 for i in range(len(cls_index))]

            data.append({'src': src_document, 'tgt':tgt_document, 'src_sent_labels': src_sent_labels, 'segs': src_document_types, 'clss': cls_index, 'src_txt': self.src_string[idx], 'tgt_txt': self.tgt_string[idx][0]})
        return data
