#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-12 00:31:34
# * @modify date 2025-05-12 00:31:34
# * @desc [description]
#*/



from torch.utils.data import Dataset
from tokenizers import Tokenizer, Encoding
from helper.sentence_destructor import destruct_sentence, mask_sentence
import json
from pathlib import Path
from typing import TypedDict
import random
from tqdm import tqdm
from array import array
from torch import Tensor
import torch
from os import PathLike
import lmdb
import pickle

class LMDBDataset():
    def __init__(self, lmdb_path: PathLike):
        super().__init__()

        self.db = lmdb.open(lmdb_path, readonly=True)
        self.tx = self.db.begin()
        count = self.tx.get(b"count")
        self.count: int = pickle.loads(count)
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        with self.tx.cursor() as cursor:
            data: bytes = cursor.get(f"{index}".encode())
            data: tuple[list[int], list[int]] = pickle.loads(data)
            return data[0], data[1]

def my_collete_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
    max_source_length = max(len(i) for i, _ in batch)
    max_target_length = max(len(i) for _, i in batch)
    
    pad_value = [0] * max(max_source_length, max_target_length)
    def pad_sequence(sequence: list[int], length: int) -> list[int]:
        return sequence + pad_value[0:length - len(sequence)]
    
    encoder_in_tensor = torch.tensor([pad_sequence(i, max_source_length) for i, _ in batch])
    decoder_out_tensor = torch.tensor([pad_sequence(i, max_target_length) for _, i in batch])
    
    return encoder_in_tensor, decoder_out_tensor