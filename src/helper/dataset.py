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

class SentenceDataset():
    def __init__(self, sentence_files: list[PathLike], input_tokenizer: Tokenizer, output_tokenizer: Tokenizer, broken_sentence_variation_count: int = 2):
        super().__init__()

        expanded_sentences: list[str] = []
        for sentence_file_path in sentence_files:
            with open(sentence_file_path, "r") as file:
                sentences: list[str] = json.load(file)
            # print(f"sentence count before split: {len(sentences)}")

            for s in sentences:
                ss = [ss for ss in s.strip().split(".") if not (ss.strip() == "")]
                ss = [s for s in ss if len(s) > 5 and len(s) < 500]
                ss = [s.lower() for s in ss]
                expanded_sentences.extend(ss)
        
        self.data: list[tuple[list[int], list[int]]] = []
        print(f"sentence count after split: {len(expanded_sentences)}")
        for s in tqdm(expanded_sentences, desc="loading dataset..."):
            decoder_out: list[int] = output_tokenizer.encode(f"<SOS>{s}<EOS>").ids
            encoder_in: list[int] = input_tokenizer.encode(f"<SOS>{s}<EOS>").ids
            self.data.append((encoder_in, decoder_out))
            for _ in range(broken_sentence_variation_count):
                encoder_in = f"<SOS>{destruct_sentence(input_tokenizer, s, 0.10 + (random.random() * 0.05))}<EOS>"
                encoder_in: list[int] = input_tokenizer.encode(encoder_in).ids
                self.data.append((encoder_in, decoder_out))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def my_collete_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
    max_source_length = max(len(i) for i, _ in batch)
    max_target_length = max(len(i) for _, i in batch)
    
    pad_value = [0] * max(max_source_length, max_target_length)
    def pad_sequence(sequence: list[int], length: int) -> list[int]:
        return sequence + pad_value[0:length - len(sequence)]
    
    encoder_in_tensor = torch.tensor([pad_sequence(i, max_source_length) for i, _ in batch])
    decoder_out_tensor = torch.tensor([pad_sequence(i, max_target_length) for _, i in batch])
    
    return encoder_in_tensor, decoder_out_tensor