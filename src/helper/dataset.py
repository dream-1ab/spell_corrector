#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-12 00:31:34
# * @modify date 2025-05-12 00:31:34
# * @desc [description]
#*/



from torch.utils.data import Dataset
from tokenizers import Tokenizer, Encoding
from helper.sentence_destructor import destruct_sentence
import json
from pathlib import Path
from typing import TypedDict
import random
from tqdm import tqdm
from array import array
from torch import Tensor
import torch

class DatasetItemIndex(TypedDict):
    encoder_input_id: int
    decoder_input_id: int

def make_autoregressive(value: list[int]) -> list[list[int]]:
    values = []
    for i in range(1, len(value) + 1):
        values.append(value[:i])
    return values


class SentenceDataset(Dataset[DatasetItemIndex]):
    def __init__(self, sentence_file_path: str, input_tokenizer: Tokenizer, output_tokenizer: Tokenizer, broken_sentence_variation_count: int = 20):
        super().__init__()

        self.sentence_file_path = sentence_file_path

        with open(sentence_file_path, "r") as file:
            sentences: list[str] = json.load(file)
        # print(f"sentence count before split: {len(sentences)}")
        expanded_sentences: list[str] = []

        for s in sentences:
            ss = [ss for ss in s.strip().split(".") if not (ss.strip() == "")]
            ss = [s for s in ss if len(s) > 15]
            ss = [s.lower() for s in ss]
            expanded_sentences.extend(ss)
        # print(f"sentence count after split: {len(expanded_sentences)}")

        self.max_correct_sentence_length = 0
        self.max_broken_sentence_length = 0

        self.broken_sentences:  list[list[int]] = []
        self.correct_sentences: list[tuple[list[int], int]] = []

        self.indices: list[DatasetItemIndex] = []


        for s in tqdm(expanded_sentences, desc="Preparing dataset...", ncols=200):
            correct_sentence_tokens: list[list[int]] = make_autoregressive(output_tokenizer.encode(f"<SOS>{s}<EOS>").ids)
            target_tokens = [i[-1] for i in correct_sentence_tokens[1:]]
            correct_sentence_tokens = correct_sentence_tokens[:len(target_tokens)]
            correct_sentence_tokens = [(correct_sentence_tokens[i], target_tokens[i]) for i in range(len(correct_sentence_tokens))]
            correct_sentence_tokens: list[tuple[list[int], int]]
            broken_sentence_tokens:  list[list[int]] = [input_tokenizer.encode(f"<SOS>{destruct_sentence(input_tokenizer, s, (random.random() * 0.1) + 0.1)}<EOS>").ids for _ in range(broken_sentence_variation_count)]

            correct_sentence_indices = range(len(self.correct_sentences), len(self.correct_sentences) + len(correct_sentence_tokens))
            broken_sentence_indices  = range(len(self.broken_sentences), len(self.broken_sentences) + len(broken_sentence_tokens))

            self.correct_sentences.extend(correct_sentence_tokens)
            self.broken_sentences.extend(broken_sentence_tokens)

            indices = []
            for c in correct_sentence_indices:
                for b in broken_sentence_indices:
                    indices.append(DatasetItemIndex(encoder_input_id=b, decoder_input_id=c))
            self.indices.extend(indices)
        del s
        for s in self.correct_sentences:
            self.max_correct_sentence_length = max(len(s[0]), self.max_correct_sentence_length)
        del s
        for s in self.broken_sentences:
            self.max_broken_sentence_length = max(len(s), self.max_broken_sentence_length)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index) -> DatasetItemIndex:
        return self.indices[index]
    
    def get_dataset_item(self, index: DatasetItemIndex):
        return self.broken_sentences[index["encoder_input_id"]], self.correct_sentences[index["decoder_input_id"]]




def my_collate_fn(dataset: SentenceDataset, items: list[DatasetItemIndex], encoder_input_buffer: Tensor, decoder_input_buffer: Tensor, decoder_output_target_buffer: Tensor):
    encoder_input_buffer[:, :] = 0
    decoder_input_buffer[:, :] = 0
    decoder_output_target_buffer[:, :] = 0
    items_of_pair = [dataset.get_dataset_item(x) for x in items]

    # Determine max lengths
    max_encoder_input = max(len(enc) for enc, _ in items_of_pair)
    max_decoder_input = max(len(dec) for _, (dec, _) in items_of_pair)
    
    encoder_input_tokens, decoder_input_tokens, decoder_output_target_tokens = [], [], []
    encoder_input_tokens: list[list[int]]
    decoder_input_tokens: list[list[int]]
    decoder_output_target_tokens: list[list[int]]

    for _encoder_input_token, (_decoder_input_token, next_token) in items_of_pair:
        #make a shallow copy of list to avoid affect original list.
        encoder_input_token = _encoder_input_token[:]
        decoder_input_token = _decoder_input_token[:]
        decoder_target_token = decoder_input_token[1:]
        decoder_target_token.append(next_token)
        #padding
        encoder_input_token.extend([0 for _ in range(max_encoder_input - len(encoder_input_token))])
        decoder_input_token.extend([0 for _ in range(max_decoder_input - len(decoder_input_token))])
        decoder_target_token.extend([0 for _ in range(max_decoder_input - len(decoder_target_token))])

        encoder_input_tokens.append(encoder_input_token)
        decoder_input_tokens.append(decoder_input_token)
        decoder_output_target_tokens.append(decoder_target_token)
    
    copy_padded_tokens_into_buffer(encoder_input_tokens, encoder_input_buffer)
    copy_padded_tokens_into_buffer(decoder_input_tokens, decoder_input_buffer)
    copy_padded_tokens_into_buffer(decoder_output_target_tokens, decoder_output_target_buffer)

    a, b, c = encoder_input_buffer[:, :max_encoder_input], decoder_input_buffer[:, :max_decoder_input], decoder_output_target_buffer[:, :max_decoder_input]
    return a, b, c

def copy_unpadded_tokens_into_buffer(tokens: list[list[int]], buffer: Tensor):
    if len(tokens) == 0: return
    for i, row in enumerate(tokens):
        length = len(row)
        buffer[i, :length] = torch.tensor(row, dtype=buffer.dtype, device=buffer.device)

def copy_padded_tokens_into_buffer(tokens: list[list[int]], buffer: Tensor):
    if len(tokens) == 0: return
    buffer[:len(tokens), :len(tokens[0])] = torch.tensor(tokens, dtype=buffer.dtype, device=buffer.device)
