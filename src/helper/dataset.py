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