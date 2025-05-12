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


class SentenceDataset(Dataset[tuple[list[int], int]]):
    def __init__(self, sentence_file_path: str, input_tokenizer: Tokenizer, output_tokenizer: Tokenizer, broken_sentence_variation_count: int = 20, broke_factor = random.random() * 0.1 + 0.1):
        super().__init__()

        self.sentence_file_path = sentence_file_path

        with open(sentence_file_path, "r") as file:
            sentences: list[str] = json.loads(file.read())
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

        self.broken_sentences: list[tuple[list[int], int]] = []
        self.correct_sentences: list[list[int]] = []


        for i, s in enumerate(tqdm(expanded_sentences, desc="Processing dataset...", ncols=200)):
            correct_ids: list[int] = output_tokenizer.encode(f"<SOS>{s}<EOS>").ids
            self.correct_sentences.append(correct_ids)
            broken_idss: list[tuple[list[int], int]] = [(input_tokenizer.encode(f"<SOS>{destruct_sentence(input_tokenizer, s, broke_factor)}<EOS>").ids, i) for _ in range(broken_sentence_variation_count)]
            self.broken_sentences.extend(broken_idss)

            self.max_correct_sentence_length = max(len(correct_ids), self.max_correct_sentence_length)
            for b in broken_idss:
                self.max_broken_sentence_length = max(len(b[0]), self.max_broken_sentence_length)
        
        #sort sentences.
        self.correct_sentences.sort(key=lambda item: len(item))
        self.broken_sentences.sort(key=lambda item: len(item[0]))
    
    def __len__(self):
        return len(self.broken_sentences)
    
    def __getitem__(self, index) -> tuple[list[int], int]:
        return self.broken_sentences[index]