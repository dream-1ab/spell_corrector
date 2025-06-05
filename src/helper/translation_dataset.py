


from tokenizers import Tokenizer
from torch import Tensor
from tqdm import tqdm
import csv
import random
from typing import TypedDict
from os import PathLike

class TranslationDirection(TypedDict):
    source: str
    target: str
    
class TranslationFile(TypedDict):
    path: PathLike
    directions: list[TranslationDirection]
    
class SentencePairDataset:
    def __init__(self, tokenizer: Tokenizer, tsv_files: list[TranslationFile], max_allowed_tokens = 300):
        self.tokenizer = tokenizer
        self.data: list[tuple[list[int], list[int]]] = []
        
        for tsv_file in tsv_files:
            with open(tsv_file["path"], encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t")
                all_tsv_columns = next(reader)
                column_index = {item: index for index, item in enumerate(all_tsv_columns)}
                
                dirs = ["{source}->{target}".format(source=d["source"], target=d["target"]) for d in tsv_file["directions"]]
                print(f"tsv file: {tsv_file["path"]} contains the following columns: {all_tsv_columns} and will include these columns as language directions: {dirs}")
                for row in tqdm(reader, desc="loading dataset..."):
                    def append_data(row: tuple[str, str], source_language_code: str, target_language_code: str):
                        if len(row) >= 2:
                            source = f"<SOS><{target_language_code}>{row[0].strip().lower()}</{target_language_code}><EOS>"
                            target = f"<SOS><{target_language_code}>{row[1].strip().lower()}</{target_language_code}><EOS>"
                            source_language: list[int] = tokenizer.encode(source).ids
                            target_language: list[int] = tokenizer.encode(target).ids
                            if len(source_language) > max_allowed_tokens or len(target_language) > max_allowed_tokens:
                                return
                            self.data.append((source_language, target_language))
                            
                            # randomly add same sentence as source and target to maintain language tag consistency.
                            if random.random() > 0.95:
                                self.data.append((target_language, target_language))
                    for dir in tsv_file["directions"]:
                        append_data((row[column_index[dir["source"]]], row[column_index[dir["target"]]]), dir["source"], dir["target"])
                        #the reverse is blocked because single source -> multiple target sentence causes issue.
                        # append_data((row[1], row[0]), language_pair[1], language_pair[0])
                    
                    
        self.data = sorted(self.data, key=lambda item: len(item))
        self.print_data_count()
    
    def print_data_count(self):
        print(f"size of dataset: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def split_into_train_set_validation_set(self, trainset_count = 4096) -> tuple["SentencePairDataset", "SentencePairDataset"]:
        trainset_count = min(trainset_count, 32768)
        trainset_count = max(trainset_count, 128)
        train, test = SentencePairDataset(self.tokenizer, []), SentencePairDataset(self.tokenizer, [])
        testset_indices: set[int] = set()
        
        #Append number of testset sample into testset.
        while True:
            if len(testset_indices) >= trainset_count:
                break
            idx = random.randint(0, len(self.data) - 1)
            if idx in testset_indices:
                continue
            test.data.append(self.data[idx])
            testset_indices.add(idx)
        
        #Append remaining data to trainset.
        for i, item in enumerate(self.data):
            if i in testset_indices:
                continue
            else:
                train.data.append(item)
        
        train.print_data_count()
        test.print_data_count()
        return train, test

def collete_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
    """
    first tuple item in batch is Uyghur and second tuple item is English.
    we have to calculate the maximum length of item in batch and make them as tuple of tensor.
    """
    # INSERT_YOUR_CODE
    import torch

    # Separate Uyghur and English sequences
    source_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]

    # Find max lengths
    max_source_len = max(len(seq) for seq in source_seqs)
    max_target_len = max(len(seq) for seq in target_seqs)

    # Pad sequences
    def pad_sequence(seqs, max_len, pad_value=0):
        return [
            seq + [pad_value] * (max_len - len(seq))
            for seq in seqs
        ]

    source_padded = pad_sequence(source_seqs, max_source_len)
    target_padded = pad_sequence(target_seqs, max_target_len)

    # Convert to tensors
    source_tensor = torch.tensor(source_padded, dtype=torch.long)
    target_tensor = torch.tensor(target_padded, dtype=torch.long)

    return source_tensor, target_tensor