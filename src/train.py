#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-12 03:34:21
# * @modify date 2025-05-12 03:34:21
# * @desc [description]
#*/
from tokenizers import Tokenizer, Encoding
from pathlib import Path
from helper.dataset import SentenceDataset
import torch
from torch import Tensor
from arch.model import SpellCorrectorNet, LayerConfig
from torch.utils.data import DataLoader
from functools import reduce
from tqdm import tqdm
from helper.autoregressive_generator import RegressiveBufferGenerator

def my_collete_fn(items: list[tuple[list[int], int]]):
    max_sentence_length = reduce(lambda a, b: max(a, b), map(lambda item: len(item[0]), items))
    for item in items:
        item[0].extend([0 for _ in range(max_sentence_length - len(item[0]))])
    return items


def copy_unpadded_tokens_into_buffer(tokens: list[list[int]], buffer: Tensor):
    buffer.zero_()
    if len(tokens) == 0: return
    for i, row in enumerate(tokens):
        length = len(row)
        buffer[i, :length] = torch.tensor(row, dtype=buffer.dtype, device=buffer.device)

def copy_padded_tokens_into_buffer(tokens: list[list[int]], buffer: Tensor):
    buffer.zero_()
    if len(tokens) == 0: return
    buffer[:len(tokens), :len(tokens[0])] = torch.tensor(tokens, dtype=buffer.dtype, device=buffer.device)




def train(model: SpellCorrectorNet, device: str, dataset: SentenceDataset, epoch = 20, batch_size = 64):
    my_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collete_fn)
    scaler = torch.GradScaler(device=device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    input_buffer = torch.zeros((batch_size, dataset.max_broken_sentence_length), device=device, dtype=torch.int32)
    original_sentences_buffer = torch.zeros((len(dataset.correct_sentences), dataset.max_correct_sentence_length), device=device, dtype=torch.int32)
    decoder_input_buffer = torch.zeros((batch_size, dataset.max_correct_sentence_length), device=device, dtype=torch.int32)
    copy_unpadded_tokens_into_buffer(dataset.correct_sentences, original_sentences_buffer)

    for e in range(epoch):
        for batch_index, x in tqdm(enumerate(my_dataloader), desc=f"{e}th epoch...", total=len(my_dataloader)):
            x: list[tuple[list[int], int]]
            copy_padded_tokens_into_buffer(list(map(lambda i: i[0], x)), input_buffer)
            generator = RegressiveBufferGenerator(original_sentence_buffer=original_sentences_buffer, lengths=list(map(lambda a: len(a), dataset.correct_sentences)))

            max_sentence_length_in_current_batch = 0
            index_map = {i: dataset.correct_sentences[i] for _, i in x}
            for item in index_map.values():
                max_sentence_length_in_current_batch = max(max_sentence_length_in_current_batch, len(item))
            
            for i in range((generator.item_count / batch_size.__ceil__())):
                copied = generator.generate(decoder_input_buffer, batch_size)

            


device = "cuda:0"

input_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/input_tokenizer.json"))
output_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/output_tokenizer.json"))

model = SpellCorrectorNet(
    encoder_config=LayerConfig(vocab_size=input_tokenizer.get_vocab_size(), d_model=128, n_layer=8, n_head=4),
    decoder_config=LayerConfig(vocab_size=output_tokenizer.get_vocab_size(), d_model=128, n_layer=8, n_head=4),
).to(device)

torch.save(model.state_dict(), str(Path(__file__).parent.parent / "checkpoints/model.pth"))

def main():
    
    data_dir = Path(__file__).parent.parent / ".data" / "text"
    files = ["0.json"]
    for f in files:
        dataset_file = (data_dir / f).as_posix()
        my_dataset = SentenceDataset(dataset_file, input_tokenizer, output_tokenizer, broken_sentence_variation_count=20)

        train(model=model, device=device, dataset=my_dataset, epoch=1, batch_size=512)


main()

