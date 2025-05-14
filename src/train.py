#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-12 03:34:21
# * @modify date 2025-05-12 03:34:21
# * @desc [description]
#*/
from tokenizers import Tokenizer, Encoding, decoders
from pathlib import Path
from helper.dataset import SentenceDataset, DatasetItemIndex
import torch
from torch import Tensor
from arch.model import SpellCorrectorNet, LayerConfig
from torch.utils.data import DataLoader
from functools import reduce
from tqdm import tqdm

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

def train(model: SpellCorrectorNet, device: str, dataset: SentenceDataset, epoch = 20, batch_size = 64, learning_rate=0.0002):
    model.train()
    scaler = torch.GradScaler(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    encoder_input_buffer = torch.zeros((batch_size, dataset.max_broken_sentence_length), device=device, dtype=torch.int32)
    decoder_input_buffer = torch.zeros((batch_size, dataset.max_correct_sentence_length), device=device, dtype=torch.int32)
    target_buffer = torch.zeros((batch_size, dataset.max_correct_sentence_length), device=device, dtype=torch.int32)

    my_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batches: my_collate_fn(dataset, batches, encoder_input_buffer, decoder_input_buffer, decoder_output_target_buffer=target_buffer))
    
    for e in range(epoch):
        progressbar = tqdm(enumerate(my_dataloader), desc=f"{e}th epoch...", total=len(my_dataloader))
        for batch_index, item in progressbar:
            item: tuple[Tensor, Tensor, Tensor]
            encoder_input, decoder_input, decoder_output_target = item

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                memory = model.generate_memory(x=encoder_input)
                output: Tensor = model(x=decoder_input, memory=memory)
                output = output.reshape(-1, output.shape[2])
                target = decoder_output_target.reshape(-1).to(dtype=torch.int64)
                loss: Tensor = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progressbar.set_description(f"loss: {str(loss.item())[:8]}")

        torch.save(model.state_dict(), str(Path(__file__).parent.parent / "checkpoints/model.pth"))
    

device = "cuda:0"

input_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/input_tokenizer.json"))
output_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/output_tokenizer.json"))
output_tokenizer.decoder = decoders.Metaspace()

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
        print(f"Length of dataset: {len(my_dataset)}")
        train(model=model, device=device, dataset=my_dataset, epoch=20, batch_size=240)

main()

