#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-12 03:34:21
# * @modify date 2025-05-12 03:34:21
# * @desc [description]
#*/
from tokenizers import Tokenizer, Encoding, decoders
from pathlib import Path
from helper.dataset import SentenceDataset, DatasetItemIndex, my_collate_fn
import torch
from torch import Tensor
from arch.model import SpellCorrectorNet, LayerConfig
from torch.utils.data import DataLoader
from functools import reduce
from tqdm import tqdm
from typing import Callable, Any
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model: SpellCorrectorNet, optimizer: torch.optim.Adam, grad_scaler: torch.GradScaler, epoch: int, file_name = "checkpoint.pth"):
    dir = Path(__file__).parent.parent / "checkpoints"
    path = dir / file_name
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "grad_scaler": grad_scaler.state_dict(),
        "epoch": epoch,
    }, path)
    torch.save(model.state_dict(), dir / "model.pth")

def load_checkpoint(model: SpellCorrectorNet, optimizer: torch.optim.Adam, grad_scaler: torch.GradScaler, set_epoch: Callable[[int], None] | None = None, file_name = "checkpoint.pth") -> dict[str, Any]:
    path = Path(__file__).parent.parent / f"checkpoints/{file_name}"
    if not path.exists():
        return {
            "epoch": 0
        }
    checkpoint: dict = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    grad_scaler.load_state_dict(checkpoint["grad_scaler"])
    if set_epoch is not None: set_epoch(checkpoint["epoch"])
    return checkpoint

def train(model: SpellCorrectorNet, device: str, dataset: SentenceDataset, logger: SummaryWriter, n_epoch = 20, batch_size = 64, learning_rate=0.00005):
    model.train()
    scaler = torch.GradScaler(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    checkpoint = load_checkpoint(model, optimizer, scaler, None)

    encoder_input_buffer = torch.zeros((batch_size, dataset.max_broken_sentence_length), device=device, dtype=torch.int32)
    decoder_input_buffer = torch.zeros((batch_size, dataset.max_correct_sentence_length), device=device, dtype=torch.int32)
    target_buffer = torch.zeros((batch_size, dataset.max_correct_sentence_length), device=device, dtype=torch.int32)

    my_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batches: my_collate_fn(dataset, batches, encoder_input_buffer, decoder_input_buffer, decoder_output_target_buffer=target_buffer))
    
    counter = 0
    for e in range(checkpoint["epoch"], n_epoch):
        progressbar = tqdm(enumerate(my_dataloader), total=len(my_dataloader))
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

            progressbar.set_description(f"epoch: {e}, loss: {str(loss.item())[:8]}")
            logger.add_scalar("loss", loss.item(), counter, new_style=True)

            if batch_index % 100 == 0:
                save_checkpoint(model, optimizer, scaler, e)
            counter += 1
        save_checkpoint(model, optimizer, scaler, e)

device = "cuda:0"

input_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/input_tokenizer.json"))
output_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/output_tokenizer.json"))
output_tokenizer.decoder = decoders.Metaspace()

model = SpellCorrectorNet(
    encoder_config=LayerConfig(vocab_size=input_tokenizer.get_vocab_size(), d_model=256, n_layer=12, n_head=4),
    decoder_config=LayerConfig(vocab_size=output_tokenizer.get_vocab_size(), d_model=128, n_layer=8, n_head=4),
).to(device)

torch.save(model.state_dict(), str(Path(__file__).parent.parent / "checkpoints/model.pth"))

def main():
    data_dir = Path(__file__).parent.parent / ".data" / "text"
    files = [f"{i * 1000}.json" for i in range(20)]
    logger = SummaryWriter(".logs")

    for f in files:
        dataset_file = (data_dir / f).as_posix()
        my_dataset = SentenceDataset(dataset_file, input_tokenizer, output_tokenizer, broken_sentence_variation_count=40)
        print(f"Length of dataset: {len(my_dataset)} in file: {f}")
        train(model=model, device=device, dataset=my_dataset, logger=logger, n_epoch=1, batch_size=200)

main()

