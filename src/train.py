#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-12 03:34:21
# * @modify date 2025-05-12 03:34:21
# * @desc [description]
#*/
from tokenizers import Tokenizer, Encoding, decoders, pre_tokenizers
from pathlib import Path
from helper.dataset import SentenceDataset, my_collete_fn
import torch
from torch import Tensor
from arch.model import SpellCorrectorNet, LayerConfig
from torch.utils.data import DataLoader, random_split
from functools import reduce
from tqdm import tqdm
from typing import Callable, Any
from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention import SDPBackend, sdpa_kernel
from helper.datasetfrom_lmdb import LMDBDataset, my_collete_fn

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

def eval(model: SpellCorrectorNet, device: str, dataset: SentenceDataset, batch: int):
    model.eval()
    data_loader = DataLoader(dataset, batch, False, collate_fn=my_collete_fn)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    all_loss = 0.0
    for encoder_input, _decoder_input in data_loader:
        encoder_input: Tensor; _decoder_input: Tensor
        encoder_input = encoder_input.to(device)
        _decoder_input = _decoder_input.to(device)
        
        encoder_input = encoder_input
        decoder_input = _decoder_input[:, :-1]
        decoder_output = _decoder_input[:, 1:]
        with torch.autocast(device_type=device, dtype=torch.float16):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                memory, memory_mask = model.generate_memory(encoder_input)
                output: Tensor = model(decoder_input, memory, memory_mask)
        output = output.reshape(-1, output.shape[2])
        target = decoder_output.reshape(-1)
        loss: Tensor = criterion(output, target)
        all_loss += loss.item()
    avg_loss = all_loss / len(data_loader)
    return avg_loss
        

def train(model: SpellCorrectorNet, device: str, trainset: SentenceDataset, validationset: SentenceDataset, logger: SummaryWriter, n_epoch = 20, batch_size = 64, learning_rate=0.00003):
    model.train()
    scaler = torch.GradScaler(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    
    epoch_start = 0
    def set_epoch(_epoch: int):
        nonlocal epoch_start
        epoch_start = _epoch
    load_checkpoint(model, optimizer, scaler, set_epoch=set_epoch)
    
    my_dataloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, collate_fn=my_collete_fn)
    
    counter = 0
    current_validation_loss = 0
    previous_validation_loss = 1000.0
    for e in range(epoch_start, n_epoch):
        progressbar = tqdm(enumerate(my_dataloader), total=len(my_dataloader), ncols=200)
        for batch_index, item in progressbar:
            item: tuple[Tensor, Tensor]
            source_input, target_input = item
            source_input = source_input.to(device)
            target_input = target_input.to(device)
            
            encoder_input = source_input
            decoder_input = target_input[:, :-1]
            decoder_output = target_input[:, 1:]
            

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    memory, memory_key_padding_mask = model.generate_memory(x=encoder_input)
                    output: Tensor = model(x=decoder_input, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
                
            output = output.reshape(-1, output.shape[2])
            target = decoder_output.reshape(-1)
            loss: Tensor = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progressbar.set_description(f"epoch: {e}, train loss: {str(loss.item())[:8]}, validation_loss: {str(current_validation_loss)[:8]}")

            if counter % 100 == 0:
                current_validation_loss = eval(model, device, validationset, batch_size)
                logger.add_scalar("train loss", loss.item(), counter, new_style=True)
                logger.add_scalar("validation loss", current_validation_loss, counter, new_style=True)
                model.train()
                if current_validation_loss < previous_validation_loss:
                    save_checkpoint(model, optimizer, scaler, e)
                    previous_validation_loss = current_validation_loss
            counter += 1
            
device = "cuda:0"

input_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/input_tokenizer.json"))
output_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/output_tokenizer.json"))
output_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
output_tokenizer.decoder = decoders.Metaspace()
# output_tokenizer = input_tokenizer

model = torch.compile(SpellCorrectorNet(
    encoder_config=LayerConfig(vocab_size=input_tokenizer.get_vocab_size(), d_model=256, n_layer=6, n_head=8),
    decoder_config=LayerConfig(vocab_size=output_tokenizer.get_vocab_size(), d_model=256, n_layer=4, n_head=8),
).to(device))

def main():
    # data_dir = Path(__file__).parent.parent / ".data" / "text"
    # files = [data_dir / f"{i * 1000}.json" for i in range(120, 140)]

    data_dir = Path(__file__).parent.parent / ".data"
    # "text/2000.json"
    files = [
        data_dir / "dataset.json",
        # data_dir / "text/2000.json",
        # data_dir / "text/3000.json",
        # data_dir / "text/4000.json",
        # data_dir / "text/5000.json",
        # data_dir / "text/6000.json",
        # data_dir / "text/7000.json",
        # data_dir / "text/8000.json",
        # data_dir / "text/9000.json",
        # data_dir / "text/10000.json",
    ]

    logger = SummaryWriter(".logs")
    
    dataset = LMDBDataset(str(Path(__file__).parent.parent / ".data/temp/by_sentences_lmdb"))
    train_set, validation_Set = random_split(dataset, [len(dataset) - 2048, 2048])
    train(model=model, device=device, trainset=train_set, validationset=validation_Set, logger=logger, n_epoch=1800, batch_size=200, learning_rate=0.00005)

    # my_dataset = SentenceDataset(files, input_tokenizer, output_tokenizer, broken_sentence_variation_count=3)
    # train_set, validation_Set = random_split(my_dataset, [len(my_dataset) - 2048, 2048])
    # train(model=model, device=device, trainset=train_set, validationset=validation_Set, logger=logger, n_epoch=1800, batch_size=256, learning_rate=0.00006)
    # train(model=model, device=device, trainset=my_dataset, validationset=my_dataset, logger=logger, n_epoch=1800, batch_size=256, learning_rate=0.00006)

main()

