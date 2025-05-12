#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-10 13:20:25
# * @modify date 2025-05-10 13:20:25
# * @desc [description]
#*/

import torch
from arch.model import SpellCorrectorNet, LayerConfig
from tokenizers import Tokenizer
from pathlib import Path
from helper.sentence_destructor import destruct_sentence

device = "cuda:0"

input_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/input_tokenizer.json"))
output_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/output_tokenizer.json"))

model = SpellCorrectorNet(
    encoder_config=LayerConfig(vocab_size=input_tokenizer.get_vocab_size(), d_model=128, n_layer=8, n_head=4),
    decoder_config=LayerConfig(vocab_size=output_tokenizer.get_vocab_size(), d_model=128, n_layer=8, n_head=4),
).to(device)

torch.save(model.state_dict(), str(Path(__file__).parent.parent / "checkpoints/model.pth"))



