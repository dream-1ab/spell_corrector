#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-10 13:20:25
# * @modify date 2025-05-10 13:20:25
# * @desc [description]
#*/

import torch
from arch.model import SpellCorrectorNet, LayerConfig
from tokenizers import Tokenizer, decoders, pre_tokenizers
from pathlib import Path
from helper.sentence_destructor import destruct_sentence

device = "cuda:1"

input_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/input_tokenizer.json"))
output_tokenizer: Tokenizer = Tokenizer.from_file(str(Path(__file__).parent.parent / "config/output_tokenizer.json"))
output_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
output_tokenizer.decoder = decoders.Metaspace()
# output_tokenizer = input_tokenizer

model = torch.compile(SpellCorrectorNet(
    encoder_config=LayerConfig(vocab_size=input_tokenizer.get_vocab_size(), d_model=256, n_layer=6, n_head=8),
    decoder_config=LayerConfig(vocab_size=output_tokenizer.get_vocab_size(), d_model=256, n_layer=4, n_head=8),
).to(device))


model.load_state_dict(torch.load("checkpoints/model.pth"))


while True:
    text = input("/>")
    generated = model.generate_text(text, input_tokenizer, output_tokenizer, device, max_length=1024)
    # generated = model.generate("تەرخەمە", input_tokenizer, output_tokenizer, device)
    print(f"{generated}\n\n")