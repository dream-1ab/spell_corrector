#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-09 23:31:14
# * @modify date 2025-05-09 23:31:14
# * @desc [description]
#*/


from torch.nn import Linear, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, CrossEntropyLoss, Module
from torch.optim import Adam


class SpellCorrectorNet(Module):
    def __init__(self, n_input_vocab: int, n_output_vocab: int):
        super().__init__()
