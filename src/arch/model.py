#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-09 23:31:14
# * @modify date 2025-05-09 23:31:14
# * @desc [description]
#*/


from torch.nn import Linear, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, CrossEntropyLoss, Module, Embedding, Sequential, GLU, LayerNorm, GELU, ReLU
from torch.optim import Adam
import arch.positional_encoding as positional_encoding
from typing import TypedDict
from torch import Tensor, tensor
import torch
from tokenizers import Tokenizer
import math

class LayerConfig(TypedDict):
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a [max_len, d_model] matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Not a parameter, but persistent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of same shape with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class LinearTransformer(Module):
    def __init__(self, encoder_d_model: int, decoder_d_model: int):
        super().__init__()
        self.activation = ReLU()
        self.layer0 = Linear(encoder_d_model, out_features=encoder_d_model)
        self.layer1 = Linear(encoder_d_model, out_features=decoder_d_model)
        self.layer2 = Linear(decoder_d_model, out_features=decoder_d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        #x: shape of [batch_size, sentence_length, encoder_d_model]
        x = self.layer0(x)
        x = self.activation(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        #x: shape of [batch_size, sentence_length, decoder_d_model]
        return x

class TokenClassification(Module):
    def __init__(self, decoder_d_model: int, vocab_size: int):
        super().__init__()
        self.activation = ReLU()
        self.layer1 = Linear(in_features=decoder_d_model, out_features=decoder_d_model)
        self.layer2 = Linear(in_features=decoder_d_model, out_features=vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class SpellCorrectorNet(Module):
    def __init__(self, encoder_config: LayerConfig, decoder_config: LayerConfig):
        """
        the max_sentence_length parameter is used to pre-generate causal mask, key padding mask to avoid memory allocation overhead to improve training and inference performance but may takes a little space in model file.
        """
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        #common for encoder and decoder.
        self.encoder_position_encoder = PositionalEncoding(d_model=encoder_config["d_model"], max_len=4096)
        #Encoders part.
        self.encoder_embedding = Embedding(num_embeddings=encoder_config["vocab_size"], embedding_dim=encoder_config["d_model"], padding_idx=0)
        self.encoder_layer = TransformerEncoderLayer(d_model=encoder_config["d_model"], nhead=encoder_config["n_head"], dropout=0.2, activation="gelu", batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=encoder_config["n_layer"])
        #MLP layers for encoder-decoder memory d_model linear transform.
        self.encoder_decoder_linear_transformer = LinearTransformer(encoder_d_model=encoder_config["d_model"], decoder_d_model=decoder_config["d_model"])
        #decoder part.
        self.decoder_position_encoder = PositionalEncoding(d_model=decoder_config["d_model"], max_len=4096)
        self.decoder_embedding = Embedding(num_embeddings=decoder_config["vocab_size"], embedding_dim=decoder_config["d_model"], padding_idx=0)
        self.decoder_layer = TransformerDecoderLayer(d_model=decoder_config["d_model"], nhead=decoder_config["n_head"], dropout=0.2, activation="gelu", batch_first=True)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=decoder_config["n_layer"])
        self.decoder_layer_normalizer = LayerNorm(decoder_config["d_model"])
        #vocabulary classification part.
        self.decoder_token_classification = TokenClassification(decoder_d_model=decoder_config["d_model"], vocab_size=decoder_config["vocab_size"])
    
    def generate_memory(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        there should be never mask applied to the input tensor x.
        also there is no reason to apply a padding mask to input x.
        """
        encoder_key_padding_mask = (x == 0)  #assume 0 meaning becomes <PAD> token, Boolean mask where True=masked
        #shape of x gonna be [batch_size, sequence_length]
        x = self.encoder_embedding(x) * math.sqrt(self.encoder_config["d_model"])
        #shape of x gonna be [batch_size, sequence_length, encoder_d_model]
        x = self.encoder_position_encoder(x)
        # No need for encoder_mask as each token should attend to all other tokens
        x = self.encoder(src=x, src_key_padding_mask=encoder_key_padding_mask)
        #now x becomes memory of decoder.
        #shape of x still gonna be [batch_size, sequence_length, encoder_d_model]
        x = self.encoder_decoder_linear_transformer(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        return x, encoder_key_padding_mask
    
    def forward(self, x: Tensor, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        target_key_padding_mask = (x == 0)  # Boolean mask where True=masked positions
        #shape of x gonna be [batch_size, sequence_length]
        x = self.decoder_embedding(x) * math.sqrt(self.decoder_config["d_model"])
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        x = self.decoder_position_encoder(x)
        
        # Create proper causal mask for decoder self-attention (float tensor with -inf)
        seq_len = x.shape[1]
        target_causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) == 1.0
        
        # No need for memory causal mask - each target position should attend to all memory positions
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=target_causal_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        x = self.decoder_layer_normalizer(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        x = self.decoder_token_classification(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_vocab_size]
        return x
    
    def generate_text(self, text: str, input_tokenizer: Tokenizer, output_tokenizer: Tokenizer, device="cuda:0", max_length: int | None = None) -> str:
        if len(text) == 0:
            return ""
        self.eval()
        if max_length == None:
            max_length = len(text) * 2
        token_ids: list[int] = input_tokenizer.encode(f"<SOS>{text}<EOS>").ids
        x = torch.tensor(token_ids, dtype=torch.long, device=device).reshape(1, -1)
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.float16):
                memory, memory_key_padding_mask = self.generate_memory(x)
        sos_token = output_tokenizer.encode("<SOS>").ids[0]
        eos_token = output_tokenizer.encode("<EOS>").ids[0]
        eos_token: int
        sos_token: int

        buffer = torch.zeros(1, max_length, dtype=torch.long, device=device)
        position = 0
        buffer[-1][position] = sos_token
        position += 1
        while True:
            with torch.no_grad():
                with torch.autocast(device, dtype=torch.float16):
                    y: Tensor = self(buffer[:, :position], memory, memory_key_padding_mask)
            # torch.topk(y[-1, -1], 5).
            next_token: int = torch.argmax(y[-1, -1]).item()
            buffer[-1, position] = next_token
            position += 1
            if next_token == eos_token:
                # print("hit of <EOS>")
                break
            if position == max_length:
                break
        output_token_ids = buffer[-1, :position].tolist()
        text: str = output_tokenizer.decode(output_token_ids, skip_special_tokens=False)
        return text

