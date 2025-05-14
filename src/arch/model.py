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

class LayerConfig(TypedDict):
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int

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

        #common for encoder and decoder.
        self.position_encoder = positional_encoding.PositionalEncoding(d_model=encoder_config["d_model"])
        #Encoders part.
        self.encoder_embedding = Embedding(num_embeddings=encoder_config["vocab_size"], embedding_dim=encoder_config["d_model"])
        self.encoder_layer = TransformerEncoderLayer(d_model=encoder_config["d_model"], nhead=encoder_config["n_head"], dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=encoder_config["n_layer"])
        #MLP layers for encoder-decoder memory d_model linear transform.
        self.encoder_decoder_linear_transformer = LinearTransformer(encoder_d_model=encoder_config["d_model"], decoder_d_model=decoder_config["d_model"])
        #decoder part.
        self.decoder_embedding = Embedding(num_embeddings=decoder_config["vocab_size"], embedding_dim=decoder_config["d_model"])
        self.decoder_layer = TransformerDecoderLayer(d_model=decoder_config["d_model"], nhead=decoder_config["n_head"], dropout=0.1, activation="gelu", batch_first=True)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=decoder_config["n_layer"])
        self.decoder_layer_normalizer = LayerNorm(decoder_config["d_model"])
        #vocabulary classification part.
        self.decoder_token_classification = TokenClassification(decoder_d_model=decoder_config["d_model"], vocab_size=decoder_config["vocab_size"])
    
    def generate_memory(self, x: Tensor) -> Tensor:
        """
        there should be never mask applied to the input tensor x.
        also there is no reason to apply a padding mask to input x.
        """
        encoder_key_padding_mask = x == 0 #assume 0 meaning becomes <PAD> token.
        #shape of x gonna be [batch_size, sequence_length]
        x = self.encoder_embedding(x)
        #shape of x gonna be [batch_size, sequence_length, encoder_d_model]
        x = self.position_encoder(x)
        encoder_mask = torch.zeros((x.shape[1], x.shape[1]), device=x.device).bool()
        x = self.encoder(src=x, mask=encoder_mask, src_key_padding_mask=encoder_key_padding_mask)
        #now x becomes memory of decoder.
        #shape of x still gonna be [batch_size, sequence_length, encoder_d_model]
        x = self.encoder_decoder_linear_transformer(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        return x
    
    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        target_key_padding_mask = x == 0
        #shape of x gonna be [batch_size, sequence_length]
        x = self.decoder_embedding(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        x = self.position_encoder(x)
        target_causal_mask = torch.triu(torch.ones((x.shape[1], x.shape[1]), device=x.device), diagonal=1).bool()
        memory_causal_mask = torch.zeros(x.shape[1], memory.shape[1], device=x.device)
        memory_key_padding_mask = torch.zeros(memory.shape[0], memory.shape[1], device=x.device)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=target_causal_mask, memory_mask=memory_causal_mask, tgt_key_padding_mask=target_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        x = self.decoder_layer_normalizer(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_d_model]
        x = self.decoder_token_classification(x)
        #shape of x gonna be [batch_size, sequence_length, decoder_vocab_size]
        return x


