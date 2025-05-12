#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-11 15:26:02
# * @modify date 2025-05-11 15:26:02
# * @desc [description]
#*/
from tokenizers import Tokenizer
import random

def destruct_sentence(tokenizer: Tokenizer, text: str, level: float) -> str:
    """
    Destruct the input sentence by randomly adding, removing, or replacing characters.
    
    Args:
        tokenizer (Tokenizer): A HuggingFace Tokenizer object.
        text (str): The input sentence to destruct.
        level (float): Destruction level between 0.0 and 1.0. 
                       Represents the percentage of characters to corrupt.
    
    Returns:
        str: Corrupted sentence.
    """
    assert 0.0 <= level <= 1.0, "level must be between 0.0 and 1.0"

    chars = list(text)
    num_changes = max(1, int(len(chars) * level))
    ops = ['replace', 'insert', 'delete']
    
    for _ in range(num_changes):
        if not chars:
            break  # nothing to corrupt
        op = random.choice(ops)
        idx = random.randint(0, len(chars) - 1)
        all_chars = list(tokenizer.get_vocab().keys())
        all_chars.remove("<SOS>")
        all_chars.remove("<EOS>")
        all_chars.remove("<PAD>")
        all_chars.remove("<UNK>")
        random_char: str = random.choice(list(all_chars))
        
        if op == 'replace':
            chars[idx] = random_char
        elif op == 'insert':
            chars.insert(idx, random_char)
        elif op == 'delete' and len(chars) > 1:
            del chars[idx]

    return ''.join(chars)