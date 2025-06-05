#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-11 15:26:02
# * @modify date 2025-05-11 15:26:02
# * @desc [description]
#*/
from tokenizers import Tokenizer
import random
import re

# Common Uyghur confusable character pairs
CONFUSABLE_CHARS = [
    ('ا', 'ە'),  # alef and e
    ('ې', 'ى'),  # e and i - commonly confused 
    ('ى', 'ي'),  # i and y
    # Group similar-looking vowels together
    ('و', 'ۇ', 'ۆ', 'ۈ'),  # o, u, ö, ü - commonly confused due to similar appearance
    ('ھ', 'خ'),  # h and kh
    ('ق', 'غ', 'ف'),  # q, gh, and f - commonly confused
    ('ك', 'گ', 'ڭ'),  # k, g and ng
    ('ز', 'ژ', 'ر'),  # z, zh and r
    ('س', 'ش'),  # s and sh - commonly confused
    ('ش', 'چ'),  # sh and ch
    ('ج', 'چ'),  # j and ch - commonly confused
    ("و", "ا"),  # o and a - some people misuse them because of dialect

    #some words are only typed while pressing the shift key.
    #they are:
    ("د", "ت"),
    ("ف", "ا"),
    ("گ", "ە"),
    ("خ", "ى"),
    ("ج", "ق"),
    ("ك", "ۆ"),
]

# Common Uyghur suffix errors
COMMON_SUFFIXES = [
    ('لار', 'لەر'),  # plural suffixes
    ('دە', 'تە'),    # locative suffixes
    ('دا', 'تا'),    # locative suffixes
    ('دىن', 'تىن'),  # ablative suffixes
    ('غا', 'قا'),    # dative suffixes
    ('گە', 'كە'),    # dative suffixes
    ('نىڭ', 'نى'),   # genitive and accusative confusion
    ('لىق', 'لىك'),  # adjectival suffixes
]

def destruct_sentence(tokenizer: Tokenizer, text: str, level: float) -> str:
    """
    Destruct the input sentence by randomly adding, removing, or replacing characters
    with Uyghur-specific error patterns. Multiple error types are applied simultaneously.
    
    Args:
        tokenizer (Tokenizer): A HuggingFace Tokenizer object.
        text (str): The input sentence to destruct.
        level (float): Destruction level between 0.0 and 1.0. 
                       Represents the percentage of characters to corrupt.
    
    Returns:
        str: Corrupted sentence.
    """
    assert 0.0 <= level <= 1.0, "level must be between 0.0 and 1.0"
    
    # Apply multiple error types with probability weights
    corrupted_text = text
    
    # Apply space-related errors - very common
    if random.random() < 0.80:  # 80% chance
        corrupted_text = modify_spaces(corrupted_text, level * random.uniform(0.3, 0.8))
    
    # Apply omitting ى character - very common (highest priority)
    if random.random() < 0.9:  # 90% chance
        corrupted_text = omit_i_character(corrupted_text, level * random.uniform(1, 5))
    
    # Apply confusable character replacements - also very common
    if random.random() < 0.8:  # 80% chance
        corrupted_text = replace_with_confusable_chars(corrupted_text, level * random.uniform(0.1, 0.8))
    
    # Apply suffix errors - fairly common
    if random.random() < 0.6:  # 60% chance
        corrupted_text = introduce_suffix_errors(corrupted_text, level * random.uniform(0.3, 0.7))
    
    # Apply vowel harmony breakage - less common
    if random.random() < 0.6:  # 60% chance
        corrupted_text = break_vowel_harmony(corrupted_text, level * random.uniform(0.2, 0.6))
    
    # Apply character transposition - less common
    if random.random() < 0.8:  # 80% chance
        corrupted_text = transpose_characters(corrupted_text, level * random.uniform(0.2, 0.8))
    
    # Apply basic character operations as fallback if no changes were made
    if corrupted_text == text:
        corrupted_text = basic_character_operations(tokenizer, text, level)
    
    return corrupted_text

def basic_character_operations(tokenizer: Tokenizer, text: str, level: float) -> str:
    """Original character-level operations (replace, insert, delete)"""
    chars = list(text)
    num_changes = max(1, int(len(chars) * level))
    ops = ['replace', 'insert', 'delete']
    
    for _ in range(num_changes):
        if not chars:
            break  # nothing to corrupt
        op = random.choice(ops)
        idx = random.randint(0, len(chars) - 1)
        all_chars = list(tokenizer.get_vocab().keys())
        if "<SOS>" in all_chars:
            all_chars.remove("<SOS>")
        if "<EOS>" in all_chars:
            all_chars.remove("<EOS>")
        if "<PAD>" in all_chars:
            all_chars.remove("<PAD>")
        if "<UNK>" in all_chars:
            all_chars.remove("<UNK>")
        if "<MASK>" in all_chars:
            all_chars.remove("<MASK>")
        random_char: str = random.choice(list(all_chars))
        
        if op == 'replace':
            chars[idx] = random_char
        elif op == 'insert':
            chars.insert(idx, random_char)
        elif op == 'delete' and len(chars) > 1:
            del chars[idx]

    return ''.join(chars)

def replace_with_confusable_chars(text: str, level: float) -> str:
    """Replace characters with commonly confused ones in Uyghur"""
    chars = list(text)
    num_changes = max(1, int(len(chars) * level))
    
    # Create lookup for quick checks
    confusable_lookup = {}
    for group in CONFUSABLE_CHARS:
        for char in group:
            confusable_lookup[char] = [c for c in group if c != char]
    
    changes_made = 0
    # Try to make the specified number of changes
    for _ in range(min(num_changes * 3, len(chars))):  # Try more times than needed to ensure we get enough changes
        if changes_made >= num_changes:
            break
            
        idx = random.randint(0, len(chars) - 1)
        char = chars[idx]
        
        if char in confusable_lookup and confusable_lookup[char]:
            chars[idx] = random.choice(confusable_lookup[char])
            changes_made += 1
    
    return ''.join(chars)

def introduce_suffix_errors(text: str, level: float) -> str:
    """Introduce errors in Uyghur suffixes"""
    # Split into words
    words = text.split()
    num_changes = max(1, int(len(words) * level))
    
    for _ in range(min(num_changes, len(words))):
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        # Check if word ends with any of our target suffixes
        for correct, misspelled in COMMON_SUFFIXES:
            if word.endswith(correct):
                # Replace correct suffix with misspelled one
                words[word_idx] = word[:-len(correct)] + misspelled
                break
            elif word.endswith(misspelled):
                # Replace misspelled suffix with correct one
                words[word_idx] = word[:-len(misspelled)] + correct
                break
    
    return ' '.join(words)

def transpose_characters(text: str, level: float) -> str:
    """Swap adjacent characters - a common typing error"""
    chars = list(text)
    num_changes = max(1, int(len(chars) * level) // 2)  # Divide by 2 since each swap affects 2 characters
    
    for _ in range(min(num_changes, len(chars) - 1)):
        idx = random.randint(0, len(chars) - 2)
        # Swap characters
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    
    return ''.join(chars)

def break_vowel_harmony(text: str, level: float) -> str:
    """Break Uyghur vowel harmony rules"""
    # Uyghur vowels
    front_vowels = set(['ە', 'ۆ', 'ۈ', 'ې'])
    back_vowels = set(['ا', 'و', 'ۇ', 'ى'])
    
    # Split into words
    words = text.split()
    num_changes = max(1, int(len(words) * level))
    
    for _ in range(min(num_changes, len(words))):
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        # Check if word is long enough and has vowels to manipulate
        if len(word) < 3:
            continue
            
        word_chars = list(word)
        vowel_positions = [i for i, c in enumerate(word_chars) if c in front_vowels or c in back_vowels]
        
        if len(vowel_positions) < 2:
            continue
            
        # Choose a vowel position to replace
        vowel_pos = random.choice(vowel_positions)
        current_vowel = word_chars[vowel_pos]
        
        # Replace with opposite harmony vowel
        if current_vowel in front_vowels:
            # Replace front vowel with corresponding back vowel
            if current_vowel == 'ە':
                word_chars[vowel_pos] = 'ا'
            elif current_vowel == 'ۆ':
                word_chars[vowel_pos] = 'و'
            elif current_vowel == 'ۈ':
                word_chars[vowel_pos] = 'ۇ'
            elif current_vowel == 'ې':
                word_chars[vowel_pos] = 'ى'
        else:
            # Replace back vowel with corresponding front vowel
            if current_vowel == 'ا':
                word_chars[vowel_pos] = 'ە'
            elif current_vowel == 'و':
                word_chars[vowel_pos] = 'ۆ'
            elif current_vowel == 'ۇ':
                word_chars[vowel_pos] = 'ۈ'
            elif current_vowel == 'ى':
                word_chars[vowel_pos] = 'ې'
                
        words[word_idx] = ''.join(word_chars)
    
    return ' '.join(words)

def omit_i_character(text: str, level: float) -> str:
    """
    Simulates a common error in Uyghur writing where the character ى (i) is omitted.
    For example: بىلدىم -> بىلدم
    """
    chars = list(text)
    
    # Find positions of ى character
    i_positions = [idx for idx, char in enumerate(chars) if char == 'ى']
    if not i_positions:
        return text  # No ى characters to omit
    
    # Calculate how many ى characters to omit based on level
    num_to_omit = max(1, int(len(i_positions) * level))
    # print(num_to_omit)
    
    # Randomly select positions to omit
    positions_to_omit = random.sample(i_positions, min(num_to_omit, len(i_positions)))
    # Create a new list without the omitted characters
    result = []
    for idx, char in enumerate(chars):
        if idx not in positions_to_omit:
            result.append(char)
    
    return ''.join(result)

def modify_spaces(text: str, level: float) -> str:
    """
    Modify spaces in text by either removing necessary spaces or adding 
    unnecessary spaces. This simulates common typing errors in Uyghur text.
    
    Args:
        text (str): The input text.
        level (float): Error level between 0.0 and 1.0.
        
    Returns:
        str: Text with modified spaces.
    """
    if not text or level <= 0:
        return text
    
    words = text.split()
    if len(words) <= 1:
        return text  # Need at least two words to modify spaces
    
    result = []
    num_spaces_to_modify = max(1, int((len(words) - 1) * level))
    
    # Decide whether to remove spaces or add extra spaces
    remove_spaces = random.random() < 0.70  # 70% chance to remove spaces, 40% to add spaces
    
    if remove_spaces:
        # Remove spaces by joining words
        space_positions = list(range(len(words) - 1))
        positions_to_modify = random.sample(space_positions, min(num_spaces_to_modify, len(space_positions)))
        
        i = 0
        while i < len(words):
            if i in positions_to_modify:
                # Join this word with the next word
                result.append(words[i] + words[i+1])
                i += 2
            else:
                result.append(words[i])
                i += 1
    else:
        # Add extra spaces within words
        for word in words:
            if len(word) >= 3 and random.random() < level:
                # Add a space somewhere in the middle of the word
                pos = random.randint(1, len(word) - 1)
                result.append(word[:pos] + ' ' + word[pos:])
            else:
                result.append(word)
    
    return ' '.join(result)

def mask_sentence(sentence: str, threshold: float) -> str:
    #first split the sentence into words and mask the words with <MASK> token
    words = sentence.split()
    masked_words = []
    for word in words:
        if random.random() < threshold:
            masked_words.append("<MASK>")
        else:
            masked_words.append(word)
    return ' '.join(masked_words)