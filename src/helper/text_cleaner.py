#/**
# * @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
# * @email ug-project@outlook.com
# * @create date 2025-05-04 22:29:51
# * @modify date 2025-05-04 22:29:51
# * @desc [description]
#*/

uyghur_symbols = {
  "ھەرىپلەر": [
    'ا', 'ە', 'ب', 'پ', 'ت', 'ج', 'چ', 'خ', 'د', 'ر', 'ز', 'ژ', 'س', 'ش', 'غ', 'ف',
    'ق', 'ك', 'گ', 'ڭ', 'ل', 'م', 'ن', 'ھ', 'و', 'ۇ', 'ۆ', 'ۈ', 'ۋ', 'ې', 'ى', 'ي',
    "ئ"
  ],
  "تىنىش_بەلگىلىرى": [
    "۔", "،", "؟", "!", "-", "«", "»", "؛", ":", "'", "\"", "]", "[", " ", "›", "‹"
  ],
  "سانلار": [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  ],
}

english_symbols = {
  "characters": [
    # 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
  ],
  "symbols": [
    ".", "!", "?", "-", "<", ">", "=", "+", "*", "/", "|", "\\", "~", "_", "^", "@", "{", "}", "[", "]", "(", ")", "#", "%", "$", "€", "£", "¥", " "
  ]
}

# Export all symbols:
symbols_list = uyghur_symbols["ھەرىپلەر"] + uyghur_symbols["تىنىش_بەلگىلىرى"] + uyghur_symbols["سانلار"] + english_symbols["characters"] + english_symbols["symbols"]
symbols = {i: i for i in symbols_list}

def clean_unknown_symbols(text: str) -> str:
  return ''.join(c for c in text if c in symbols)

def clean_rare_symbols(text: str) -> str:
    rare_symbols = {
        '◆', '℃', 'Ｃ', 'Ｓ', 'ラ', 'ド', '٪', 'Ａ', '⑤', '）', '○', '“', '％',
        '（', 'Ｎ', '●', '④', 'ä', '▲', 'Ⅳ', '–', 'Ｄ', '－', 'Ⅱ', '①', '③',
        '△', '。', '\ufeff', '\x7f', '…', 'β', '’', '—', '②', '：', 'ト', 'é',
        '−', 'ル', '、', '\u3000', '⑥', '~'
    }

    return ''.join(c for c in text if c not in rare_symbols)


# def clean_unknown_text(text: str) -> str:
#    constructed = 
def normalize_extended_uyghur_characters(text: str) -> str:
    characters = [
      #["standard", "individual", "beginning", "middle", "end"]
      ["ئ","ﺋ","ﺋ","ﺌ","ﺌ"],
      ["ا","ﺍ","ﺍ","ﺎ","ﺎ"],
      ["ە","ﻩ","ﻩ","ﻪ","ﻪ"],
      ["ب","ﺏ","ﺑ","ﺒ","ﺐ"],
      ["پ","ﭖ","ﭘ","ﭙ","ﭗ"],
      ["ت","ﺕ","ﺗ","ﺘ","ﺖ"],
      ["ج","ﺝ","ﺟ","ﺠ","ﺞ"],
      ["چ","ﭺ","ﭼ","ﭽ","ﭻ"],
      ["خ","ﺥ","ﺧ","ﺨ","ﺦ"],
      ["د","ﺩ","ﺩ","ﺪ","ﺪ"],
      ["ر","ﺭ","ﺭ","ﺮ","ﺮ"],
      ["ز","ﺯ","ﺯ","ﺰ","ﺰ"],
      ["ژ","ﮊ","ﮊ","ﮋ","ﮋ"],
      ["س","ﺱ","ﺳ","ﺴ","ﺲ"],
      ["ش","ﺵ","ﺷ","ﺸ","ﺶ"],
      ["غ","ﻍ","ﻏ","ﻐ","ﻎ"],
      ["ف","ﻑ","ﻓ","ﻔ","ﻒ"],
      ["ق","ﻕ","ﻗ","ﻘ","ﻖ"],
      ["ك","ﻙ","ﻛ","ﻜ","ﻚ"],
      ["گ","ﮒ","ﮔ","ﮕ","ﮓ"],
      ["ڭ","ﯓ","ﯕ","ﯖ","ﯔ"],
      ["ل","ﻝ","ﻟ","ﻠ","ﻞ"],
      ["م","ﻡ","ﻣ","ﻤ","ﻢ"],
      ["ن","ﻥ","ﻧ","ﻨ","ﻦ"],
      ["ھ","ﮪ","ﮬ","ﮭ","ﮫ"],
      ["و","ﻭ","ﻭ","ﻮ","ﻮ"],
      ["ۇ","ﯗ","ﯗ","ﯘ","ﯘ"],
      ["ۆ","ﯙ","ﯙ","ﯚ","ﯚ"],
      ["ۈ","ﯛ","ﯛ","ﯜ","ﯜ"],
      ["ۋ","ﯞ","ﯞ","ﯟ","ﯟ"],
      ["ې","ﯤ","ﯦ","ﯧ","ﯥ"],
      ["ى","ﻯ","ﯨ","ﯩ","ﻰ"],
      ["ي","ﻱ","ﻳ","ﻴ","ﻲ"],
      ["ۅ","ﯠ","ﯠ","ﯡ","ﯡ"],
      ["ۉ","ﯢ","ﯢ","ﯣ","ﯣ"],
      ["ح","ﺡ","ﺣ","ﺤ","ﺢ"],
      ["ع","ﻉ","ﻋ","ﻌ","ﻊ"]
    ]
    replacement_table: dict[str, str] = {}
    #Create a replacement table.
    for char_map in characters:
      for char in char_map[1:]:
        replacement_table[char] = char_map[0]
    
    replacement_table['ﻼ'] = "لا" #add some additional exceptional symbols

    clean = ""
    #Replace the extended characters with their standard unicode ones.
    for char in text:
      if char in replacement_table:
        char = replacement_table[char]
      clean += char
    
    return clean

import re
def clean_chinese_text(text: str) -> str:
   return re.sub(r'[\u4e00-\u9fff]', '', text)

def clean_http_links(text: str) -> str:
   return re.sub(r'https?://[^\s]+', '', text)

def collapse_spaces(text: str) -> str:
   return re.sub(r'\s+', ' ', text)

def lower_text(text: str) -> str:
   return text.lower()

def clean_input_text(text: str) -> str:
   text = clean_rare_symbols(text)
   text = normalize_extended_uyghur_characters(text)
   text = clean_chinese_text(text)
   text = clean_unknown_symbols(text)
   text = clean_http_links(text)
   text = collapse_spaces(text)
   text = lower_text(text)
   return text