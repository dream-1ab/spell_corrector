#/**
 #* @author Dream lab software technologies muhtarjaan mahmood (مۇختەرجان مەخمۇت)
 #* @email ug-project@outlook.com
 #* @create date 2025-05-12 10:01:07
 #* @modify date 2025-05-12 10:01:07
 #* @desc [description]
#*/

from torch import Tensor

class RegressiveBufferGenerator:
    def __init__(self, original_sentence_buffer: Tensor, lengths: list[int]):
        self.original_sentence_buffer = original_sentence_buffer
        self.lengths = lengths
        self.cursor_row = 0
        self.cursor_col = 0
        self.item_count = sum(lengths)

    def generate(self, target_buffer: Tensor, row_count: int) -> int:
        generated_count = 0
        max_rows = target_buffer.shape[0]

        while self.cursor_row < len(self.lengths):
            sentence_length = self.lengths[self.cursor_row]

            while self.cursor_col < sentence_length:
                prefix_length = self.cursor_col + 1
                value = self.original_sentence_buffer[self.cursor_row, :prefix_length]

                if generated_count >= row_count or generated_count >= max_rows:
                    return generated_count  # Stop early

                # Fill the next row in the target buffer
                target_buffer[generated_count, :prefix_length] = value
                generated_count += 1
                self.cursor_col += 1

            # Move to next sentence
            self.cursor_row += 1
            self.cursor_col = 0

        return generated_count

    def __len__(self):
        return self.item_count

    @staticmethod
    def calculate_sum_of(n: int):
        return int(n * (n + 1) / 2)