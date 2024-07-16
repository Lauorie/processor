import re

class TextSplitter:
    def __init__(self, length):
        self.length = length

    def calculate_text_len(self, text):
        cleaned_text = re.sub(r'[\d!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)
        english_words = re.findall(r'[a-zA-Z]+', cleaned_text)
        english_word_count = len(english_words)
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', cleaned_text)
        chinese_character_count = len(chinese_characters)
        total_length = english_word_count + chinese_character_count
        return total_length

    def split_text(self, response_text):
        sentences = response_text.split('\n')
        result = []
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = self.calculate_text_len(sentence)

            if current_length + sentence_length > self.length:
                result.append(current_chunk)
                current_chunk = sentence
                current_length = sentence_length
            else:

                if current_chunk:  # 如果当前chunk不为空，添加一个换行符作为分隔
                    current_chunk += '\n'
                current_chunk += sentence
                current_length += sentence_length

        if current_chunk:
            result.append(current_chunk)

        return result