from tqdm import tqdm
import sys
from colorama import Fore

def offsets(token, text): #Slow_token
    offsets_span = []
    for i in token:
        if i == '<unk>': len_i = 1
        else: len_i = len(i.split("@")[0])

        if not offsets_span:
            offsets_span.append((0, len_i))
        else:
            len_whitespace = len(text[offsets_span[-1][1]:]) - len(text[offsets_span[-1][1]:].lstrip())
            if len_whitespace > 0:
                offsets_span.append((offsets_span[-1][1]+len_whitespace, offsets_span[-1][1]+len_whitespace + len_i))
            else:
                offsets_span.append((offsets_span[-1][1],offsets_span[-1][1] + len_i))
    return offsets_span
        

class Sample:
    def __init__(self, tokenizer, expansion, context, start_char_idx, len_acronym, label, max_seq_lenght=256):
        self.tokenizer = tokenizer #tokenizer BertWordPieceTokenizer
        self.expansion = expansion
        self.context = context
        self.start_char_idx = start_char_idx
        self.len_acronym = len_acronym
        # self.id = ids
        self.max_seq_lenght = max_seq_lenght
        self.skip = False
        
        self.start_token_idx = -1
        self.end_token_idx = -1

        self.label = int(label)
        
    def preprocess(self):
        tokenized_expansion = self.tokenizer(self.expansion)
        tokenized_context = self.tokenizer(self.context)
        
        end_char_idx = self.start_char_idx + self.len_acronym
        if end_char_idx > len(self.context): 
            self.skip = True
            return
        
        is_char_in_context = [0]*len(self.context)
        for idx in range(self.start_char_idx, end_char_idx):
            is_char_in_context[idx] = 1
        
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_context["input_ids"][1:-1])
        offsets_span = offsets(tokens, self.context)
        arc_token_idx  = []
        for idx, (start, end) in enumerate(offsets_span):
            if sum(is_char_in_context[start:end]) > 0: arc_token_idx.append(idx)
        if len(arc_token_idx) == 0:
            self.skip = True
            return
        self.start_token_idx = arc_token_idx[0] + 1
        self.end_token_idx = arc_token_idx[-1] + 1

        # ###################### START AND END TOKEN #############################
        # tokenized_context_ids = tokenized_context.ids
        # tokenized_context_ids.insert(self.start_token_idx, 30522)
        # tokenized_context_ids.insert(self.end_token_idx+2, 30523)

        # self.start_token_idx += 1
        # self.end_token_idx += 1
        
        # ######################################################################

        input_ids = tokenized_context["input_ids"] + tokenized_expansion["input_ids"][1:]
        token_type_ids = tokenized_context["token_type_ids"] + [1]*len(tokenized_expansion["token_type_ids"][1:])
        attention_mask = [1] * len(input_ids)
        
        
        padding_length = self.max_seq_lenght - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0]* padding_length)
            token_type_ids = token_type_ids + ([0]* padding_length)
            attention_mask = attention_mask + ([0]* padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask


def create_examples(raw_data, desc, tokenizer):
    p_bar = tqdm(total=len(raw_data), desc=desc,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    examples = []
    for item in raw_data:
        expansion = item["expansion"]
        context = item["text"]
        start_char_idx = item["start_char_idx"]
        length_acronym = item["length_acronym"]
        label = item["label"]
        # ids  = item["id"]
        example = Sample(tokenizer, expansion, context, start_char_idx, length_acronym, label)
        example.preprocess()
        examples.append(example)
        p_bar.update(1)
    p_bar.close()
    return examples