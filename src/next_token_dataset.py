import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(self, train_path, token2id, max_len):
        self.token2id = token2id
        self.max_len = max_len
        self.pad_id = token2id["<pad>"]
        self.unk_id = token2id["<unk>"]
        self.bos_id = token2id["<bos>"]
        self.eos_id = token2id["<eos>"]

        with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
            self.texts = [line.strip() for line in f if line.strip()] #загружаем файл как список строк

    def __len__(self):
        return len(self.texts)

    def _encode(self, text):
        words = text.split()
        return [self.token2id.get(w, self.unk_id) for w in words] # слово>id, если нет в словаре то <unk>

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = [self.bos_id] + self._encode(text) + [self.eos_id]

        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len - 1] + [self.eos_id] #гарантия того что <eos> остаётся
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        return torch.LongTensor(input_ids), torch.LongTensor(target_ids)



def collate_fn(batch, pad_id, return_attention_mask=False):
    B_size = len(batch)
    # длины последовательностей в батче
    lengths = [x[0].shape[0] for x in batch]
    T = max(lengths) #максимальная длинна батча
    #padded тензоры
    input_ids_padded = torch.full((B_size, T), pad_id, dtype=torch.long)
    target_ids_padded = torch.full((B_size, T), pad_id, dtype=torch.long)
    if return_attention_mask: # attention_mask реализуем как опцию
        attention_mask = torch.zeros((B_size, T), dtype=torch.long)
    else:
        attention_mask = None

    for i, (inp, tgt) in enumerate(batch):
        t = inp.shape[0]
        input_ids_padded[i, :t] = inp
        target_ids_padded[i, :t] = tgt
        if return_attention_mask:
            attention_mask[i, :t] = 1

    if return_attention_mask:
        return input_ids_padded, target_ids_padded, attention_mask
    return input_ids_padded, target_ids_padded
