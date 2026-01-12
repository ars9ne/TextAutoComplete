import torch
import torch.nn as nn

class LSTMNextToken(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_id):
        super().__init__()
    
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embd = self.emb(x)
        output, _ = self.lstm(embd)
        logits = self.fc(output)
        return logits
    
    def evaluate(self, input_text, token2id, id_to_token, num_tokens=8):
        self.eval()
        words = input_text.strip().lower().split()
        unk_id = token2id["<unk>"]
        bos_id = token2id["<bos>"]
        eos_id = token2id["<eos>"]
        #prefix id = <bos> encode(text)
        ids = [bos_id] + [token2id.get(w, unk_id) for w in words]
        device = next(self.parameters()).device
        with torch.no_grad():
            for _ in range(num_tokens):
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                logits = self.forward(inp)
                # logits последнего токена
                last_logits = logits[0, -1, :]
                next_id = int(torch.argmax(last_logits).item())
                ids.append(next_id)
                # стоп, если eos
                if next_id == eos_id:
                    break
        #decode
        special = {"<pad>", "<unk>", "<bos>", "<eos>"}
        out_tokens = []
        for i in ids:
            tok = id_to_token[i] if 0 <= i < len(id_to_token) else "<unk>"
            if tok in special:
                continue
            out_tokens.append(tok)
        return " ".join(out_tokens)