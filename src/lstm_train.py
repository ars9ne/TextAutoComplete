import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.eval_lstm import evaluate_rouge, generate_last_quarter


def train_one_epoch(model, loader, optimizer, criterion, pad_id, device):
    model.train()
    sum_loss = 0
    #проход по батчам
    for input_ids, target_ids in tqdm(loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids) #forward
        B, T, V = logits.shape #размер батча, длина последовательности, размер словаря
        loss = criterion(logits.reshape(B*T, V), target_ids.reshape(B*T))
        loss.backward() #backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #предотвращаем взрыв градиентов
        optimizer.step() #обновление весов
        sum_loss += loss.item()

    return sum_loss / len(loader) #средний loss по эпохе



def evaluate_loss(model, loader, criterion, pad_id, device):
    with torch.no_grad():
        model.eval()
        sum_loss = 0.0

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B*T, V), target_ids.reshape(B*T))
            sum_loss += loss.item()

        return sum_loss / len(loader) #средний loss по вал. батчам



def show_examples(model, loader, token2id, id2token, device, n_examples=3):
    with torch.no_grad():
        model.eval()
        pad_id = token2id["<pad>"]

        shown = 0
        for input_ids, target_ids in loader:
            lengths = (input_ids != pad_id).long().sum(dim=1).tolist()
            for i, L in enumerate(lengths): #пример по 1 батчу
                if shown >= n_examples:
                    return
                seq = input_ids[i, :L]
                res = generate_last_quarter(model, seq, token2id, id2token, q=0.25)
                if res is None: #если не получилось сгенерить
                    continue
                prompt_text, target_text, pred_text = res
                print("-------Пример----------")
                print("PROMPT:", prompt_text)
                print("TARGET:", target_text)
                print("PRED  :", pred_text)
                shown += 1
            break


def train_model(model, train_loader, val_loader, token2id, id2token,
                n_epochs=5, lr=1e-3, save_path="models/lstm_best.pth",
                rouge_batches=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    pad_id = token2id["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id) #функция потерь без учета pad
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#AdamW оптимизатор

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_rouge2 = -1

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, pad_id, device)
        val_loss = evaluate_loss(model, val_loader, criterion, pad_id, device)
        rouge1, rouge2 = evaluate_rouge(model, val_loader, token2id, id2token, max_batches=rouge_batches, q=0.25)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val r-1: {rouge1:.4f} | Val r-2: {rouge2:.4f}")

        #сохранение лучшей модели по r2
        if rouge2 > best_rouge2:
            best_rouge2 = rouge2
            torch.save(model.state_dict(), save_path)
        
        #показ примеров
        if epoch == 0 or epoch == n_epochs - 1:
            show_examples(model, val_loader, token2id, id2token, device, n_examples=3)
    print("Best val Rouge2:", best_rouge2)


if __name__ == "__main__":
    pass