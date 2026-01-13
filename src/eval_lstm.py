import torch
from collections import Counter
from src.tokenizer_utils import decode

def rouge_f1(ref_text, pred_text, n=1):
    ref = [w for w in ref_text.strip().split() if w] #спиоск слов эталлонго текста
    pred = [w for w in pred_text.strip().split() if w]#список слов предсказанного текста

    if len(ref) < n or len(pred) < n:
        return 0
    
    def ngrams(tokens):
        # список всех последовательностей из n слов
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    #сколько раз встречается n-грамма в reference
    ref_ngr = Counter(ngrams(ref))
    #сколько раз встречается в prediction
    pred_ngr = Counter(ngrams(pred))

    overl = 0 #счетчик совпадений

    for g, c in pred_ngr.items():
        #считаем пересечение путем выбора минимума между кол-вом в pred и ref
        overl += min(c, ref_ngr.get(g,0))
    #precision = колво совпадений/ колво n_gram в предсказании
    prec = overl / max(1, sum(pred_ngr.values()))
    
    #recall = кол-во совпадений/колво n_gram в reference
    rec = overl / max(1, sum(ref_ngr.values()))
    d = (prec+ rec)
    if d == 0:
        return 0
    return(2*prec * rec / (prec + rec)) #F1

def greedy_generate(model, prompt_ids, eos_id, max_new_tokens, device):
    with torch.no_grad():
        ids = prompt_ids.tolist()
        new_ids = []

        for _ in range(max_new_tokens):
            model_input = torch.tensor([ids], dtype=torch.long, device=device) #вход модели
            logits = model(model_input) #прогоняем через модель
            next_id = int(torch.argmax(logits[0, -1]).item()) #получаем самый вероятный токен
            ids.append(next_id) #добавляем к текущей последовательности
            if next_id == eos_id: #если сгенерирован end of seq - конец
                break
            new_ids.append(next_id)
        return new_ids #возвращаем список сгенерированных id токенов

def generate_last_quarter(model, input_ids_1d, token2id, id2token, q=0.25):
    with torch.no_grad():
        pad_id = token2id["<pad>"]
        eos_id = token2id["<eos>"]
        
        device = next(model.parameters()).device
        #убираем pad справа
        #проверка на pad в inp_ids последовательности
        if (input_ids_1d == pad_id).any(): #если есть паддинги
            nonpad = (input_ids_1d != pad_id).nonzero(as_tuple=False) #тензор nonpad индексов

            if len(nonpad) == 0: #проверка на пустой вход
                return None
            
            L = int(nonpad[-1].item()) + 1
            seq = input_ids_1d[:L] #обрезаем seq до последнего не паддинг токена
        else:
            seq = input_ids_1d
        
        words_len = int(seq.numel()) - 1 #кол-во слов без <bos>
        if seq.numel() < 3: #если слишком короткая seq
            return None
        
        #сколько слов оставить в промпте (для нашей задачи четверть)
        words2keep = max(1, int((1 - q) * words_len))
        prompt_len = 1 + words2keep
        promt_ids = seq[:prompt_len] #начало текста
        target_ids = seq[prompt_len:] #конец который модель должна продолжить
        gen_len = int(target_ids.numel()) #таргет кол-ва генерации токенов = длина конца текста
        gen_ids = greedy_generate(model, promt_ids, eos_id, gen_len, device) #генерация с argmax
        
        promt_text = decode(promt_ids.tolist(), id2token, skip_special_tokens=True)#декодим promt id токенов в слова
        target_text = decode(target_ids.tolist(), id2token, skip_special_tokens=True)#декод для target ids
        pred_text = decode(gen_ids, id2token, skip_special_tokens=True)#декод для gen ids

        return promt_text, target_text, pred_text

def evaluate_rouge(model, loader, token2id, id2token, max_batches=100, q=0.25):
    with torch.no_grad():
        model.eval()
        pad_id = token2id["<pad>"]
        r1_sum, r2_sum, n = 0,0,0
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches: #ограничение по числа батчей
                break
            
            input_ids, target_ids = batch
            lengths = (input_ids != pad_id).long().sum(dim=1).tolist() #список длин

            for i, L in enumerate(lengths):
                if L < 3: #не подходит под для 3/4 1/4 
                    continue
                seq = input_ids[i, :L]
                res = generate_last_quarter(model, seq, token2id, id2token, q)
                prompt_text, target_text, pred_text = res
                r1_sum += rouge_f1(target_text, pred_text, n=1)#rouge1
                r2_sum += rouge_f1(target_text, pred_text, n=2)#rouge2
                n+=1
        return r1_sum / n, r2_sum / n #средние rouge1/rouge2