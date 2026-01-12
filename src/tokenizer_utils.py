import collections



PAD_token = "<pad>"
EOS_token = "<eos>"
BOS_token = "<bos>"
UNK_token = "<unk>"

special_tokens = [PAD_token, UNK_token, BOS_token, EOS_token]

def build_vocab(train_path, max_vocab_size=None, min_freq=1):
    counter = collections.Counter()

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as t_f:
        for line in t_f:
            tokens = line.split() #разбиваем строку по пробелам на токены
            counter.update(tokens)

    vocab = {} #token>id
    id_to_token = [] #id>token
    for stok in special_tokens: #добавляем спец токены в словарь
        vocab[stok] = len(id_to_token)
        id_to_token.append(stok)
    #сортируем по убыванию кол-во токенов,если частота одинакова, сортируем по алфавиту
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    if max_vocab_size is not None: #ограничитель словаря
        limit = max(0, max_vocab_size - len(special_tokens))
    else:
        limit = None
    
    t_added = 0 #счётчик для лимита
    for token, freq in items:
        if freq < min_freq:
            break
        if limit is not None and t_added >= limit:
            break
        vocab[token] = len(id_to_token) #присваиваем id токену
        id_to_token.append(token)#добавляем в список id2token
        t_added+=1
    return vocab, id_to_token

def encode(text, vocab, add_bos=False, add_eos=False):
    tokens = text.strip().split()
    if add_bos:
        tokens = [BOS_token] + tokens
    if add_eos:
        tokens = tokens + [EOS_token]
    unk_id = vocab[UNK_token] #на случай если токена нет в словаре
    ids = [vocab.get(tok, unk_id) for tok in tokens] #преобразуем список токенов в список id
    return ids

def decode(ids, id_to_token, skip_special_tokens=True):
    tokens = []
    for i in ids:
        if i <0 or i >= len(id_to_token):
            tok = UNK_token
        else: tok = id_to_token[i]

        if skip_special_tokens and tok in special_tokens:
            continue
        tokens.append(tok)
    return(" ".join(tokens))