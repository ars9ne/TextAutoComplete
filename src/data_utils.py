import re
import os
from datasets import load_dataset

def clean(text): #очищаем датасет
    text = text.lower() #нижний регистр
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text) #удаление ссылок http(s) и www.
    text = re.sub(r'@\w+', '', text) #удаление упоминаний
    text = re.sub(r'[^0-9A-Za-z\s.,!?;:\-()]+', '', text) #удаление нестандартные символы
    text = re.sub(r'\s+', ' ', text).strip() #удаление повторяющихся пробелы + табов в начале строк
    return text

def process_file(raw_data, proccesed_data): #построчно очищаем датасет
    with open(raw_data, 'r', encoding='utf-8', errors='ignore') as raw,\
        open(proccesed_data, 'w', encoding='utf-8') as out:
        for line in raw:
            c = clean(line)
            if c:
                out.write(c + '\n')

def save_dataset(dataset, output_path):
    with open(output_path, 'w', newline='\n', encoding='utf-8') as out_file:
        for x in dataset["text"]:
            out_file.write(x + "\n")

def split_file(file_path): #разбиваем датасет на train/val/test 80/10/10
    ds = load_dataset("text", data_files={"data": file_path})["data"]
    ds = ds.shuffle(seed=1)
    tmp = ds.train_test_split(test_size=0.2, seed=1) #делим датасет на 80/20 выборку
    train_ds = tmp["train"]
    tmp2 = tmp["test"].train_test_split(test_size=0.5, seed=1) #оставшиеся 20 делим на 10/10 под val и test
    val_ds = tmp2["train"]
    test_ds = tmp2["test"]
    save_dataset(train_ds, "./data/splits/train.txt")
    save_dataset(val_ds,   "./data/splits/val.txt")
    save_dataset(test_ds,  "./data/splits/test.txt")

def main():
    raw_data_file = './data/raw/tweets.txt'
    clean_data_file = './data/clean/tweets_clean.txt'
    process_file(raw_data=raw_data_file, proccesed_data=clean_data_file)
    split_file(clean_data_file)

if __name__ == "__main__":
    main()