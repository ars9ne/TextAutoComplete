# Text AutoComplete (LSTM vs disitlgpt2)

Проект решает задачу автодополнения текста.
Сравниваются две модели:
 - **LSTM** (легковесная модель, обученная на датасете Sentiment140)
 - **distilgpt2** (предобученный трансформер из Transformers, используемый без допобучения)

Качество оценивается метриками Rouge-1/Rouge-2 (F1 score) в сценарии генерации последней четверти текста.

## Структура проекта
- `data/raw/` — исходный датасет (`tweets.txt`)
- `data/clean/` — очищенный датасет (`tweets_clean.txt`)
- `data/splits/` — разбиение на `train.txt / val.txt / test.txt`
- `src/` — код проекта:
  - `data_utils.py` — очистка и разбиение данных
  - `tokenizer_utils.py` — построение словаря + encode/decode
  - `next_token_dataset.py` — Dataset + collate_fn для next-token prediction
  - `lstm_model.py` — LSTM модель
  - `lstm_train.py` — обучение LSTM и сохранение весов
  - `eval_lstm.py` — оценка LSTM (ROUGE)
  - `eval_transformer_pipeline.py` — оценка distilgpt2 (ROUGE)
- `models/` — сохранённые веса (`lstm_best.pth`)
- `solution.ipynb` — ноутбук со всем пайплайном и выводами

## Запуск пайплайна
1. Установка зависимостей из `requirements.txt`
2. Установка исходного датасета в `data/raw/tweets.txt`
3. Запуск `src/data_utils.py`, для очистки текста и создания `train/val/test` в `data/splits`
4. Выполнение в `solution.ipynb` ячеек для построения словаря и DataLoader`ов.
5. Запуск обуения LSTM в ноутбуке, веса сохраняются в `models/lstm_best.pth`.
6. Запуск оценки (ROUGE) и тестирование LSTM и distilgpt2.
