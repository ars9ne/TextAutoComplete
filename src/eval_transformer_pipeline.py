from transformers import pipeline
from rouge_score import rouge_scorer


def read_lines(path, max_samples=None):
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            lines.append(t)
            if max_samples is not None and len(lines) >= max_samples:
                break
    return lines


def split_prompt_target(text, ratio=0.75, min_words=8):  #делим на prompt/target 75/25
    words = text.split()
    if len(words) < min_words:
        return None, None

    cut = int(ratio * len(words))
    cut = max(1, min(cut, len(words) - 1))  #чтобы не было пустого prompt

    prompt = " ".join(words[:cut]).strip()
    target = " ".join(words[cut:]).strip()
    if not prompt or not target:
        return None, None
    return prompt, target


def build_generator(device=-1):  #создание генератора текста для destilgpt2
    return pipeline("text-generation", model="distilgpt2", device=device)


def estimate_max_new_tokens(target_text):
    n_words = len(target_text.split())  #кол-во слов в target
    return max(10, n_words * 2)  #примерное число токенов (мин 10, иначе 2 токена на слова)


def generate_continuation(generator, prompt, max_new_tokens, gen_kwargs):
    out = generator(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)  #генерируем продолжение
    full = out[0]["generated_text"]

    if full.startswith(prompt):  #вырезаем промт
        cont = full[len(prompt):]
    else:
        cont = full

    return cont.strip()


def evaluate_transformer(val_path, device=-1, max_samples=None, num_print=5):
    generator = build_generator(device=device)
    # ROUGE-1/2
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

    #параметры генерации
    gen_kwargs = {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.8,
    }

    lines = read_lines(val_path, max_samples=max_samples)

    total_r1 = 0.0
    total_r2 = 0.0
    used = 0
    shown = 0

    for text in lines:
        prompt, target = split_prompt_target(text, ratio=0.75)
        if prompt is None:
            continue

        max_new = estimate_max_new_tokens(target)
        pred = generate_continuation(generator, prompt, max_new, gen_kwargs)

        scores = scorer.score(target, pred)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure

        total_r1 += r1
        total_r2 += r2
        used += 1

        if shown < num_print:
            print("=" * 60)
            print("Prompt:", prompt)
            print("Target:", target)
            print("Prediction:", pred)
            print(f"ROUGE-1: {r1:.4f} | ROUGE-2: {r2:.4f}")
            shown += 1

    if used == 0:
        return 0.0, 0.0, 0

    return total_r1 / used, total_r2 / used, used


if __name__ == "__main__":
    r1, r2, n = evaluate_transformer("./data/splits/val.txt", device=-1, max_samples=200, num_print=5)
    print(f"\nVAL: n={n}  ROUGE-1={r1:.4f}  ROUGE-2={r2:.4f}")
