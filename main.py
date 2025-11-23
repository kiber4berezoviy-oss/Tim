from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM 
import torch

# Исправляем название модели для анализа тональности
sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# Альтернативные модели, если вышеуказанная не сработает:
# sentiment_analyzer = pipeline("sentiment-analysis", model="seara/rubert-tiny2-russian-sentiment")

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

app = Flask(__name__)

def generate_recommendation(mood):
    prompt = f"Посоветуй один фильм для человека, у которого настроение {mood}. Посоветуй один фильм и кратко объясни почему"
    
    # Добавляем pad_token, если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Исправляем опечатку: max_lenght -> max_length
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,  # Исправлено!
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.ne(tokenizer.pad_token_id)
        )
    
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = ""
    user_text = ""
    ai_result = ""  # Инициализируем переменную
    
    if request.method == "POST":
        user_text = request.form["message"]
        result = sentiment_analyzer(user_text)[0]  # Исправлено!
        label = result["label"]
        score = result["score"]
        
        print(f"Analyzed: '{user_text}' -> {label} ({score:.3f})")  # Для отладки
        
        if label == "POSITIVE" or label == "LABEL_1":  # Зависит от модели
            mood = "позитивное"
            recommendation = "У тебя отличное настроение!"
        elif label == "NEGATIVE" or label == "LABEL_0":  # Зависит от модели
            mood = "негативное"
            recommendation = "Ты немного грустишь"
        else:
            mood = "нейтральное"
            recommendation = "Нейтральное настроение"
        
        # Генерируем рекомендации для любого настроения
        ai_text = generate_recommendation(mood)
        ai_result = f"Настроение: {recommendation}. Рекомендации: {ai_text}"

    return render_template("index.html", recommendation=recommendation, user_text=user_text, ai_result=ai_result)

if __name__ == '__main__':
    app.run(debug=True)