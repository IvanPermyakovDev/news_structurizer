import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import sys

def generate_attributes(text, generator):
    # Подготавливаем промпты для всех атрибутов
    tasks = {
        "title": "заголовок: ",
        "key_events": "событие: ",
        "location": "локация: ",
        "key_names": "имена: "
    }
    
    results = {}
    
    for key, prefix in tasks.items():
        input_text = prefix + text
        # Генерация
        output = generator(
            input_text, 
            max_length=200, 
            num_beams=4,       # Лучевой поиск для лучшего качества
            early_stopping=True
        )[0]['generated_text']
        
        results[key] = output
        
    return results

def main():
    model_path = "./final_model"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        # Fallback to base model for demonstration if trained model doesn't exist? 
        # No, better to fail or warn. But let's allow using the base model if user wants to test pipeline structure
        # but the prompt implies we use the trained model.
        return

    print(f"Loading model from {model_path}...")
    device = 0 if torch.cuda.is_available() else -1
    
    # We can load directly with pipeline if the folder contains model and tokenizer
    try:
        generator = pipeline("text2text-generation", model=model_path, tokenizer=model_path, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Example text (you can also read from stdin or a file)
    new_text_asr = "эфир продолжаем сейчас казань и вот прямо live у здани администрации слышно шум люд и автобусы подъеzzают..."
    
    # If arguments provided, use them
    if len(sys.argv) > 1:
        # If arg is a file
        if os.path.isfile(sys.argv[1]):
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                new_text_asr = f.read()
        else:
            new_text_asr = " ".join(sys.argv[1:])

    print(f"Processing text: {new_text_asr[:100]}...")
    attributes = generate_attributes(new_text_asr, generator)

    print("=== Результат генерации ===")
    print(json.dumps(attributes, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
