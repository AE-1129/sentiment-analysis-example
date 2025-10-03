
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configurations
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 128

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Example inputs
    examples = [
        "This movie was a fantastic masterpiece, I loved the acting and story.",
        "I wasted two hours. The plot is boring and the acting was terrible.",
    ]

    # Tokenize inputs
    inputs = tokenizer(examples, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

    # Run inference
    outputs = model(**inputs)
    preds = outputs.logits.argmax(axis=-1).tolist()

    # Show results
    for ex, p in zip(examples, preds):
        sentiment = "Positive" if p == 1 else "Negative"
        print(f"{sentiment} -> {ex}")

if __name__ == "__main__":
    main()
