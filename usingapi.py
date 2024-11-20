from openai import OpenAI
import os
import re

def remove_numbering(text):
    return re.sub(r'^\d+\.\s*', '', text)

# Set your Perplexity API key as an environment variable
os.environ["PERPLEXITY_API_KEY"] = "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"

# Initialize the OpenAI client with Perplexity's base URL
client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")

def generate_paraphrases(sentence, num_paraphrases=10):
    paraphrases = []
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates paraphrases of given sentences."
        },
        {
            "role": "user",
            "content": f"Generate {num_paraphrases} unique paraphrases of the following sentence: '{sentence}'. Provide only the paraphrases without any extra text or numbers"
        }
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instruct",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    
    generated_text = response.choices[0].message.content
    paraphrases = [p.strip() for p in generated_text.split('\n') if p.strip()]
    
    return paraphrases[:num_paraphrases]

# Example usage
sentence = "After the French threatened to open fire, the battlewagon turned back."
paraphrase_list = generate_paraphrases(sentence)
paraphrase_list = [remove_numbering(item) for item in paraphrase_list]

print("Original sentence:", sentence)
print("\nParaphrases:")
for i, paraphrase in enumerate(paraphrase_list, 1):
    print(f"{i}. {paraphrase}")