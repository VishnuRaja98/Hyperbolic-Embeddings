import csv
from openai import OpenAI
from tqdm import tqdm
import os
import re

def remove_numbering(text):
    return re.sub(r'^\d+\.\s*', '', text)

# Set your Perplexity API key as an environment variable
os.environ["PERPLEXITY_API_KEY"] = "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"
# os.environ["PERPLEXITY_API_KEY"] = "<insert key here>"

# Initialize the OpenAI client with Perplexity's base URL
client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")

def generate_paraphrases(sentence, num_paraphrases=5):
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

def read_tab_separated_file(file_path):
    data = []
    problem_rows = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter='\t', quotechar=None)
        headers = next(csv_reader, None)  # Read and skip the header row

        for row in csv_reader:
            if len(row) >= 5:
                data.append({
                    'Quality': row[0],
                    'id1': row[1],
                    'id2': row[2],
                    'string1': row[3],
                    'string2': row[4]
                })
            else:
                problem_rows += 1

    print("!!!!!!total problem rows = ", problem_rows)
    return data

# Example usage
file_path_train = 'MSRParaphraseCorpus/msr_paraphrase_train.txt'
parsed_data_train = read_tab_separated_file(file_path_train)

# Print the first few rows to verify
print("example = ")
new_paraphrase_db = []
for row in tqdm(parsed_data_train[:1000], desc="Generating Paraphrases"):
    sent1 = row["string1"]
    sent2 = row["string2"]
    # print("Generating paraphrases = ", sent1)
    paraphrase_list = generate_paraphrases(sent1)
    paraphrase_list = [remove_numbering(item) for item in paraphrase_list]
    new_paraphrase_db = new_paraphrase_db + [[1,sent1, newsent] for newsent in paraphrase_list]
    # print("Generating paraphrases = ", sent2)
    paraphrase_list = generate_paraphrases(sent2)
    paraphrase_list = [remove_numbering(item) for item in paraphrase_list]
    new_paraphrase_db = new_paraphrase_db + [[1,sent2, newsent] for newsent in paraphrase_list]

out_file_name = 'output.txt'

# Append to the tab-separated file
with open(out_file_name, 'a') as file:
    # Append the list as tab-separated lines
    for element in new_paraphrase_db:
        file.write('\t'.join(map(str, element)) + '\n')

# # Verify the contents of the file
# with open(out_file_name, 'r') as file:
#     content = file.read()
#     print(content)
    