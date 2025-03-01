import csv
from openai import OpenAI
import networkx as nx
# import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
import random

def remove_numbering(text):
    return re.sub(r'^\d+\.\s*', '', text)

# Set your Perplexity API key as an environment variable
os.environ["PERPLEXITY_API_KEY"] = "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"
# os.environ["PERPLEXITY_API_KEY"] = "<insert key here>"

# Initialize the OpenAI client with Perplexity's base URL
client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")

def generate_paraphrase(sentence):
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates only one paraphrase of a given sentences. You dont put any extra text or numbers in the output"
        },
        {
            "role": "user",
            "content": f"Generate paraphrase of the following sentence: '{sentence}'. Provide only the paraphrase without any extra text or numbers"
        }
    ]
    
    response = client.chat.completions.create(
        model="sonar",
        messages=messages,
        max_tokens=120,
        temperature=0.7
    )
    
    generated_text = response.choices[0].message.content
    return generated_text

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
file_path_train = 'MSRParaphraseCorpus/msr_paraphrase_test.txt'
parsed_data_train = read_tab_separated_file(file_path_train)

# Print the first few rows to verify
print("example = ")
new_paraphrase_db = []
# G = nx.Graph()  # Create an undirected graph    #GRAPH
# sent_id = {}    # USed only for plotting the graph   #GRAPH
# cur_id=0    # USed only for plotting the graph   #GRAPH
for row in tqdm(parsed_data_train[2:30], desc="Generating Paraphrases"):
    current_sentences = [row["string1"], row["string2"]]
    # if row["string1"] not in sent_id:    #GRAPH
    #     sent_id[row["string1"]] = cur_id    #GRAPH
    #     cur_id+=1    #GRAPH
    # if row["string2"] not in sent_id:    #GRAPH
    #     sent_id[row["string2"]] = cur_id    #GRAPH
    #     cur_id+=1    #GRAPH
    new_paraphrase_db.append([row["Quality"], row["string1"], row["string2"]])

    # Add initial connection to the graph
    # if row["Quality"] == 1: #GRAPH
    #     G.add_edge(sent_id[row["string1"]], sent_id[row["string2"]]) #GRAPH

    # choose a senctence from the paraphrases 10 times ranomly
    for i in range(10):
        sentence = random.choice(current_sentences)
        paraphrase = generate_paraphrase(sentence)
        current_sentences.append(paraphrase)
        new_paraphrase_db.append([1, sentence, paraphrase])
        # if sentence not in sent_id:    #GRAPH
        #     sent_id[sentence] = cur_id    #GRAPH
        #     cur_id+=1    #GRAPH
        # if paraphrase not in sent_id:    #GRAPH
        #     sent_id[paraphrase] = cur_id
        #     cur_id+=1
        # G.add_edge(sent_id[sentence], sent_id[paraphrase]) #GRAPH
out_file_name = 'output2_2.txt'
# Visualize the graph
# plt.figure(figsize=(12, 8)) #GRAPH
# pos = nx.spring_layout(G)  # Use spring layout for better visualization #GRAPH
# nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=20, font_size=10, font_weight='bold') #GRAPH
# plt.title("Paraphrase Graph", fontsize=16) #GRAPH
# plt.savefig("paraphrase_graph.png") #GRAPH

# Append to the tab-separated file
with open(out_file_name, 'a') as file:
    # Append the list as tab-separated lines
    for element in new_paraphrase_db:
        file.write('\t'.join(map(str, element)) + '\n')

# # Verify the contents of the file
# with open(out_file_name, 'r') as file:
#     content = file.read()
#     print(content)
    