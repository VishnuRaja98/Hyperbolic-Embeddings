import csv

# Function to read a TSV file and process it
def read_tsv_with_conditions(file_path):
    # Initialize an empty list to store the processed rows
    data = []
    
    # Open the TSV file
    with open(file_path, mode='r', encoding='utf-8') as tsv_file:
        # Use csv.DictReader with delimiter set to '\t' for TSV files
        reader = csv.DictReader(tsv_file, delimiter='\t')
        
        # Iterate over each row in the file
        for row in reader:
            # Condition: Check if entailment_AB is 'A_neutral_B' and entailment_BA is 'B_neutral_A'
            if row['entailment_AB'] == 'A_neutral_B' and row['entailment_BA'] == 'B_neutral_A':
                # Append the row to the data list if conditions are met
                data.append(row)
    
    return data

# Example usage
file_path = 'SICK-Relatedness database\SICK\SICK.txt'  # Replace with the path to your TSV file
processed_data = read_tsv_with_conditions(file_path)

# Print the processed data (optional)
for entry in processed_data:
    print(entry)
