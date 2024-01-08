import json
import glob

def load_processed_code_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def save_list_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# old code for single file
# data = load_processed_code_from_file("code_solutions.json")
# print(len(data))

files = glob.glob('code_solutions*.json')
data = []

# Load and concatenate data from all files
for file in files:
    print(f"loading {file}")
    data.extend(load_processed_code_from_file(file))


data_cleaned = []
for i in range(len(data)):
    if len(data[i]["answer"]) > 80:
        data_cleaned.append(data[i]["answer"])

print(len(data_cleaned))

save_list_to_json(data_cleaned, "llama_code_24.json")
