import json
import os
from bs4 import BeautifulSoup
# from llamaapi import LlamaAPI

# Initialize the SDK with your API token
llama = "INSERT_API"
assert llama is not "INSERT_API", "please replace `llama` with llama token, otherwise the program won't work"
if llama is "INSERT_API":
    quit()

def read_html_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def extract_problem_description(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    problem_description = soup.get_text(separator="\n")
    return problem_description


from openai import OpenAI

client = OpenAI(
    api_key = llama,
    base_url = "https://api.llama-api.com"
)

# Path to the folder containing problem descriptions
problem_folder_path = "problem_descriptions"


def extract_cpp_code(assistant_code):
    start_marker = "```"
    end_marker = "```"
    start = assistant_code.find(start_marker)

    if start == -1:
        print(f"Warning: Start marker not found in {filename}")
        return None

    start += len(start_marker)
    end = assistant_code.find(end_marker, start)

    if end == -1:
        print(f"Warning: End marker not found in {filename}")
        return None

    cpp_code = assistant_code[start:end].strip()
    return cpp_code


code_list = []
# Iterate over each HTML file in the folder
for filename in sorted(os.listdir(problem_folder_path)):
    if filename.endswith(".html"):
        html_content = read_html_file(os.path.join(problem_folder_path, filename))
        problem_description = extract_problem_description(html_content)
        
        prompt = f"\nplease pretend you are a coding robot that is aiming at writing a standard solution for a coding contest. Your task is to output clean, well-named, commented, and well-formatted C++ code for the following problem. NOTE: Enclose your C++ code within triple backticks (```) for clear formatting. Ensure that the code includes:\n1. Descriptive comments explaining the logic of every step.\n2. Well-named variables for easy understanding.\n3. Proper formatting and structure adhering to C++ best practices.\nOnce the problem is understood, please provide the C++ solution. Here's the problem:\n{problem_description}\n\n"
        # print(prompt)
        response = client.chat.completions.create(model="llama-70b-chat", messages=[{"role": "user", "content": prompt}])
        # Assuming 'response' is the response object you received from the API
        if response.choices:
            # Extracting the content from the first choice (assuming it's the relevant one)
            assistant_message = response.choices[0].message.content
            if assistant_message:
                cpp_code = extract_cpp_code(assistant_message)
                if cpp_code is not None:
                    code_dict = {"id": filename, "answer": cpp_code}
                    code_list.append(code_dict)
                    print(f"{filename} successfully generated!")
            else:
                print(f"WARNING! for {filename} No code was generated.")
        else:
            print("No choices available in the response.")
    if len(code_list)%36 == 1:
        with open('code_solutions_sortedQ.json', 'w') as file:
            json.dump(code_list, file, indent=4)
            print(f"C++ code has been saved to 'code_solutions_sortedQ.json' at {len(code_list)}")
