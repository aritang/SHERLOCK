import json
def load_processed_code_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def save_list_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
import numpy as np
import pandas as pd


def split(data):
    M = len(data[-1])
    N = len(data) - M
    data_ori = []
    data_per = []
    for i in range(M):
        data_ori.append(data[i])
    for j in range(M):
        data_ = []
        for i in range(M, M+N):
            data_.append(data[i][j])
        data_per.append(data_)
    return data_ori, data_per

GPT_results = load_processed_code_from_file("GPT_result_24.json")
GPT_results_ori, GPT_results_per = split(GPT_results)
GPT_results_per = np.array(GPT_results_per)

human_results = load_processed_code_from_file("human_result_24.json")
human_results_ori, human_results_per = split(human_results)
human_results_per = np.array(human_results_per)

z = list(GPT_results_ori - GPT_results_per.mean(axis = 1))
y = list(human_results_ori - human_results_per.mean(axis = 1))

df = pd.DataFrame({
    "prob" : y + list(z),
    "label": list(np.zeros(len(y))) + list(np.ones(len(z)))
    })

import seaborn as sns
sns.histplot(data = df, x = "prob", hue = "label")
import matplotlib.pyplot as plt
plt.show()
