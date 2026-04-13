from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd
import time
import json
import traceback
import pickle
import random
# import numpy as np

from threading import Lock

# ==================== Progress Tracking ====================
completed_count = 0
count_lock = Lock()

# ==================== Dataset Parameters ====================
num_drug = 1020
num_se = 5599
rank = 20

# ==================== LLM Configuration ====================
max_workers = 5
modeltype = 'plus'
model = "qwen-plus-2025-12-01"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "xxxxxxxxxxxxxx"

# ==================== Load Selected Drug Indices ====================
selected_filename = f'data_pt/drugs_top{num_drug}.pkl'
with open(selected_filename, 'rb') as f:
    drug_selected, drug_index = pickle.load(f)

# ==================== Load and Sample Side Effects ====================
se_df = pd.read_excel('data_pt/se_id.xlsx', sheet_name='Sheet1')
se_all = se_df['side effect'].tolist()

random.seed(0)
se_index = random.sample(range(len(se_all)), num_se)
se_selected = [se_all[i] for i in se_index]
print(len(se_selected))

# ==================== Build Drug‑Side Effect Matrix ====================
# For each side effect, sample up to `rank` related drugs (matrix value == 1)
matrix_df = pd.read_excel('data_pt/new_drug_eff.xlsx')
matrix_df.set_index(matrix_df.columns[0], inplace=True)
selected_matrix_df = matrix_df.iloc[drug_index, se_index]
selected_matrix_df.columns = se_selected

_related_drugs_dict = {}
for side_effect in se_selected:
    related_drugs = selected_matrix_df[selected_matrix_df[side_effect] == 1].index.tolist()
    if len(related_drugs) > rank:
        related_drugs = random.sample(related_drugs, rank)
    _related_drugs_dict[side_effect] = related_drugs

# ==================== File Paths ====================
save_path = f'data_drug_se_s2/ds_0403_ns{num_se}_nd{num_drug}_r{rank}_{modeltype}'
s1_path_drug = f'data_drug_s1/drug_0403_combine_n{num_drug}_{modeltype}_s1.json'
s1_path_se = 'data_pt/se_definition.xlsx'

# ==================== Load Precomputed Stage‑1 Data ====================
with open(s1_path_drug, 'r') as f:
    drug_dict = json.load(f)

side_effects_df = pd.read_excel(s1_path_se, sheet_name='Sheet1')
side_effects_df.iloc[se_index, :]
side_effects_df.set_index('side effect', inplace=True)

# ==================== LLM API Call Function ====================
def call_qwen_api(query, api_key, base_url, retry_delay=10, max_retries=3):
    """
    Call the Qwen LLM API with the given query.
    Implements retry logic for rate-limit errors (429).
    Returns the cleaned JSON string from the API response.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    retries = 0

    while retries <= max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': 'You are an expert in pharmacy.'},
                    {'role': 'user', 'content': query}
                ],
            )
            jstr = completion.choices[0].message.content
            print('before:', jstr)
            jstr_cleaned = jstr.strip("`' json \n\t").strip()
            return jstr_cleaned

        except Exception as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            print("\n".join(tb))
            with open('error_log.txt', 'a') as f:
                f.write(f"{query}")

            if "429" in str(e) and retries < max_retries:
                print(f"429 error, wait for {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                print("Reached maximum retry attempts or a non-429 error, giving up retry...")
                return "None"

# ==================== Query Template ====================
QUERY = '''\
Please analyze how the drug {drug} with the description "{description}" causes \
    the side effect {side_effect} with the definition "{definition}", \
    from the following four categories: administration route, metabolism pathway, target selectivity, and structural properties.
Your answer should include the following information:
1. The **category** that is the cause (one of the four listed above)
2. A concise **explanation** of how this category leads to the side effect, less than 30 words:
   - If the cause is the administration route, specify the exact route of administration and the involved organs or systems.
   - If the cause is the metabolism pathway, specify which organ is responsible for metabolism and what substances are being metabolized.
   - If the cause is target selectivity, specify the location of the target distribution.
   - If the cause is structural properties, specify the exact structure and characteristics.
   - Choose the one reason that you think is the most likely.
3. The **summary** of the reasoning behind the above answer.
Requirements:
1. Your response should be in JSON format, with the following keys: category, explanation and summary.
2. Do not return any information outside of the JSON.
3. The answer should avoid redundant information, ensuring high information density.
4. Answer in English.
'''

# ==================== Process a Single Side Effect ====================
def generate_description(side_effect):
    """
    For a given side effect, query the LLM for each of its related drugs.
    Returns a tuple (side_effect, outputs_dict) where outputs_dict maps drug names
    to the LLM's JSON response.
    """
    related_drugs = _related_drugs_dict.get(side_effect, [])
    print('related drugs:', len(related_drugs))

    definition = side_effects_df.loc[side_effect, 'definition']
    outputs = {}

    for drug in related_drugs:
        description = drug_dict[drug]
        query = QUERY.format(side_effect=side_effect, drug=drug, definition=definition, description=description)
        jstr_cleaned = call_qwen_api(query, api_key, base_url)
        outputs[drug] = jstr_cleaned

    # Update global progress counter (thread-safe)
    global completed_count
    with count_lock:
        completed_count += 1
        print(f"Progress: {completed_count}/{len(se_selected)} side effects completed")

    return side_effect, outputs

# ==================== Main Parallel Processing Function ====================
def job1_model_query(side_effects, save_path, max_workers=5):
    """
    Main function: process a list of side effects in parallel using ThreadPoolExecutor.
    For each side effect, calls generate_description().
    Results are saved as a JSON file.
    """
    total_outputs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_description, side_effect): side_effect for side_effect in side_effects}
        for future in as_completed(futures):
            side_effect, outputs = future.result()
            total_outputs[side_effect] = outputs

    with open(save_path + '.json', 'w', encoding='utf-8') as fp:
        json.dump(total_outputs, fp, ensure_ascii=False, indent=2)

# ==================== Entry Point ====================
if __name__ == '__main__':
    job1_model_query(se_selected, save_path=save_path)