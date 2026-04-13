from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd
import time
import json
import pickle

# ==================== Input Hyperparameters ====================
num_drug = 1020

max_workers = 20
modeltype = 'plus'
model = "qwen-plus-2025-12-01"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "xxxxxxxxxxxxx"

# ==================== Output File Paths ====================
# Combined new drug descriptions (JSON)
json_path_combine = f'data_drug_s1/drug_0403_combine_n{num_drug}_{modeltype}_s1.json'
# Split descriptions as a table (Excel)
excel_path_split = f'data_drug_s1/drug_0403_split_n{num_drug}_{modeltype}_s1.xlsx'
# Queries sent to the LLM (JSON)
json_path_query = f'data_drug_s1/drug_0403_query_n{num_drug}_{modeltype}_s1.json'

# ==================== Load Drug and Side Effect Names ====================
drugs_df = pd.read_excel('C:/Users/Administrator/Desktop/MLDSP+/profile_generation_all/profile_generation_all/data_pt/drug_id.xlsx', sheet_name='Sheet1')
se_df = pd.read_excel('C:/Users/Administrator/Desktop/MLDSP+/profile_generation_all/profile_generation_all/data_pt/se_id.xlsx', sheet_name='Sheet1')

drug_all = drugs_df['drug'].tolist()
se_all = se_df['side effect'].tolist()

# ==================== Build Drug-Side Effect Matrix ====================
matrix_df = pd.read_excel('C:/Users/Administrator/Desktop/MLDSP+/profile_generation_all/profile_generation_all/data_pt/drug_eff.xlsx')
matrix_df = matrix_df.set_index(matrix_df.columns[0])   # First column as index (drug names)
matrix_df.columns = se_all                              # Columns = side effects

# Select top-num_drug drugs with highest total side effect counts
drug_selected = matrix_df.sum(axis=1).sort_values(ascending=False).index.tolist()[:num_drug]
index_selected = [drug_all.index(drug) for drug in drug_selected]

# Save selected drug names and their original indices
selected_filename = f'C:/Users/Administrator/Desktop/MLDSP+/profile_generation_all/profile_generation_all/data_pt/drugs_top{num_drug}.pkl'
with open(selected_filename, 'wb') as f:
    pickle.dump([drug_selected, index_selected], f)

# ==================== Load Initial Drug Descriptions ====================
excel_inpath = f"C:/Users/Administrator/Desktop/MLDSP+/profile_generation_all/profile_generation_all/data_pt/drug_definition.xlsx"
drug_des_df = pd.read_excel(excel_inpath, sheet_name='Sheet1')
drug_descriptions = dict(zip(drug_des_df['drug'], drug_des_df['drug-target interactions']))

# ==================== LLM API Call Function ====================
def call_qwen_api(query, api_key, base_url, retry_delay=10, max_retries=3):
    """
    Call the Qwen LLM API with retry logic for rate‑limit errors (429).
    Returns a tuple: (raw_response, cleaned_json_string).
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
                ]
            )
            print('Completed one request')
            jstr = completion.choices[0].message.content
            print('before:', jstr)
            jstr_cleaned = jstr.strip().strip("```").strip("'''").strip("json").strip()
            return jstr, jstr_cleaned
        except Exception as e:
            print(f"Error: {e}")
            if "429" in str(e) and retries < max_retries:
                print(f"429 error, wait for {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                print("Reached maximum retry attempts or a non-429 error, giving up retry...")
                return "None", "None"

# ==================== Query Template ====================
QUERY = '''\
You will serve as an expert in pharmacology to provide information about {drug}, covering:
1. Indicate the administration route of {drug}: oral, intravenous, intramuscular, subcutaneous, buccal, rectal, ophthalmic, nasal, topical \
    or a combination of these.
2. Analyze the anatomical location where {drug} exerts its action, including the involved systems and organs.
3. Identify the distribution of {drug}'stargets (including the involved systems and organs), based on the provided drug-target interactions: {interactions}.
4. Determine which structural or toxicological features of {drug} could lead to side effects, \
    including functional groups, stereochemistry, lipophilicity/hydrophilicity, and metabolites.
5. Explain the metabolic pathways of {drug}, including the involved systems and organs, enzymes, and metabolites formed.
Output the results in the following JSON format:
{{
    "routes of administration": "Administration routes of {drug}",
    "sites of action": "Anatomical location of action for {drug}",
    "distribution of targets": "Distribution of {drug}'s targets",
    "structural features": "Structural or toxicological features of {drug} that may cause side effects",
    "metabolic pathways": "Metabolic pathways of {drug}"
}}
Requirements:
1. Respond in English.
2. If any information does not exist, use None to indicate it.
3. Do not return any information outside of the JSON.
4. When answering regarding systems, use one of the following: "musculoskeletal system, circulatory system, \
    respiratory system, digestive system, urinary system, nervous system, endocrine system, reproductive system".
'''

# ==================== Process a Single Drug ====================
def generate_description(drug):
    """Build query, call LLM, parse JSON, and return structured data."""
    interactions = drug_descriptions[drug]
    query = QUERY.format(drug=drug, interactions=interactions)

    jstr, jstr_cleaned = call_qwen_api(query, api_key, base_url)

    if jstr_cleaned:
        try:
            json_dict = json.loads(jstr_cleaned)
            return drug, query, jstr, json_dict
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Invalid JSON string: {jstr_cleaned}")
            return drug, query, jstr, {}
    return drug, query, "None", {}

# ==================== Main Execution ====================
new_drug_descriptions = {}   # raw cleaned responses (string)
new_query = {}               # queries sent to LLM
new_drug_df = pd.DataFrame(columns=["drug", "routes of administration", "sites of action",
                                    "distribution of targets", "structural features", "metabolic pathways"])

# Parallel processing using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(generate_description, drug): drug for drug in drug_selected}

    for future in as_completed(futures):
        drug, query, jstr, json_dict = future.result()

        # Store raw cleaned response
        new_drug_descriptions[drug] = jstr.strip("`' json \n\t{}").strip()
        new_query[drug] = query

        # Append structured data to DataFrame
        len_df = len(new_drug_df)
        print(len_df)
        new_drug_df.loc[len_df] = {
            "drug": drug,
            "routes of administration": json_dict.get("routes of administration", "None"),
            "sites of action": json_dict.get("sites of action", "None"),
            "distribution of targets": json_dict.get("distribution of targets", "None"),
            "structural features": json_dict.get("structural features", "None"),
            "metabolic pathways": json_dict.get("metabolic pathways", "None")
        }

# Save results
with open(json_path_combine, "w") as f:
    json.dump(new_drug_descriptions, f, indent=4)

with open(json_path_query, "w") as f:
    json.dump(new_query, f, indent=4)

new_drug_df.to_excel(excel_path_split, index=False, sheet_name='Sheet1')

print(f"New drug descriptions have been generated and saved to {json_path_combine} and {excel_path_split}")