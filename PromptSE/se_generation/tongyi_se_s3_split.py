from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd
import time
import json
import pickle
import traceback
import numpy as np

# ==================== Configuration ====================
max_workers = 20
modeltype = 'plus'
model = "qwen-plus-2025-12-01"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "xxxxxxxxxxxxxxx"

# ==================== Load Side Effect Data ====================
s1_path_se = 'data_pt/se_definition.xlsx'
side_effects_df = pd.read_excel(s1_path_se, sheet_name='Sheet1')
side_effects_df.set_index('side effect', inplace=True)

# ==================== LLM API Call Function ====================
def call_qwen_api(query, api_key, base_url, retry_delay=10, max_retries=3):
    """
    Call the Qwen LLM API with retry logic for rate-limit errors (429).
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
                ],
            )
            print('Completed one request')
            jstr = completion.choices[0].message.content
            print('before:', jstr)
            jstr_cleaned = jstr.strip().strip("```").strip("'''").strip("json").strip()
            return jstr, jstr_cleaned
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
                return "None", "None"

# ==================== Query Template ====================
QUERY = '''\
Please summarize the causes of the side effect "{side_effect}" (defined as: "{definition}") based on \
    the provided explanations of how specific drugs cause "{side_effect}": "{listed_exps}".
The causes fall into four categories: administration route, metabolism pathway, target selectivity, \
    and structural properties. Categorize each cause and provide a concise summary per category:
    - If the cause is **administration route**, list only the specific route (e.g., Ophthalmic) causing "{side_effect}" (max 5 words). Do not include additional explanations or effects.
    - If the cause is **metabolism pathway**, summarize the metabolic organs/systems and the types of metabolites (e.g., Liver, Oxidative Products) (max 15 words). \
        Use specific metabolite categories (e.g., Pyrimidine Metabolites, Oxidative Products) rather than specific metabolite names.
    - If the cause is **target selectivity**, summarize the receptor types and their locations (organ/system) (e.g., Adrenergic Receptor, digestive system) (max 15 words). Do not include specific receptor names.
    - If the cause is **structural properties**, summarize the drug's structural features (e.g., Lipophilicity) (max 10 words).
    - If any category is absent, respond with "None".
    - If how the drug causes "{side_effect}" is unclear, guess the most likely cause based on the "{definition}",

**Requirements:**
1. Only include factors contributing to "{side_effect}".
2. Your response must be in JSON format with the keys: "administration route", "metabolism pathway", "target selectivity", and "structural properties".
3. Do not include any information outside of the JSON.
4. Identify at least one cause for "{side_effect}".
5. Avoid redundancy and ensure high information density.
6. Respond in English and avoid abbreviations.
7. When answering regarding systems, use one of the following: "musculoskeletal system, circulatory system, \
    respiratory system, digestive system, urinary system, nervous system, endocrine system, reproductive system".
'''

# ==================== Process a Single Side Effect ====================
def generate_description(side_effect, listed_analysis):
    """Build query, call LLM, parse JSON, and return structured data."""
    analysis = " ".join(listed_analysis)
    definition = side_effects_df.loc[side_effect, 'definition']
    query = QUERY.format(side_effect=side_effect, definition=definition, listed_exps=analysis)

    jstr, jstr_cleaned = call_qwen_api(query, api_key, base_url)

    if jstr_cleaned:
        try:
            json_dict = json.loads(jstr_cleaned)
            return side_effect, listed_analysis, jstr, json_dict
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Invalid JSON string: {jstr_cleaned}")
            return side_effect, listed_analysis, jstr, {}

    return side_effect, listed_analysis, "None", {}

# ==================== Parse Intermediate JSON ====================
def job3_parse_query(json_path):
    """
    Load a JSON file containing drug-wise analyses for each side effect,
    extract the 'analysis' text from each entry, and group by side effect.
    Returns a dict: side_effect -> list of analysis strings.
    """
    raw_data = []
    with open(json_path, 'r', encoding='utf-8') as fp:
        total_outputs = json.load(fp)
        for side_effect, outputs in total_outputs.items():
            for drug, json_str in outputs.items():
                print('---', drug, '---')
                print(json_str)
                _dict = json.loads(json_str)
                category = _dict['category']
                if '' in category:
                    analysis = _dict['category'] + ':' + _dict['explanation']
                raw_data.append([side_effect, drug, analysis])

    # Collect unique side effects
    side_effects = set()
    print('len(raw_data):', len(raw_data))
    for side_effect, drug, analysis in raw_data:
        side_effects.add(side_effect)
    print('len(side_effects):', len(side_effects))

    # Group analyses by side effect
    new_data = {side_effect: [] for side_effect in side_effects}
    for side_effect, drug, analysis in raw_data:
        new_data[side_effect].append([drug, analysis])

    # Keep only the analysis strings (ignore drug names)
    new_data2 = {}
    for side_effect, data in new_data.items():
        list_analysis = [x[1] for x in data]
        if len(list_analysis) > 0:
            new_data2[side_effect] = list_analysis

    return new_data2

# ==================== Main Processing Function ====================
def job1_model_query(se_selected, input_path, save_path, max_workers=10):
    """
    For each selected side effect, aggregate existing analyses (from input_path)
    and call the LLM to produce a categorized summary. Results are saved as Excel.
    """
    # Parse intermediate results to get per‑side‑effect analysis lists
    new_data2 = job3_parse_query(input_path)
    print(len(new_data2))

    # Ensure all requested side effects are present (fill missing with empty list)
    for side_effect in se_selected:
        if side_effect not in new_data2:
            new_data2[side_effect] = ""

    print(len(new_data2))

    # Prepare output DataFrame
    new_se_df = pd.DataFrame(columns=["side effect", "listed explanations", "summary",
                                      "administration route", "metabolism pathway",
                                      "target selectivity", "structural properties"])

    # Parallel processing of side effects
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_description, side_effect, analysis): (side_effect, analysis)
                   for side_effect, analysis in new_data2.items()}

        for future in as_completed(futures):
            side_effect, analysis, jstr, json_dict = future.result()

            len_df = len(new_se_df)
            print(len_df)
            new_se_df.loc[len_df] = {
                "side effect": side_effect,
                "listed explanations": analysis,
                "summary": jstr.strip("`' json \n\t{}").strip(),
                "administration route": json_dict.get("administration route", "None"),
                "metabolism pathway": json_dict.get("metabolism pathway", "None"),
                "target selectivity": json_dict.get("target selectivity", "None"),
                "structural properties": json_dict.get("structural properties", "None")
            }

    new_se_df.to_excel(save_path, index=False, sheet_name='Sheet1')

# ==================== Entry Point ====================
if __name__ == '__main__':
    with open('data_drug_se_s2/ds_0403_ns5599_nd1020_r20_plus.json', 'r') as f:
        total_outputs = json.load(f)
    se_selected = list(total_outputs.keys())
    print(len(se_selected))

    # Paths for input (stage‑2) and output (stage‑3)
    num_se = 5599
    filepath = f'data_drug_se_s2/ds_0403_ns{num_se}_nd1020_r20_'
    save_path = f'data_se_s3/ds_0403_ns{num_se}_nd1020_r20_ca+ex_split.xlsx'
    in_path = filepath + 'plus.json'

    job1_model_query(se_selected, input_path=in_path, save_path=save_path)


