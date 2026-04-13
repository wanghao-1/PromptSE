## LLM-Generated Text

step 1. 
Goal: Generate a core description of each drug based only on its basic information, without providing any side-effect information.
Data used:
	- Required: data_pt/se_id.xlsx provides drug names; data_pt/drug_definition.xlsx provides drug target information
Code: tongyi_drug_s1.py
Output for each drug: "administration route", "site of action", "distribution of targets", "structural features", "metabolic pathways"
Output files:
	- 'data_drug_s1/drug_0403_combine_n1020_plus_s1.json': all drug information written into a single string
	- 'data_drug_s1/drug_0403_split_n1020_plus_s1.xlsx': different information fields written into separate columns of an Excel file
	- 'data_drug_s1/drug_0403_query_n1020_plus_s1.json': the query for each drug

step 2.
Goal: For each (drug, side effect) pair, generate the reason why the drug causes the side effect, explained in four categories: "administration route", "target selectivity", "structural properties", and "metabolism pathway".
Data used:
	data_pt/se_id.xlsx: provides side effect names
	data_pt/drug_eff.xlsx: training adjacency matrix
	'data_pt/se_definition.xlsx': side effect definitions
	'data_drug_s1/drug_0403_combine_n1020_plus_s1.json': drug information generated in step 1
Code: tongyi_ds_s2.py
Output: For each (drug, side effect) pair, the category and specific explanation of the mechanism.
Output file: data_drug_se_s2/ds_0220_ns5599_nd1020_r20_plus.json

step 3
Goal: Summarize the causes of side effects based on the (drug, side effect) pairs.
Data used: data_drug_se_s2/ds_0403_ns5599_nd1020_r20_plus.json
Code: tongyi_se_s3_split.py
Output: For each side effect, a description of the causes from the four perspectives: "administration route", "target selectivity", "structural properties", and "metabolism pathway".
Output file: ds_0403_ns5599_nd1020_r20_plus_ca+ex_split.xlsx


## Text to Vector Embeddings
Code: vec_all_split.py
Convert the text in the four categories ("administration route", "target selectivity", "structural properties", "metabolism pathway") into vectors, then sum them. If the text is "None", it is padded with a zero vector.


