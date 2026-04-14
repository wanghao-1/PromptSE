python train_PromptSE_cold.py \
  --split_mode custom_drug \
  --custom_test_drugs cold_drug_effect_exports/cold_drug_effect_fold01_seed10.json \
  --save_model
python train_PromptSE_cold.py \
  --split_mode custom_drug \
  --custom_test_drugs cold_drug_effect_exports/cold_drug_effect_fold01_seed10.json \
  --save_model \
  --no_pretrain