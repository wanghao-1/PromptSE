from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import validation


def m_sigmoid(x):
    """Return the sigmoid function result."""
    return 1 / (1 + np.exp(-x))


def vectorize(file_path, excel_path, cname):
    """
    Load text data from an Excel file, generate BioBERT embeddings for each row,
    and save the embeddings to a new Excel file.
    """
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print('load data from', file_path)

    # Extract sentences based on cname type
    if isinstance(cname, str):
        sentences = df[cname].tolist()
    elif isinstance(cname, list):
        sentences = df[cname].agg('. '.join, axis=1).tolist()
    else:
        raise TypeError("cname should be a string or a list")

    # Initialize tokenizer and model
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    model_name = "../dmis-lab/biobert-base-cased-v1.2"
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print('load model from local directory')

    # Optional linear layer to change embedding dimension (commented out)
    # embedding_dim = 512
    # embedding_layer = nn.Linear(768, embedding_dim)

    batch_size = 32
    all_sentence_embeddings_transformed = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        batch_sentences = [n if isinstance(n, str) else 'None' for n in batch_sentences]
        batch_sentences = [n if n == 'None' else n.replace('None', '') for n in batch_sentences]
        batch_index = [True if n == 'None' else False for n in batch_sentences]

        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # Use [CLS] token embedding as sentence representation
        sentence_embeddings = last_hidden_states[:, 0, :]
        # sentence_embeddings_transformed = embedding_layer(sentence_embeddings)
        sentence_embeddings_transformed = sentence_embeddings
        sentence_embeddings_transformed[batch_index] = torch.zeros_like(sentence_embeddings_transformed[0])

        all_sentence_embeddings_transformed.append(sentence_embeddings_transformed)

    if all_sentence_embeddings_transformed:
        all_sentence_embeddings_transformed = torch.cat(all_sentence_embeddings_transformed, dim=0)

    # Convert tensor to numpy and save
    sentence_embeddings_np = all_sentence_embeddings_transformed.detach().numpy()
    df = pd.DataFrame(sentence_embeddings_np)
    df.to_excel(excel_path, index=False)
    return sentence_embeddings_np





