import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset


class SimilarityEngine:
    def __init__(self, checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        # self.candidates_dataset = None
        self.embeddings_dataset = None
        
    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)
    
    def index(self, candidates: Dataset, field_name):
        self.embeddings_dataset = candidates.map(lambda x: {
            "embeddings": self.get_embeddings(x[field_name]).detach().cpu().numpy()[0]},
        )
        self.embeddings_dataset.add_faiss_index(column="embeddings")
    
    def similar(self, text, n):
        question_embedding = self.get_embeddings([text]).cpu().detach().numpy()
        scores, samples = self.embeddings_dataset.get_nearest_examples(
            "embeddings", question_embedding, k=n)
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=True, inplace=True)
        return samples_df

    def load(self, df, colname, filename):
        self.embeddings_dataset = Dataset.from_pandas(pd.DataFrame(df))
        self.embeddings_dataset.load_faiss_index(colname, filename)

# class IndexLoader:
#     def __init__(self, df: pd.DataFrame):
#         index_dataset = Dataset.from_pandas(pd.DataFrame(df))