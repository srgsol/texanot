"""
This module contains similarity index functionality.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset


class SimilarityEngine:
    """Search on the vector space. Embedding with Transformers.

    Use Hugging-Face pre-trained model to extract embeddings from text and
    build a FAISS index. Search similar texts in the vector space.
    """
    def __init__(self, checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.embeddings_dataset = None
        
    def cls_pooling(self, model_output):
        """Extractes (pooling) on the last hidden state of the model output.

        This method performs pooling on the last hidden state of the model
        output to obtain a fixed-size representation for the input text. It
        returns the pooled representation for the [CLS] token."""
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        """Generate embedding from the given list of texts"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)
    
    def index(self, candidates: Dataset, field_name):
        """Build a FAISS index.

        Extracts embeddings from the given dataset and adds them to the index.
        """
        self.embeddings_dataset = candidates.map(lambda x: {
            "embeddings": self.get_embeddings(x[field_name]).detach().cpu().numpy()[0]},
        )
        self.embeddings_dataset.add_faiss_index(column="embeddings")
    
    def similar(self, text: str, n: int):
        """Given a text, serach N similar texts in the index."""
        question_embedding = self.get_embeddings([text]).cpu().detach().numpy()
        scores, samples = self.embeddings_dataset.get_nearest_examples(
            "embeddings", question_embedding, k=n)
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=True, inplace=True)
        return samples_df

    def load(self, df, colname, filename):
        """Loads a FAISS index.

        It requires the index and de text used to build the index. That allows
        us to retrieve the original text given an index item.
        """
        self.embeddings_dataset = Dataset.from_pandas(pd.DataFrame(df))
        self.embeddings_dataset.load_faiss_index(colname, filename)
