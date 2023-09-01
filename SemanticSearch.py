import os
import openai
import pandas as pd
import numpy as np
from ast import literal_eval

openai.api_key = "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"

datafile_path = "/Users/isaackim/Python/TestDataSets/data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# Here we compare the cosine similarity of the embeddings of the query and the documents, and show top_n best matches.

from openai.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    # obtaining the embedding values for the product descr
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    # create a similarity column that holds all the similarity values from cosine sim w embed vals and prod desc embed
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    # get the n most sim in a list
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )

    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


results = search_reviews(df, "Corn", n=3)