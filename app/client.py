import requests
import json
import pandas as pd
import numpy as np

table = pd.read_csv("chinese-english.csv")

url = 'http://localhost:8001/compare_embedding_similarity'
payload = {
    'documents': [[row.english, row.chinese] for row in table.itertuples()],
    'language': 'multi',
    'type': 'document',
    'document_type': 'pooling',
    'embeddings': ['bert']
}

response = requests.get(url, data=json.dumps(payload), headers={'content-type':'text/plain'})

output = json.loads(response.content)
np.mean(output['distances'])