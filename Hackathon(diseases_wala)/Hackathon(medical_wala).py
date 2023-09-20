#pip install sentence_transformers
#pip install google
#pip install numpy

from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import json
try:
    from googlesearch import search
except ImportError:
    print("No module named 'google' found")
 

path = r"disease_symptoms.json"
array = list(open(path,'r'))
something = list()
i = 0
while i < len(array):
    something.append(array[2::4])
    i+=1
disease_name = list()
for i in range(len(something[0])):
    disease_name.append(something[0][i])

something2 = list()
i = 0
while i < len(array):
    something2.append(array[3::4])
    i+=1
disease_symptom = list()
for i in range(len(something2[0])):
    disease_symptom.append(something2[0][i])

model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

docs = disease_symptom

query = r"mild fever, cold, cough, dizzyness, uncomfortable"

query_emb = model.encode(query)
#doc_emb = model.encode(docs)

#doc_emb = model.encode(filtered_json)
#print("After corpus_embedding")
#filtered_json = np.load("filtered_json.npy")
#all_embeddings = doc_emb
#all_embeddings = np.array(all_embeddings)
#np.save('Hackathon(medical_wala).npy', all_embeddings)

doc_emb = np.load(r"Hackathon(medical_wala).npy")

#Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

disease_list = list()
for i in range(3):
    disease_list.append(disease_name[disease_symptom.index(doc_score_pairs[i][0])])

disease_symptom_list = list()
for i in range(3):
    disease_symptom_list.append(doc_score_pairs[i])

for i in range(3):
    print("\n",disease_list[i])
    print(disease_symptom_list[i])
    query = disease_list[i]

    for j in search(query, tld="co.in", num=3, stop=3, pause=2):
        print("For more Information :", j)
        time.sleep(1)

