from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def bert_cosine_similarity(sent1, sent2):
    embeddings = model.encode([sent1, sent2])
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(cosine_sim)
