# Silence ignorable UNEXPECTED embeddings.position_ids warning
from transformers.utils import logging as logging
logging.set_verbosity_error()
from config_data import config_data
from sentence_transformers import SentenceTransformer, util
from device_manager import DeviceManager

default_model = "paraphrase-MiniLM-L6-v2"  # Fallback in case no model is specified
model_for_text_similarity = config_data.get("model_for_text_similarity", default_model)

device_manager = DeviceManager()
device = device_manager.device

model = SentenceTransformer(model_for_text_similarity, device=str(device))

def bert_cosine_similarity(sent1, sent2):
    embeddings = model.encode([sent1, sent2], convert_to_tensor=True, device=str(device))
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(cosine_sim)
