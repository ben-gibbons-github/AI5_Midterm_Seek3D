import os
from typing import List
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
import json
import torch
import open_clip
import numpy as np
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-H/14-378"
pretrained = "dfn5b"
clip_model, nuttin, clip_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
clip_model = clip_model.to(device)

saved_model_data_filename = './saved_model_data.json'
existing_image_features_filename = './existing_image_features_big.pt'
existing_text_features_filename = "./existing_text_features_big.pt"

with open(saved_model_data_filename, 'r') as json_file:
    saved_model_data = json.load(json_file)
    existing_searches_done = saved_model_data["existing_searches_done"]
    list_of_all_models = saved_model_data["list_of_all_models"]
    index_to_model_uid = saved_model_data["index_to_model_uid"]
    print("index_to_model_uid", len(index_to_model_uid))
existing_image_features = torch.load(existing_image_features_filename)
print("existing_image_features", existing_image_features.shape)
existing_text_features = torch.load(existing_text_features_filename)
print("existing_text_features", existing_text_features.shape)

def zscore(similarity):
    return (similarity - similarity.mean()) / similarity.std()

def load_model_from_vec(objectName):

    pos_prompts = [objectName, "cohesive attractive design", 'high poly high quality', 'attractive and pretty colors', 'cohesive single object', 'object with a solid base']
    pos_weights=[1.1, 0.2, 0.2, 0.2, 0.4, 0.2]
 
    neg_prompts=[f"object in a room", "a complete room", 'ugly or low quality', 'low poly', 'abstract and disconnected', 'black and white', 'greyscale'] 
    neg_weights=[0.3, 0.3, 0.3, 0.3, 0.2, 0.1, 0.1]
    
    print("load_model_from_vec - Positive Prompts:", pos_prompts)
    print("load_model_from_vec - Negative Prompts:", neg_prompts)
    
    # ----- Compute positive embeddings -----
    pos_tokens = open_clip.tokenize(pos_prompts).to(device)
    with torch.no_grad():
        pos_embeddings = clip_model.encode_text(pos_tokens).to(device)
        pos_embeddings = pos_embeddings / pos_embeddings.norm(dim=-1, keepdim=True)
    
    # Weight and combine the positive embeddings.
    weighted_pos = torch.stack([w * emb for w, emb in zip(pos_weights, pos_embeddings)], dim=0)
    combined_pos = weighted_pos.sum(dim=0)
    
    # ----- Compute negative embeddings (if any) -----
    if neg_prompts:
        neg_tokens = open_clip.tokenize(neg_prompts).to(device)
        with torch.no_grad():
            neg_embeddings = clip_model.encode_text(neg_tokens).to(device)
            neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=-1, keepdim=True)
        weighted_neg = torch.stack([w * emb for w, emb in zip(neg_weights, neg_embeddings)], dim=0)
        combined_neg = weighted_neg.sum(dim=0)
    else:
        # If no negative prompts are provided, use a zero tensor of the same shape.
        combined_neg = torch.zeros_like(combined_pos)
    
    # ----- Combine positive and negative embeddings -----
    combined_text = combined_pos - combined_neg
    combined_text = combined_text / combined_text.norm()
    
    # ----- Compute similarity with existing image features -----
    # (Assuming existing_image_features is a [N x D] tensor where D is the embedding dim.)
    combined_similarity = (existing_image_features @ combined_text.T).squeeze()
    combined_similarity = zscore(combined_similarity.cpu().numpy())

    # Get the best match index based on the combined similarity
    num_top_models = 4
    
    top_X_indices = np.argsort(-combined_similarity)[:num_top_models]

    top_X_models = [list_of_all_models[index_to_model_uid[idx]] for idx in top_X_indices]
    
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Clear GPU memory

    return top_X_models


@cl.on_chat_start
async def on_chat_start():
    
    # Create a dict vector store
    vector_db = VectorDatabase()
    
    # Let the user know that the system is ready
    msg = cl.Message(
        content=f"Enter the name of a 3D model to search for.  It can be a short filename or a long description (" + str(len(list_of_all_models)) + " models in database)"
    )
    await msg.send()

    # cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


@cl.on_message
async def main(message):
    # chain = cl.user_session.get("chain")
    # result = await chain.arun_pipeline(message.content)

    best_models = load_model_from_vec(message.content)

    msg = cl.Message(
        content="Searching..."
    )
    await msg.send()

    the_elements = []
    for model in best_models:
        the_elements.append(cl.Image(name=model["name"], url=model["image"]))
        the_elements.append(cl.Text(content="[" + model["name"] + "](https://sketchfab.com/models/" + model['uid'] + "/)"))

    # msg = cl.Message(content=message.content)
    msg = cl.Message(
        content="Models loaded:",
        elements=the_elements
    )

    await msg.send()