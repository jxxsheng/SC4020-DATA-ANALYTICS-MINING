import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.cluster import KMeans
import re
# Path to kaggle image dataset for hs: C:\Users\User\.cache\kagglehub\datasets\adityajn105\flickr8k\versions\1

# Define a singular orb instance
orb = cv2.ORB_create(nfeatures = 500, scaleFactor = 1.1, nlevels = 10, fastThreshold=2)

def semantic_chunk_dataset(article_list, embeddings_model=None):
    """
    Optional semantic chunking.
    Returns a list of chunks with metadata.
    """
    metadatas = [{"article_id": i} for i in range(len(article_list))]
    
    # If no embeddings model provided, default to Ollama
    embeddings_model = embeddings_model or OllamaEmbeddings(model="mxbai-embed-large")
    chunker = SemanticChunker(embeddings=embeddings_model, breakpoint_threshold_type="percentile")
    
    chunks = chunker.create_documents(article_list, metadatas=metadatas)
    return chunks

def prepare_chunks_for_similarity(chunks):
    """
    For each chunk, create:
      - token_set: for Jaccard similarity
      - cleaned_text: for cosine similarity (embedding)
    """
    processed_chunks = []
    
    for chunk in chunks:
        text = chunk.page_content
        
        # 1. Clean text
        text_clean = text.lower()
        text_clean = re.sub(r"[^\w\s]", "", text_clean)  # remove punctuation
        tokens = text_clean.split()
        token_set = set(tokens)
        cleaned_text = " ".join(tokens)  # for embedding
        
        processed_chunks.append({
            "metadata": chunk.metadata,
            "original_text": text,
            "token_set": token_set,       # Jaccard
            "cleaned_text": cleaned_text  # Cosine
        })
    
    return processed_chunks

def embed_chunks_for_cosine(processed_chunks, embeddings_model=OllamaEmbeddings(model="mxbai-embed-large")):
    
    for chunk in processed_chunks:
        embedding_vector = embeddings_model.embed_query(chunk["cleaned_text"])
        chunk["embedding"] = embedding_vector
    
    return processed_chunks


# def embed_dataset(dataset_path):
#     df = pd.read_csv(dataset_path)
#     article_texts = df["Article text"]
#     article_list = article_texts.dropna().tolist()
#     metadatas = [{"article_id": i} for i in range(len(article_list))]

#     article_list = article_list[:3]
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#     chunker = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type = "percentile")
#     chunks = chunker.create_documents(article_list, metadatas=metadatas)

#     chunk_array = []
#     for chunk in chunks:
#         embedding_vector = embeddings.embed_query(chunk.page_content)
#         chunk_array.append({
#         "text": chunk.page_content,
#         "embedding": embedding_vector,
#         "metadata": chunk.metadata
#         })
    
#     return chunk_array

def print_processed_chunks(processed_chunks, k=3):
    """
    Print the first k chunks with all relevant info for inspection.
    """
    chunk_array_temp = processed_chunks[:k]
    
    for i, c in enumerate(chunk_array_temp):
        print(f"--- Chunk {i+1} ---")
        print("Original Text:", c["original_text"][:200], "..." if len(c["original_text"]) > 200 else "")
        print("Cleaned Text:", c["cleaned_text"][:200], "..." if len(c["cleaned_text"]) > 200 else "")
        print("Token Set:", list(c["token_set"])[:20], "..." if len(c["token_set"]) > 20 else "")  # show first 20 tokens
        if "embedding" in c:
            print("Embedding Vector (first 10 values):", c["embedding"][:10])
            print("Embedding Length:", len(c["embedding"]))
        print()

def extract_descriptors(img):
    """Extract ORB descriptors from an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors

def extract_all_descriptors(image_files):
    all_descriptors = []
    for file in tqdm(image_files, desc="Extracting ORB features"):
        img = cv2.imread(file)
        if img is None:
            continue  # skip if any file is corrupted or unreadable
        keypoints, desc = extract_descriptors(img)
        if desc is not None:
            all_descriptors.append(desc)
    print(f"Extracted descriptors for {len(all_descriptors)} images.")

    return all_descriptors


def visualize_kp_on_img(sample_img_path):
    # Step 5: Visualize keypoints on one sample image
    sample_img = cv2.imread(sample_img_path)
    sample_kp, sample_desc = extract_descriptors(sample_img)

    img_with_kp = cv2.drawKeypoints(
        sample_img,
        sample_kp,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Convert to RGB for display with matplotlib
    img_with_kp = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))
    plt.imshow(img_with_kp)
    plt.title(f"Keypoints on sample image\n{os.path.basename(sample_img_path)}")
    plt.axis("off")
    plt.show()

def compare_image(image_path, kmeans_model, num_clusters, reference_histograms, reference_files):
    # Step 1: Load and extract ORB descriptors from query image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be read.")

    keypoints, descriptors = orb.detectAndCompute(img.astype(np.uint8), None)
    if descriptors is None or len(descriptors) == 0:
        raise ValueError("No descriptors found in the query image.")

    # Step 2: Assign descriptors to clusters
    cluster_assignments = kmeans_model.predict(descriptors)

    # Step 3: Build histogram (counts of cluster assignments)
    hist, _ = np.histogram(cluster_assignments, bins=np.arange(num_clusters + 1))
    binary_hist = (hist > 0).astype(int)

    # Step 4: Compute Jaccard similarity with all reference histograms
    similarities = [
        jaccard_similarity_manual(binary_hist, ref_hist)
        for ref_hist in reference_histograms
    ]

    # Step 5: Get top 5 most similar images
    top_indices = np.argsort(similarities)[::-1][:5]
    
    print("Top 5 similar images:")
    for idx in top_indices:
        print(f"{reference_files[idx]} (similarity: {similarities[idx]:.3f})")

    # Step 6: Plot query image + top results
    plt.figure(figsize=(16, 8))

    # --- Query image ---
    plt.subplot(2, 3, 1)
    query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')

    # --- Top 5 similar images ---
    for i, idx in enumerate(top_indices):
        img_sim = cv2.imread(reference_files[idx])
        if img_sim is None:
            continue
        img_sim = cv2.cvtColor(img_sim, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i + 2)
        plt.imshow(img_sim)
        plt.title(f"Sim: {similarities[idx]:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def jaccard_similarity_manual(a, b):
    """Compute Jaccard similarity for binary histograms."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / union if union != 0 else 0

def fit_Kmeans_model(all_descriptors_list, num_clusters):
    all_descriptors_flat = np.vstack(all_descriptors_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_descriptors_flat)
    return kmeans

def build_histograms(descriptor_list, kmeans_model, num_clusters):
    histograms = []
    for desc in descriptor_list:
        if desc is None:
            histograms.append(np.zeros(num_clusters))
        cluster_assignments = kmeans_model.predict(desc)
        hist, _ = np.histogram(cluster_assignments, bins = np.arange(num_clusters + 1))
        histograms.append(hist)
    return np.array(histograms)
