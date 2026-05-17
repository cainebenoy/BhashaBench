import pandas as pd
import requests
import json
import time
import os

# Update this to your exact live Hugging Face Space URL
API_URL = "https://cainebenoy-bhashadocs-api.hf.space/api/translate-doc"

def translate_via_bhashadocs(text, target_lang="eng_Latn"):
    """
    Sends a string to the live BhashaDocs API for translation.
    Because our API expects a file upload, we fake a text file in-memory.
    """
    # Create an in-memory "file" containing our sentence
    files = {
        'file': ('text.txt', text, 'text/plain')
    }
    data = {
        'target_language': target_lang
    }
    
    try:
        # Since it streams, we read the response chunks as they arrive
        response = requests.post(API_URL, files=files, data=data, stream=True, timeout=60)
        
        if response.status_code != 200:
            print(f"⚠️ API Error Status: {response.status_code}")
            return None
            
        full_translation = []
        
        # Read the Server-Sent Events stream
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # If the line contains JSON stream fragments, parse out the translated text
                try:
                    # Our API streams JSON formats or direct text. Adjust parsing based on exact stream syntax:
                    # If your API outputs raw text rows, use: full_translation.append(decoded_line)
                    # If it outputs JSON chunks like {"translated": "..."}, parse it here:
                    chunk_data = json.loads(decoded_line)
                    full_translation.append(chunk_data.get("translated", ""))
                except json.JSONDecodeError:
                    # Fallback if raw text or prefix metadata is sent
                    full_translation.append(decoded_line)
                    
        return " ".join(full_translation).strip()
        
    except Exception as e:
        print(f"❌ Network/Timeout Error: {e}")
        return None

def run_benchmark(lang_code="ml", sample_size=3):
    """
    Loads the ingested data and runs the BhashaDocs translation evaluation pipeline.
    """
    csv_path = f"data/indicqa_{lang_code}.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Local data file not found at {csv_path}. Run ingest.py first!")
        return
        
    df = pd.read_csv(csv_path)
    print(f"🚀 Starting BhashaBench evaluation loop on {sample_size} samples...")
    print("==================================================================\n")
    
    # Take a small slice for testing
    test_slice = df.head(sample_size)
    
    for idx, row in test_slice.iterrows():
        print(f"📋 [Sample {idx + 1}/{sample_size}] - ID: {row['id']}")
        print(f"Native Question (ML): {row['question']}")
        
        # 1. Translate Native Indic Question -> English
        print("⏳ Querying BhashaDocs API for English translation...")
        start_time = time.time()
        english_question = translate_via_bhashadocs(row['question'], target_lang="eng_Latn")
        elapsed = time.time() - start_time
        
        print(f"⏱️ API Latency: {elapsed:.2f} seconds")
        print(f"Translated Question (EN): {english_question}")
        print(f"Ground Truth Answer (ML): {row['reference_answer']}")
        print("-" * 50 + "\n")
        
        # Politeness delay to avoid overloading free tier resources rapidly
        time.sleep(2)

if __name__ == "__main__":
    run_benchmark(lang_code="ml", sample_size=3)