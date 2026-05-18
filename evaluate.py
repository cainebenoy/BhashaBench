import os
import requests
import json
import time

API_URL = "https://cainebenoy-bhashadocs-api.hf.space/api/translate-text"

# A mini benchmark dataset: English Source -> Malayalam Ground Truth
EVAL_DATA = [
    {
        "id": 1,
        "en": "The Harappan civilization was primarily an urban culture.",
        "ml_reference": "ഹാരപ്പൻ സംസ്കാരം പ്രധാനമായും ഒരു നഗര സംസ്കാരമായിരുന്നു."
    },
    {
        "id": 2,
        "en": "Artificial intelligence is changing the world very rapidly.",
        "ml_reference": "നിർമ്മിത ബുദ്ധി ലോകത്തെ വളരെ വേഗത്തിൽ മാറ്റുകയാണ്."
    },
    {
        "id": 3,
        "en": "We are building an automated testing pipeline today.",
        "ml_reference": "ഞങ്ങൾ ഇന്ന് ഒരു ഓട്ടോമേറ്റഡ് ടെസ്റ്റിംഗ് പൈപ്പ്ലൈൻ നിർമ്മിക്കുകയാണ്."
    }
]

def translate_via_bhashadocs(text, target_lang="mal_Mlym"):
    """
    Sends English text to the live BhashaDocs API for translation to an Indic language.
    """
    data = {
        'text': text,
        'target_language': target_lang
    }
    
    try:
        response = requests.post(API_URL, data=data, stream=True, timeout=60)
        
        if response.status_code != 200:
            print(f"⚠️ API Error Status: {response.status_code}")
            return None
            
        full_translation = []
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    chunk_data = json.loads(decoded_line)
                    
                    # Intercept telemetry errors from our new safe wrapper
                    if "error" in chunk_data:
                        print(f"❌ SERVER-SIDE ERROR: {chunk_data['error']}")
                        return None
                        
                    full_translation.append(chunk_data.get("translated", ""))
                except json.JSONDecodeError:
                    full_translation.append(decoded_line)
                    
        return " ".join(full_translation).strip()
        
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return None

def run_benchmark(target_lang="mal_Mlym"):
    """
    Runs the BhashaBench English-to-Indic evaluation pipeline.
    """
    print(f"🚀 Starting BhashaBench MT Evaluation...")
    print("==================================================================\n")
    
    for row in EVAL_DATA:
        print(f"📋 [Sample {row['id']}/3]")
        print(f"Source Text (EN): {row['en']}")
        
        print("⏳ Querying BhashaDocs API for translation...")
        start_time = time.time()
        
        # Translate English -> Malayalam
        model_translation = translate_via_bhashadocs(row['en'], target_lang=target_lang)
        
        elapsed = time.time() - start_time
        
        print(f"⏱️ API Latency: {elapsed:.2f} seconds")
        print(f"🤖 Model Output (ML): {model_translation}")
        print(f"🎯 Ground Truth (ML): {row['ml_reference']}")
        print("-" * 60 + "\n")
        
        time.sleep(2)

if __name__ == "__main__":
    # Test English to Malayalam
    run_benchmark(target_lang="mal_Mlym")