import os
import requests
import json
import time
import sacrebleu

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
    Runs the BhashaBench English-to-Indic evaluation pipeline and calculates chrF scores.
    """
    print(f"🚀 Starting BhashaBench MT Evaluation with chrF Scoring...")
    print("==================================================================\n")
    
    predictions = []
    references = []
    
    for row in EVAL_DATA:
        print(f"📋 [Sample {row['id']}/3]")
        print(f"Source Text (EN): {row['en']}")
        
        print("⏳ Querying BhashaDocs API for translation...")
        start_time = time.time()
        
        model_translation = translate_via_bhashadocs(row['en'], target_lang=target_lang)
        elapsed = time.time() - start_time
        
        if not model_translation:
            model_translation = ""
            
        print(f"⏱️ API Latency: {elapsed:.2f} seconds")
        print(f"🤖 Model Output (ML): {model_translation}")
        print(f"🎯 Ground Truth (ML): {row['ml_reference']}")
        print("-" * 60 + "\n")
        
        predictions.append(model_translation)
        references.append(row['ml_reference'])
        
        time.sleep(2)
        
    print("📊 Calculating Batch Metrics...")
    # Calculate chrF score (Character n-gram F-score)
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    
    print("\n==================================================================")
    print(f"🏆 Final Batch chrF Score: {chrf.score:.2f} / 100")
    print("==================================================================\n")

if __name__ == "__main__":
    run_benchmark(target_lang="mal_Mlym")