import requests
import json
import time
import sacrebleu

API_URL = "https://cainebenoy-bhashadocs-api.hf.space/api/translate-text"

# 10 Real-world sentences simulating news, weather, and general web text
OPUS_MOCK_DATA = [
    {"en": "The government has announced a new health policy.", "ml": "സർക്കാർ പുതിയ ആരോഗ്യ നയം പ്രഖ്യാപിച്ചു."},
    {"en": "Climate change is a global crisis.", "ml": "കാലാവസ്ഥാ വ്യതിയാനം ഒരു ആഗോള പ്രതിസന്ധിയാണ്."},
    {"en": "He was admitted to the hospital due to a severe fever.", "ml": "കടുത്ത പനി കാരണം അദ്ദേഹത്തെ ആശുപത്രിയിൽ പ്രവേശിപ്പിച്ചു."},
    {"en": "The stock market crashed today.", "ml": "ഓഹരി വിപണി ഇന്ന് തകർന്നു."},
    {"en": "Education is the key to success.", "ml": "വിദ്യാഭ്യാസമാണ് വിജയത്തിന്റെ താക്കോൽ."},
    {"en": "The project will be completed by next month.", "ml": "പദ്ധതി അടുത്ത മാസം പൂർത്തിയാകും."},
    {"en": "She won the gold medal in the Olympics.", "ml": "ഒളിമ്പിക്സിൽ അവർ സ്വർണ്ണ മെഡൽ നേടി."},
    {"en": "The police arrested the suspect yesterday.", "ml": "പ്രതിയെ പോലീസ് ഇന്നലെ അറസ്റ്റ് ചെയ്തു."},
    {"en": "Heavy rain is expected in Kerala tomorrow.", "ml": "നാളെ കേരളത്തിൽ കനത്ത മഴയ്ക്ക് സാധ്യതയുണ്ട്."},
    {"en": "The new smartphone features an advanced camera system.", "ml": "പുതിയ സ്മാർട്ട്ഫോണിൽ നൂതന ക്യാമറ സംവിധാനമുണ്ട്."}
]

def translate_via_bhashadocs(text, target_lang="mal_Mlym"):
    """Sends English text to the live BhashaDocs API for translation."""
    data = {'text': text, 'target_language': target_lang}
    try:
        response = requests.post(API_URL, data=data, stream=True, timeout=60)
        if response.status_code != 200:
            return None
            
        full_translation = []
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    chunk_data = json.loads(decoded_line)
                    if "error" in chunk_data:
                        return None
                    full_translation.append(chunk_data.get("translated", ""))
                except json.JSONDecodeError:
                    full_translation.append(decoded_line)
        return " ".join(full_translation).strip()
    except Exception:
        return None

def run_real_world_benchmark():
    sample_size = len(OPUS_MOCK_DATA)
    print(f"🚀 Starting BhashaBench MT Evaluation on {sample_size} real-world samples...")
    print("==================================================================\n")
    
    predictions = []
    references = []
    
    for i, row in enumerate(OPUS_MOCK_DATA):
        source_en = row['en']
        ref_ml = row['ml']
        
        print(f"📋 [Sample {i + 1}/{sample_size}]")
        print(f"Source Text (EN): {source_en}")
        
        start_time = time.time()
        model_translation = translate_via_bhashadocs(source_en, target_lang="mal_Mlym")
        elapsed = time.time() - start_time
        
        if not model_translation:
            model_translation = ""
            
        print(f"⏱️ API Latency: {elapsed:.2f} seconds")
        print(f"🤖 Model Output: {model_translation}")
        print(f"🎯 Ground Truth: {ref_ml}")
        print("-" * 60 + "\n")
        
        predictions.append(model_translation)
        references.append(ref_ml)
        time.sleep(2)
        
    print("📊 Calculating Batch Metrics...")
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    
    print("\n==================================================================")
    print(f"🏆 Final Real-World chrF Score: {chrf.score:.2f} / 100")
    print("==================================================================\n")

if __name__ == "__main__":
    run_real_world_benchmark()