import os
import pandas as pd
from datasets import load_dataset

def fetch_indicqa_data(language_code="ml"):
    """
    Pulls the IndicQA evaluation benchmark from Hugging Face.
    Supported codes: as, bn, gu, hi, kn, ml, mr, od, pa, ta, te
    """
    print(f"📥 Fetching IndicQA dataset for language: '{language_code}'...")
    
    try:
        # Load the specific language subset from the AI4Bharat benchmark
        dataset = load_dataset("ai4bharat/IndicQA", language_code, split="test")
        df = dataset.to_pandas()
        
        print(f"✅ Successfully loaded {len(df)} QA pairs.")
        
        # SQuAD format nests the text inside a dictionary. Let's extract it cleanly.
        # It looks like: {'text': ['Answer string'], 'answer_start': [123]}
        df['reference_answer'] = df['answers'].apply(lambda x: x['text'][0] if len(x['text']) > 0 else "")
        
        # Save locally for faster access during the inference loop
        os.makedirs("data", exist_ok=True)
        save_path = f"data/indicqa_{language_code}.csv"
        
        # We only need these core columns for the benchmark
        eval_df = df[['id', 'context', 'question', 'reference_answer']]
        eval_df.to_csv(save_path, index=False)
        
        print(f"💾 Cleaned and saved locally to {save_path}")
        
        # Display the first example to verify the structure
        print("\n--- Data Structure Check ---")
        print(f"Context: {eval_df.iloc[0]['context'][:100]}...")
        print(f"Question: {eval_df.iloc[0]['question']}")
        print(f"Reference Answer: {eval_df.iloc[0]['reference_answer']}")
        print("----------------------------\n")
        
        return eval_df
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None

if __name__ == "__main__":
    # Test the ingestion layer with Malayalam
    fetch_indicqa_data("ml")