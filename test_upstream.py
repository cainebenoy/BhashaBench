import requests

# The URL of your new inbound microservice
UPSTREAM_API_URL = "https://cainebenoy-bhashadocs-inbound-api.hf.space/api/translate-inbound"

def test_inbound():
    malayalam_question = "ഹാരപ്പൻ സംസ്കാരം ഒരു നഗര സംസ്കാരമായിരുന്നോ?" # "Was the Harappan civilization an urban culture?"
    
    print(f"📡 Sending Malayalam question: {malayalam_question}")
    
    data = {
        "text": malayalam_question,
        "source_lang": "mal_Mlym"
    }
    
    response = requests.post(UPSTREAM_API_URL, data=data)
    
    if response.status_code == 200:
        print(f"🟢 Success! English Output: {response.json()['english_text']}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_inbound()