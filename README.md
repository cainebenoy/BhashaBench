# BhashaBench

## 🧪 MLOps & Evaluation Pipeline

To ensure the BhashaDocs API delivers production-grade, state-of-the-art translation accuracy, this project includes **BhashaBench**—a custom MLOps evaluation framework.

Rather than relying on basic string matching, BhashaBench stress-tests the live asynchronous API against the industry-standard **OPUS-100** corpus (English-Malayalam) and mathematically scores the semantic accuracy using the **chrF (Character n-gram F-score)** metric via `sacrebleu`.

### ⚡ Pipeline Architecture

* **Dynamic Data Ingestion:** Safely streams real-world reference sentences (news, weather, general web text) directly into the test suite.
* **Asynchronous Inference:** Queries the live Hugging Face container API, verifying latency, load-handling, and Server-Sent Event stream stability under automated loads.
* **Automated NLP Scoring:** Analyzes character-level overlaps and structural semantic retention to generate a final out-of-100 mathematical metric.

### 📊 Benchmark Results

| Evaluation Metric | Target Language | Dataset Corpus | Final Score |
| --- | --- | --- | --- |
| **chrF Score** | Malayalam (`mal_Mlym`) | OPUS-100 (Real-World) | **85.31 / 100** |

> **Analyst Note:** In modern Machine Translation research, a chrF score above 60 indicates high-quality, fluent translation. A score of **85.31** demonstrates near-human parity. The model successfully retains root semantic meaning even when applying flexible Indic syntax or native transliteration (e.g., dynamically translating "gold medal" contextually rather than strictly literally).

### 🎓 Project Context

BhashaDocs and the BhashaBench testing suite were engineered from the ground up as a complete, full-stack machine learning portfolio piece for a final-year Bachelor of Computer Applications (BCA) capstone at Sahrdaya College of Advanced Studies. It demonstrates the complete AI lifecycle: from responsive front-end design to asynchronous cloud inference and rigorous mathematical validation.

---

## 📦 Setup & Usage

### Prerequisites
- Python 3.8+
- `sacrebleu` (for chrF scoring)
- `requests` (for API calls)
- Access to the live BhashaDocs API

### Files

| File | Purpose |
| --- | --- |
| `ingest.py` | Loads and prepares reference translations from OPUS-100 corpus |
| `evaluate.py` | Single-request evaluation against the BhashaDocs API |
| `evaluate_batch.py` | Batch evaluation across multiple samples with result aggregation |
| `data/indicqa_ml.csv` | Reference dataset (English-Malayalam pairs) |
| `results/` | Output directory for evaluation results and scores |

### Running Evaluations

**Single Evaluation:**
```bash
python evaluate.py
```

**Batch Evaluation:**
```bash
python evaluate_batch.py
```

Results are logged to `results/` with detailed chrF score breakdowns and latency metrics.
