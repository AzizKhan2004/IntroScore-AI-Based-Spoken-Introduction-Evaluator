# Nirmaan AI – Self-Introduction Scoring Tool

This project is a prototype AI tool that evaluates a student's spoken **self-introduction** based on a rubric.  
The original task is from the **Nirmaan AI Intern Case Study** for the Communication Program. :contentReference[oaicite:2]{index=2}

The tool:

- Accepts a **transcript text** as input.
- Computes **per-criterion scores** using a rubric (content, structure, clarity, engagement).
- Combines **rule-based checks**, **NLP semantic similarity**, and **rubric-based weighting**.
- Produces:
  - An **overall score** (0–100)
  - **Per-criterion details** (keyword coverage, semantic similarity, filler word rate, sentiment, etc.)
- Displays results in a simple **web UI**.

---

## 1. Architecture Overview

**High-level components:**

1. **Backend (FastAPI)**
   - Endpoint: `POST /score`
   - Input: `{ "transcript": "<string>" }`
   - Output (JSON):
     - `overall_score` (0–100)
     - `overall_score_0_to_1`
     - `word_count`, `sentence_count`
     - `criteria`: array of objects with:
       - `id`, `category`, `name`
       - `score_0_to_1`, `score_weighted`, `weight`
       - Extra signals: `keywords_found`, `semantic_similarity`,
         `filler_rate`, `sentiment`, `flow steps`, etc.

2. **Scoring Engine**
   - Implemented in `backend/scoring.py`
   - Uses:
     - **Rule-based** logic:
       - Keyword presence
       - Word count / length range
       - Filler word rate
       - Basic flow heuristic (greeting → self → details → closing)
     - **NLP-based semantic similarity**:
       - Uses `sentence-transformers/all-MiniLM-L6-v2`
       - Compares transcript with rubric criterion descriptions
     - **Data/rubric-driven weighting**:
       - Each criterion has a weight (from rubric)
       - Final score is a weighted sum normalized to 0–100

3. **Rubric Configuration**
   - Implemented in `backend/rubric_config.py`
   - Encodes each criterion with:
     - `id`, `category`, `name`, `description`
     - `type` (keyword_presence, semantic, flow, length, filler_rate, sentiment)
     - `weight`
     - Optional: `keywords`, `min_words`, `max_words`, `extra` fields
   - Filler rate thresholds and sentiment bands are modeled based on the Excel rubric.

4. **Frontend**
   - Simple static page (`frontend/index.html`)
   - Lets user **paste transcript**, click **Score**, and see:
     - Overall score
     - Per-criterion table
     - Raw JSON output (for debugging)

---

## 2. Scoring Formula & Logic

For each rubric criterion:

1. **Keyword Presence** (e.g., salutation, basic details)
   - Compute fraction of rubric keywords that appear in transcript:
     - `rule_score = (#keywords_found / #keywords_total)`
   - Compute semantic similarity between transcript and criterion description:
     - `semantic_similarity = cosine_similarity(embedding(transcript), embedding(description))`
     - Mapped from [-1, 1] → [0, 1]
   - Combined:
     ```python
     combined_score = 0.5 * rule_score + 0.5 * semantic_similarity
     ```

2. **Semantic Alignment (overall)**
   - Only uses semantic similarity with a generic description of a good self-introduction.
   - Score in [0,1].

3. **Flow**
   - Look for positions of:
     - Greeting
     - Self introduction
     - Optional details (family, hobbies, goals)
     - Closing
   - Score based on whether indices appear in correct order.

4. **Length**
   - Ideal range: 80–180 words (derived from rubric guidance for short self-intros).
   - If inside range → score = 1.
   - If too short or too long → score decreases linearly.

5. **Filler Word Rate**
   - Count filler words: `"um", "uh", "like", "you know", "basically", "actually", ...`
   - Compute:
     ```text
     filler_rate = (filler_word_count / total_words) * 100
     ```
   - Map filler_rate to band (from rubric):
     - 0–3% → factor = 1.0
     - 4–6% → 0.8
     - 7–9% → 0.6
     - 10–12% → 0.4
     - 13%+ → 0.2

6. **Sentiment / Engagement**
   - Use VADER sentiment (`vaderSentiment`).
   - Take `compound` score (−1 to 1).
   - Map to bands:
     - ≥ 0.9 → 1.0
     - 0.7–0.89 → 0.8
     - 0.5–0.69 → 0.6
     - 0.3–0.49 → 0.4
     - < 0.3 → 0.2

7. **Weighted Combination**
   - Each criterion has a `weight` (from rubric).
   - For criterion `i` with score `s_i` in [0,1] and weight `w_i`:
     ```python
     weighted_sum = Σ (s_i * w_i)
     overall_score_0_to_1 = weighted_sum / Σ w_i
     overall_score = overall_score_0_to_1 * 100
     ```

---

## 3. Running Locally (Short Version)

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
