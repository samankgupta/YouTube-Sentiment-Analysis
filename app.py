from flask import Flask, request, render_template, Response
import json
import time
import os
import re
import pandas as pd
import spacy
import nltk
import fasttext
import urllib.request
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import defaultdict, Counter
from deep_translator import GoogleTranslator

# ------------------ Minimal NLTK setup (only what's necessary) ------------------
# Downloading on startup is fine but in production you may want to do this in build step
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# ------------------ Setup models ------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use the small spaCy model (lighter & safer on limited hosts)
nlp = spacy.load("en_core_web_sm", disable=[])  # keep default pipeline
sia = SentimentIntensityAnalyzer()
downloader = YoutubeCommentDownloader()

# Load FastText model once at startup (downloads if missing)
FT_MODEL_PATH = "lid.176.ftz"
if not os.path.exists(FT_MODEL_PATH):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        FT_MODEL_PATH,
    )
FT_MODEL = fasttext.load_model(FT_MODEL_PATH)

app = Flask(__name__)

# ------------------ Helpers ------------------
def clean_text_simple(text):
    """Normalize whitespace and coerce to str."""
    return re.sub(r"\s+", " ", str(text)).strip()

def detect_language(text):
    """Safely detect language using FT_MODEL. Returns '__label__xx' or '__label__en' on error."""
    try:
        if not text or not isinstance(text, str) or not text.strip():
            return "__label__en"
        s = text.strip().replace("\n", " ")
        if len(s) > 5000:
            s = s[:5000]
        res = FT_MODEL.predict(s)
        # fasttext.predict typically returns (['__label__xx'], array([prob]))
        if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], (list, tuple)) and len(res[0]) >= 1:
            return res[0][0]
        # sometimes returns list-of-lists, try to be defensive:
        if isinstance(res, list) and len(res) and isinstance(res[0], (list, tuple)):
            return res[0][0]
        return "__label__en"
    except Exception as e:
        # Log quickly — avoid crashing
        print("detect_language error:", e)
        return "__label__en"

def translate_text(text):
    """Translate text to English using deep-translator; fallback to original on error."""
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        print("translate_text error:", e)
        return text

def translate_comments_generator(text_series, translate_limit=None):
    """
    Generator that translates comment texts in text_series.
    Yields dictionary progress updates (JSON-serializable).
    Returns list of translated strings via StopIteration.value.
    """
    start = time.time()
    n_total = len(text_series)
    if n_total == 0:
        return []

    n = n_total if translate_limit is None else min(n_total, translate_limit)
    translated = []

    for i in range(n):
        raw = text_series.iloc[i]
        t_clean = clean_text_simple(raw)
        if not t_clean:
            translated.append("")
        else:
            lang_label = detect_language(t_clean).replace("__label__", "")
            if lang_label == "en":
                translated.append(t_clean)
            else:
                # sync translation (deep-translator)
                translated.append(clean_text_simple(translate_text(t_clean)))

        # emit progress updates every 30 items or at the end
        if (i + 1) % 30 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            avg = elapsed / (i + 1) if (i + 1) > 0 else 0
            eta = int((n - (i + 1)) * avg + 5)
            # map translate phase to 50..80
            progress = 50 + min(int(((i + 1) / n_total) * 30), 30)
            yield {"progress": progress, "eta": eta, "stage": "translating"}

    # If translation limited, append cleaned originals for remainder
    if n < n_total:
        for j in range(n, n_total):
            translated.append(clean_text_simple(text_series.iloc[j]))

    return translated

def ensure_text_column(df):
    """Ensure df has a text column of strings."""
    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].fillna("").astype(str)

def vader_sentiment(df):
    ensure_text_column(df)
    def to_label(text):
        if not text or not text.strip():
            return "Neutral"
        try:
            score = sia.polarity_scores(text)["compound"]
        except Exception:
            return "Neutral"
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    df["VADER_Sentiment"] = df["text"].apply(to_label)
    return df

def textblob_sentiment(df):
    ensure_text_column(df)
    def to_label(text):
        if not text or not text.strip():
            return "Neutral"
        try:
            p = TextBlob(text).sentiment.polarity
        except Exception:
            return "Neutral"
        if p > 0:
            return "Positive"
        elif p < 0:
            return "Negative"
        else:
            return "Neutral"
    df["TextBlob_Analysis"] = df["text"].apply(to_label)
    return df

def final_sentiment(df):
    df["Final_Sentiment"] = [
        tb if vd == "Neutral" and tb != vd else vd
        for vd, tb in zip(df["VADER_Sentiment"], df["TextBlob_Analysis"])
    ]
    return df

def extract_entities(df):
    """
    Efficient & safe entity extraction:
    - uses nlp.pipe for batching
    - counts PERSON and ORG mentions grouped by Final_Sentiment
    """
    ensure_text_column(df)
    # prepare counters
    counters = defaultdict(Counter)  # counters[sentiment][entity] = count

    texts = df["text"].tolist()
    sentiments = df["Final_Sentiment"].tolist() if "Final_Sentiment" in df.columns else ["Neutral"] * len(texts)

    # Use nlp.pipe to process in batches; catch exceptions per doc
    try:
        for doc, sentiment in zip(nlp.pipe(texts, batch_size=32), sentiments):
            if doc is None:
                continue
            try:
                for ent in doc.ents:
                    if ent.label_ in ("PERSON", "ORG"):
                        # normalize entity text a bit
                        name = ent.text.strip()
                        if name:
                            counters[sentiment][name] += 1
            except Exception as e:
                # continue on per-doc parse errors
                print("entity parse error:", e)
                continue
    except Exception as e:
        # if nlp.pipe raises globally, fall back to per-doc safe loop
        print("nlp.pipe error:", e)
        for text, sentiment in zip(texts, sentiments):
            if not text:
                continue
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ("PERSON", "ORG"):
                        name = ent.text.strip()
                        if name:
                            counters[sentiment][name] += 1
            except Exception as e2:
                print("fallback entity error:", e2)
                continue

    # Convert counters to sorted list of tuples (top 10)
    result = {
        sentiment: sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for sentiment, counter in counters.items()
    }
    return result

# ------------------ Flask routes ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/progress", methods=["POST"])
def progress():
    video_url = request.json.get("video_url", "")
    if not video_url:
        return Response(json.dumps({"error": "no video_url"}), status=400, mimetype="application/json")

    def generate():
        # Step 1: download comments (streaming progress)
        comments = []
        count = 0
        start = time.time()
        MAX_COMMENTS = 4000  # adjust if you want fewer

        for comment in downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR):
            comments.append(comment)
            count += 1
            if count % 30 == 0:
                elapsed = time.time() - start
                avg_time = elapsed / count if count > 0 else 0
                eta = max(0, int(((MAX_COMMENTS - count) * avg_time) + 10))
                progress = min(int((count / MAX_COMMENTS) * 50), 50)  # map to 0-50
                yield f"data: {json.dumps({'progress': progress, 'eta': eta, 'stage': 'downloading'})}\n\n"
            if count >= MAX_COMMENTS:
                break

        df = pd.DataFrame(comments)
        if df.empty:
            yield f"data: {json.dumps({'progress': 100, 'eta': 0, 'stage': 'done', 'result': {'error': 'No comments found'}})}\n\n"
            return

        # Step 2: translation phase (stream progress)
        translate_gen = translate_comments_generator(df["text"])
        while True:
            try:
                upd = next(translate_gen)
                # upd is a JSON-serializable dict
                yield f"data: {json.dumps(upd)}\n\n"
            except StopIteration as e:
                translated_list = e.value
                break

        # replace texts with translations
        df = df.copy()
        df["text"] = translated_list

        # Step 3: sentiment analysis
        yield f"data: {json.dumps({'progress': 80, 'eta': 10, 'stage': 'sentiment_analysis'})}\n\n"
        df = vader_sentiment(df)
        df = textblob_sentiment(df)
        df = final_sentiment(df)

        # Step 4: entity extraction (progress update)
        yield f"data: {json.dumps({'progress': 90, 'eta': 5, 'stage': 'entity_extraction'})}\n\n"
        entities_data = extract_entities(df)

        # Step 5: prepare final result
        sentiment_counts = df["Final_Sentiment"].value_counts().to_dict()
        total_comments = len(df)
        top_positive = df[df["Final_Sentiment"] == "Positive"]["text"].head(20).tolist()
        top_negative = df[df["Final_Sentiment"] == "Negative"]["text"].head(20).tolist()

        result = {
            "sentiment_counts": sentiment_counts,
            "total_comments": total_comments,
            "top_positive_comments": top_positive,
            "top_negative_comments": top_negative,
            "top_entities": entities_data,
        }

        yield f"data: {json.dumps({'progress': 100, 'eta': 0, 'stage': 'done', 'result': result})}\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
