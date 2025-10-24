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
from collections import defaultdict
from deep_translator import GoogleTranslator

# ------------------ Setup ------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
nltk.download("vader_lexicon", quiet=True)

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
downloader = YoutubeCommentDownloader()

# Load FastText model once
FT_MODEL_PATH = "lid.176.ftz"
if not os.path.exists(FT_MODEL_PATH):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        FT_MODEL_PATH,
    )
FT_MODEL = fasttext.load_model(FT_MODEL_PATH)

app = Flask(__name__)

# ------------------ Helpers ------------------
def translate_text(text):
    """Translate text to English safely."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

def clean_text_simple(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def detect_language(text):
    try:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return "__label__en"
        text = text.strip().replace("\n", " ")
        if len(text) > 5000:
            text = text[:5000]
        result = FT_MODEL.predict(text)
        return result[0][0]
    except Exception:
        return "__label__en"

def translate_comments_generator(text_series, translate_limit=None):
    """
    Generator to translate comments with progress updates.
    Yields dicts: {"progress", "eta", "stage"}
    Returns list of translated texts via StopIteration.value
    """
    start = time.time()
    n_total = len(text_series)
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
                translated.append(translate_text(t_clean))

        # emit periodic progress updates (every 30 items or last item)
        if (i + 1) % 30 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            avg = elapsed / (i + 1) if (i + 1) > 0 else 0
            eta = int((n - (i + 1)) * avg + 5)
            progress = 50 + min(int(((i + 1) / n_total) * 30), 30)  # map translate phase to 50-80%
            yield {"progress": progress, "eta": eta, "stage": "translating"}

    # fill remaining texts if translate_limit < total
    if n < n_total:
        for j in range(n, n_total):
            translated.append(clean_text_simple(text_series.iloc[j]))

    return translated

def ensure_text_column(df):
    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].fillna("").astype(str)

def vader_sentiment(df):
    ensure_text_column(df)
    def to_label(text):
        if not text.strip():
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
        if not text.strip():
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
    sentnames = defaultdict(lambda: defaultdict(int))
    ensure_text_column(df)
    for _, row in df.iterrows():
        text = row["text"]
        if not text:
            continue
        try:
            doc = nlp(text)
        except Exception:
            continue
        for ent in doc.ents:
            if ent.label_ == "PERSON" or ent.label_ == "ORG":
                sentnames[row["Final_Sentiment"]][ent.text] += 1
    return {cat: sorted(names.items(), key=lambda kv: kv[1], reverse=True)[:10] for cat, names in sentnames.items()}

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
        # Step 1: download comments
        comments = []
        count = 0
        start = time.time()
        MAX_COMMENTS = 4000

        for comment in downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR):
            comments.append(comment)
            count += 1
            if count % 30 == 0:
                elapsed = time.time() - start
                avg_time = elapsed / count if count > 0 else 0
                eta = max(0, int(((MAX_COMMENTS - count) * avg_time) + 100))
                progress = min(int((count / MAX_COMMENTS) * 50), 50)
                yield f"data: {json.dumps({'progress': progress, 'eta': eta, 'stage': 'downloading'})}\n\n"
            if count >= MAX_COMMENTS:
                break

        df = pd.DataFrame(comments)
        if df.empty:
            yield f"data: {json.dumps({'progress': 100, 'eta': 0, 'stage': 'done', 'result': {'error': 'No comments found'}})}\n\n"
            return

        # Step 2: translation phase
        translate_gen = translate_comments_generator(df["text"])
        while True:
            try:
                upd = next(translate_gen)
                yield f"data: {json.dumps(upd)}\n\n"
            except StopIteration as e:
                translated_list = e.value
                break

        df = df.copy()
        df["text"] = translated_list

        # Step 3: sentiment analysis
        yield f"data: {json.dumps({'progress': 80, 'eta': 10, 'stage': 'sentiment_analysis'})}\n\n"
        df = vader_sentiment(df)
        df = textblob_sentiment(df)
        df = final_sentiment(df)

        # Step 4: entity extraction
        yield f"data: {json.dumps({'progress': 90, 'eta': 5, 'stage': 'entity_extraction'})}\n\n"
        entities_data = extract_entities(df)

        # Step 5: final result
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
