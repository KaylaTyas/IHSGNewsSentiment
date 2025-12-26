import os
import time
import json
import re
from time import sleep
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Config
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 5
USE_HF = os.getenv("USE_HF", "True").lower() in ("1", "true", "yes")
HF_MODEL = os.getenv("HF_MODEL", "taufiqdp/indonesian-sentiment")
DB_URL = os.getenv("DB_URL")

if not DB_URL:
    raise Exception("Set DB_URL env var first")

engine = create_engine(DB_URL, pool_size=5, max_overflow=0, pool_pre_ping=True)

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

try:
    stop_words = set(stopwords.words('indonesian'))
except:
    stop_words = set(["dan","yang","di","ke","dari","ini","untuk","adalah","dengan","pada","itu","saat","atau","akan","telah","oleh","dalam","sebagai","juga","bahwa","tersebut"])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def case_folding(text):
    return text.lower()

def tokenize(text):
    try:
        return word_tokenize(text)
    except:
        return re.findall(r"\w+", text)

def remove_stopwords(tokens):
    return [w for w in tokens if w not in stop_words and len(w) > 1]

def stemming_tokens(tokens):
    return [stemmer.stem(w) for w in tokens]

def clean_text(text):
    if text is None or str(text).strip() == "":
        return "", []
    t = case_folding(str(text))
    toks = tokenize(t)
    toks = remove_stopwords(toks)
    toks = stemming_tokens(toks)
    return " ".join(toks), toks

# Sentiment model
sentiment_pipe = None
if USE_HF:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
        sentiment_pipe = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=-1,
            return_all_scores=False
        )
        print(f"Model loaded: {HF_MODEL}")
    except Exception as e:
        print(f"Model load failed, using keywords only: {e}")

# Keyword patterns
positif_keywords = [
    "menguat", "rebound", "bullish", "surplus", "naik", "melonjak",
    "penguatan", "stabil", "optimis", "pulih", "rekor", "melambung"
]
negatif_keywords = [
    "melemah", "anjlok", "bearish", "defisit", "turun", "merosot",
    "penurunan", "tertekan", "resesi", "krisis", "minus", "collapse", "pelemahan"
]

positif_pattern = re.compile(r"\b(" + "|".join(positif_keywords) + r")\b", re.IGNORECASE)
negatif_pattern = re.compile(r"\b(" + "|".join(negatif_keywords) + r")\b", re.IGNORECASE)

def apply_keyword_override(text, model_label, model_score):
    pos_match = bool(positif_pattern.search(text))
    neg_match = bool(negatif_pattern.search(text))

    if pos_match and not neg_match:
        return "positif", 1.0
    elif neg_match and not pos_match:
        return "negatif", 1.0
    elif pos_match and neg_match:
        pos_count = len(positif_pattern.findall(text))
        neg_count = len(negatif_pattern.findall(text))
        if pos_count > neg_count:
            return "positif", 1.0
        elif neg_count > pos_count:
            return "negatif", 1.0
    return model_label, model_score

def predict_sentiment(text):
    if not text or str(text).strip() == "":
        return "netral", 0.5
    
    text = str(text)
    if sentiment_pipe is not None:
        try:
            result = sentiment_pipe(text, truncation=True, max_length=512)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            model_label_raw = result.get("label", "").lower()
            model_score = float(result.get("score", 0.5))
            
            if "neg" in model_label_raw:
                model_label = "negatif"
            elif "pos" in model_label_raw:
                model_label = "positif"
            else:
                model_label = "netral"
            
            return apply_keyword_override(text, model_label, model_score)
        except:
            pass
    
    pos = bool(positif_pattern.search(text))
    neg = bool(negatif_pattern.search(text))
    
    if pos and not neg:
        return "positif", 1.0
    elif neg and not pos:
        return "negatif", 1.0
    return "netral", 0.5

def get_progress_stats(conn):
    q = text("""
        SELECT 
            (SELECT COUNT(*) FROM raw_news) as total_raw,
            (SELECT COUNT(*) FROM processed_news) as total_processed,
            (SELECT COUNT(*) FROM raw_news r 
             LEFT JOIN processed_news p ON r.id = p.raw_id 
             WHERE p.raw_id IS NULL) as remaining
    """)
    result = conn.execute(q).fetchone()
    return {
        "total_raw": result[0],
        "total_processed": result[1],
        "remaining": result[2]
    }

def fetch_unprocessed_batch(conn, limit=BATCH_SIZE):
    q = text("""
        SELECT r.id, r.tanggal, r.kategori, r.judul, r.konten
        FROM raw_news r
        LEFT JOIN processed_news p ON r.id = p.raw_id
        WHERE p.raw_id IS NULL
        ORDER BY r.scraped_at ASC
        LIMIT :limit
    """)
    return pd.read_sql(q, conn, params={"limit": limit})

def insert_batch(conn, rows):
    insert_sql = text("""
        INSERT INTO processed_news 
            (raw_id, konten_clean, tokens, sentiment_label, sentiment_score, processed_at)
        VALUES 
            (:raw_id, :konten_clean, :tokens, :sentiment_label, :sentiment_score, now())
    """)
    
    for r in rows:
        tok_val = r.get("tokens", [])
        if isinstance(tok_val, list):
            tok_val_use = tok_val[:150]
        else:
            tok_val_use = []
        
        params = {
            "raw_id": int(r["raw_id"]),
            "konten_clean": r["konten_clean"][:10000],
            "tokens": json.dumps(tok_val_use, ensure_ascii=False),
            "sentiment_label": r["sentiment_label"],
            "sentiment_score": float(r["sentiment_score"])
        }
        conn.execute(insert_sql, params)

# main
def main_loop():
    total_processed = 0
    retry_count = 0
    start_time = time.time()
    print(f"Starting batch processing (batch_size={BATCH_SIZE}, use_hf={USE_HF})")
    
    try:
        with engine.connect() as conn:
            stats = get_progress_stats(conn)
            print(f"Status: {stats['total_processed']:,}/{stats['total_raw']:,} processed, {stats['remaining']:,} remaining")
            if stats['remaining'] == 0:
                print("All rows already processed.")
                return
    except Exception as e:
        print(f"Could not fetch stats: {e}")
    
    batch_num = 0
    while True:
        try:
            with engine.begin() as conn:
                df = fetch_unprocessed_batch(conn, limit=BATCH_SIZE)
                
                if df.empty:
                    print("Processing complete.")
                    break

                batch_num += 1
                rows_to_insert = []
                
                for idx, row in df.iterrows():
                    raw_id = int(row["id"])
                    text_field = row.get("konten") if pd.notna(row.get("konten")) else row.get("judul", "")
                    cleaned_text, tokens = clean_text(text_field)
                    sentiment_label, sentiment_score = predict_sentiment(cleaned_text)
                    
                    rows_to_insert.append({
                        "raw_id": raw_id,
                        "konten_clean": cleaned_text,
                        "tokens": tokens,
                        "sentiment_label": sentiment_label,
                        "sentiment_score": sentiment_score
                    })

                insert_batch(conn, rows_to_insert)
                total_processed += len(rows_to_insert)
                retry_count = 0
                
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                
                # Simple progress update every 2 batch
                if batch_num % 2 == 0:
                    try:
                        stats = get_progress_stats(conn)
                        pct = stats['total_processed']/max(stats['total_raw'], 1)*100
                        print(f"Batch {batch_num}: {stats['total_processed']:,}/{stats['total_raw']:,} ({pct:.1f}%) | {rate:.1f} rows/sec")
                    except:
                        print(f"Batch {batch_num}: {total_processed:,} rows processed | {rate:.1f} rows/sec")
                sleep(0.5)
                
        except OperationalError as e:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                print(f"Max retries exceeded: {e}")
                raise
            backoff = RETRY_BACKOFF_SEC * (2 ** (retry_count - 1))
            print(f"Connection error, retrying in {backoff}s... (attempt {retry_count}/{MAX_RETRIES})")
            sleep(backoff)
            continue
            
        except SQLAlchemyError as e:
            print(f"Database error: {e}")
            raise
            
        except KeyboardInterrupt:
            print(f"\nInterrupted. Processed {total_processed} rows this session.")
            break
            
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\nDone. Processed {total_processed:,} rows in {elapsed/60:.1f} minutes ({total_processed/elapsed:.1f} rows/sec)")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise