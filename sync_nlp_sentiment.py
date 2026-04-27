"""
sync_nlp_sentiment.py
─────────────────────
Reads reviews_with_nlp_features.xlsx and pushes the NLP-computed fields
(sentiment_label, sentiment_score, emotion_label, emotion_score, sentiment_value)
into the matching review subdocuments inside MongoDB.

Matching strategy:
  hotel_id  (int)  AND  user_id  (str)

After updating all reviews for a hotel, recalculates:
  - avg_sentiment_score  (mean of sentiment_score across all reviews)
  - positive_pct         (% of positive-labelled reviews)

Run:
  python sync_nlp_sentiment.py
"""

import asyncio
import os
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
import motor.motor_asyncio

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME",   "hotel_recommendation")
XLSX_PATH = os.path.join(os.path.dirname(__file__), "reviews_with_nlp_features.xlsx")

# ── Load Excel ───────────────────────────────────────────────────────────────
print(f"📂  Loading {XLSX_PATH} …")
df = pd.read_excel(XLSX_PATH)
print(f"    {len(df):,} rows loaded.  Columns: {list(df.columns)}")

# Group by hotel_id → { user_id: row_dict }
nlp_by_hotel: dict[int, dict[str, dict]] = defaultdict(dict)
for _, row in df.iterrows():
    hid = int(row["hotel_id"])
    uid = str(row["user_id"])
    nlp_by_hotel[hid][uid] = {
        "sentiment_label":  str(row.get("sentiment_label", "")).lower(),
        "sentiment_score":  float(row.get("sentiment_score", 0.0)),
        "emotion_label":    str(row.get("emotion_label",    "")),
        "emotion_score":    float(row.get("emotion_score",   0.0)),
        "sentiment_value":  int(row.get("sentiment_value",  0)),
        "language":         str(row.get("language",         "")).capitalize(),
    }

print(f"    {len(nlp_by_hotel)} unique hotel_ids in spreadsheet.")


# ── Compute Global Aggregates from Excel ───────────────────────────────────
print(f"📊  Computing global aggregate stats from Excel …")
hotel_stats = {}
hotel_groups = df.groupby("hotel_id")

for hid, group in hotel_groups:
    total = len(group)
    pos = sum(1 for label in group["sentiment_label"] if str(label).lower() == "positive")
    neg = sum(1 for label in group["sentiment_label"] if str(label).lower() == "negative")
    neu = total - pos - neg
    
    pos_pct = round(pos / total * 100) if total > 0 else 0
    neg_pct = round(neg / total * 100) if total > 0 else 0
    neu_pct = 100 - pos_pct - neg_pct
    
    avg_sent = group["sentiment_score"].mean() if "sentiment_score" in group.columns else 0.0
    
    hotel_stats[int(hid)] = {
        "total_reviews":     total,
        "positive_reviews":    pos,
        "negative_reviews":    neg,
        "neutral_reviews":     neu,
        "positive_pct":        pos_pct,
        "negative_pct":        neg_pct,
        "neutral_pct":         neu_pct,
        "avg_sentiment_score": float(avg_sent) if not pd.isna(avg_sent) else 0.0
    }

print(f"    Computed stats for {len(hotel_stats)} hotels.")


# ── Async sync ───────────────────────────────────────────────────────────────
async def main():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    db     = client[DB_NAME]

    hotels_updated   = 0
    reviews_updated  = 0
    
    # Update all hotels found in MongoDB that exist in our spreadsheet stats
    async for hotel in db.hotels.find({}):
        hid = int(hotel.get("hotel_id", -1))
        if hid not in hotel_stats:
            continue
            
        stats = hotel_stats[hid]
        uid_map = nlp_by_hotel.get(hid, {})
        reviews = hotel.get("reviews", [])
        modified_reviews = False

        # 1. Update individual reviews for consistency where they exist in DB
        for rev in reviews:
            uid = str(rev.get("user_id", ""))
            if uid in uid_map:
                nlp = uid_map[uid]
                rev["sentiment_label"]  = nlp["sentiment_label"]
                rev["sentiment_score"]  = nlp["sentiment_score"]
                rev["emotion_label"]    = nlp["emotion_label"]
                rev["emotion_score"]    = nlp["emotion_score"]
                rev["sentiment_value"]  = nlp["sentiment_value"]
                rev["language"]         = nlp["language"]
                reviews_updated += 1
                modified_reviews = True

        # 2. Update hotel document with "Global" aggregates (Ground Truth from Excel)
        update_data = {
            "total_reviews":       stats["total_reviews"],
            "positive_reviews":    stats["positive_reviews"],
            "negative_reviews":    stats["negative_reviews"],
            "neutral_reviews":     stats["neutral_reviews"],
            "positive_pct":        stats["positive_pct"],
            "negative_pct":        stats["negative_pct"],
            "neutral_pct":         stats["neutral_pct"],
            "avg_sentiment_score": stats["avg_sentiment_score"],
        }
        
        if modified_reviews:
            update_data["reviews"] = reviews

        await db.hotels.update_one(
            {"_id": hotel["_id"]},
            {"$set": update_data}
        )
        hotels_updated += 1
        
        if hotels_updated % 20 == 0:
            print(f"    Processed {hotels_updated} hotels...")

    print()
    print(f"═══════════════════════════════════════")
    print(f"  Hotels updated  : {hotels_updated}")
    print(f"  Reviews updated : {reviews_updated} (matching texts)")
    print(f"  Aggregates sync : OK (using full Excel ground truth)")
    print(f"═══════════════════════════════════════")

    client.close()


asyncio.run(main())
