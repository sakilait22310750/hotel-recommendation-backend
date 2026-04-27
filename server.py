from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import bcrypt
import jwt
import pickle
import numpy as np


def _load_pickle_compat(path):
    """Load a pickle file, handling NumPy 1.x vs 2.x module path differences."""
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                # Remap numpy._core <-> numpy.core for cross-version pickle loading
                class CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith('numpy._core.'):
                            # Pickle from NumPy 2.x, loading on NumPy 1.x
                            new_module = 'numpy.core.' + module[len('numpy._core.'):]
                            mod = __import__(new_module, fromlist=[name])
                            return getattr(mod, name)
                        if module.startswith('numpy.core.'):
                            # Pickle from NumPy 1.x, loading on NumPy 2.x
                            new_module = 'numpy._core.' + module[len('numpy.core.'):]
                            mod = __import__(new_module, fromlist=[name])
                            return getattr(mod, name)
                        return super().find_class(module, name)
                f.seek(0)
                return CompatUnpickler(f).load()
            raise
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import json
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Optional Gemini AI import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. AI Insights endpoint will be disabled.")
    print("Install with: pip install google-generativeai")

# Optional Google Drive API imports (only needed for image serving)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    from google.auth.exceptions import GoogleAuthError
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    # Logger not available yet, will print warning after logger is initialized
    print("Warning: Google Drive API packages not installed. Image serving will be disabled.")
    print("Install with: pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib")

from dotenv import load_dotenv

load_dotenv(override=True)

mongo_url = os.environ["MONGO_URL"]
db_name = os.environ["DB_NAME"]


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env', override=True)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Security
security = HTTPBearer()

# Load ML models (with NumPy 1.x/2.x pickle compatibility)
user_similarity = _load_pickle_compat(ROOT_DIR / 'user_similarity.pkl')
user_hotel_matrix = _load_pickle_compat(ROOT_DIR / 'user_hotel_matrix.pkl')
train_df = _load_pickle_compat(ROOT_DIR / 'train_df.pkl')

# Create the main app
app = FastAPI(title="Accommodation Recommendation API")
api_router = APIRouter(prefix="/api")

# ============== MODELS ==============

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    country: Optional[str] = None
    age_group: Optional[str] = None
    travel_frequency: Optional[str] = None
    preferences: Optional[List[str]] = []

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    id: str
    email: str
    name: str
    country: Optional[str]
    age_group: Optional[str]
    travel_frequency: Optional[str]
    preferences: Optional[List[str]]
    created_at: datetime

class Hotel(BaseModel):
    id: str
    hotel_id: int
    name: str
    location: str
    rating: float
    total_reviews: int
    positive_reviews: int
    negative_reviews: int
    neutral_reviews: int
    avg_sentiment_score: float
    vgg_features: Optional[List[float]]
    reviews: List[Dict]

class HotelListItem(BaseModel):
    id: str
    hotel_id: int
    name: str
    location: str
    rating: float
    total_reviews: int
    avg_sentiment_score: float

class RecommendationRequest(BaseModel):
    user_id: str
    location: Optional[str] = None
    limit: int = 1000  # Increased to show all hotels (was 10)
    amenities: Optional[List[str]] = []  # Selected amenities like ["beach_access", "pool", "wifi"]
    min_rating: Optional[float] = None  # Minimum rating filter (e.g., 4.5)
    preferences: Optional[List[str]] = []  # Travel preferences from user profile e.g. ["beach", "luxury"]

class FeedbackRequest(BaseModel):
    user_id: str
    hotel_id: int
    rating: float
    comment: Optional[str] = None

# ============== AUTH UTILITIES ==============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    """Get current user if token is provided and valid, otherwise return None"""
    if not credentials:
        return None
    try:
        token = credentials.credentials
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = await db.users.find_one({"_id": user_id})
        return user
    except HTTPException:
        # Token expired or invalid - return None instead of raising
        return None
    except Exception:
        # Any other error - return None
        return None

# ============== RECOMMENDATION ENGINE ==============

class RecommendationEngine:
    def __init__(self):
        self.user_similarity = user_similarity
        self.user_hotel_matrix = user_hotel_matrix
        self.train_df = train_df
        
        # Load image features for amenity matching
        self.img_df = {}
        for hotel_id in train_df['hotel_id'].unique():
            hotel_data = train_df[train_df['hotel_id'] == hotel_id].iloc[0]
            vgg_cols = [col for col in train_df.columns if col.startswith('vgg_feat_')]
            if vgg_cols:
                self.img_df[int(hotel_id)] = hotel_data[vgg_cols].values
    
    def _check_location_has_amenity(self, location: str, amenity: str) -> bool:
        """
        Check if a location can have a specific amenity based on location name.
        This is a rule-based approach - can be replaced with CNN-based detection later.
        
        Amenity IDs from frontend: 'pool', 'beach', 'spa', 'garden', 'wifi', 
                                   'restaurant', 'parking', 'gym'
        """
        location_lower = location.lower()
        amenity_lower = amenity.lower()
        
        # Beach/sea access locations (coastal areas)
        beach_keywords = ['beach', 'coast', 'sea', 'ocean', 'bay', 'lagoon', 'shore', 
                         'hikkaduwa', 'bentota', 'beruwala', 'negombo', 'trincomalee',
                         'tangalle', 'ahungalla', 'galle', 'kalutara', 'wadduwa',
                         'colombo', 'matara', 'mirissa', 'unawatuna', 'pasikudah']
        
        # Mountain/hill locations (no beach access)
        mountain_keywords = ['kandy', 'kandy district', 'nuwara eliya', 'dambulla', 'hills', 'mountain',
                            'central province', 'hill country', 'habarana', 'anuradhapura',
                            'polonnaruwa', 'sigiriya']
        
        # Check for beach access amenity
        if amenity_lower == 'beach' or 'beach' in amenity_lower:
            # Check if location has beach access
            has_beach_keyword = any(keyword in location_lower for keyword in beach_keywords)
            is_mountain = any(keyword in location_lower for keyword in mountain_keywords)
            
            if is_mountain:
                logger.info(f"Location '{location}' is in mountain area, cannot have beach access")
                return False
            
            if not has_beach_keyword:
                logger.info(f"Location '{location}' doesn't appear to have beach access (no coastal keywords found)")
                return False
            
            return True
        
        # For other amenities (pool, wifi, spa, garden, restaurant, parking, gym), 
        # assume available everywhere (hotels can have these regardless of location)
        return True
    
    def _filter_hotels_by_amenities(self, hotels: list, selected_amenities: list) -> list:
        """
        Filter hotels based on selected amenities using location-based rules.
        This ensures hotels match the requested amenities (e.g., no beach hotels in non-coastal areas).
        """
        if not selected_amenities or len(selected_amenities) == 0:
            return hotels
        
        filtered_hotels = []
        for hotel in hotels:
            location = hotel.get('location', '').lower()
            hotel_id = hotel.get('hotel_id')
            hotel_name = hotel.get('name', '')
            
            # Check all selected amenities - ALL must match
            matches_all_amenities = True
            for amenity in selected_amenities:
                # Check if location supports this amenity
                if not self._check_location_has_amenity(location, amenity):
                    matches_all_amenities = False
                    logger.info(f"Hotel {hotel_id} ({hotel_name}) in {location} doesn't match amenity: {amenity} - location incompatible")
                    break
            
            # Only include hotels that match all selected amenities (location-based check)
            # Note: VGG features are not required for basic filtering, only for image similarity scoring
            if matches_all_amenities:
                filtered_hotels.append(hotel)
            else:
                logger.debug(f"Excluding hotel {hotel_id} ({hotel_name}) - doesn't match amenities: {selected_amenities}")
        
        logger.info(f"Filtered {len(hotels)} hotels to {len(filtered_hotels)} based on amenities: {selected_amenities}")
        if len(filtered_hotels) == 0 and len(selected_amenities) > 0:
            logger.warning(f"No hotels match the selected amenities {selected_amenities} for the given location")
        return filtered_hotels
    
    async def hybrid_recommendation_with_vgg(
        self, user_id: str, hotels: list, selected_amenities: list,
        top_n: int = 1000, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2,
        user_preferences: list = None, db_ref=None
    ):
        """
        Hybrid recommendation:
        alpha * collaborative_filtering + beta * NLP_sentiment + gamma * VGG_image_similarity

        For new users (not in pickle matrix) we use real-time MongoDB feedback to build
        a live CF score, and fall back to preference-aware popularity ranking.
        """

        # --- 1. Amenity filtering (location / CNN-based) ---
        if selected_amenities and len(selected_amenities) > 0:
            hotels = self._filter_hotels_by_amenities(hotels, selected_amenities)

        if not hotels:
            logger.warning(f"No hotels match the selected amenities: {selected_amenities}")
            return []

        hotel_ids = [h['hotel_id'] for h in hotels]
        hotel_map = {h['hotel_id']: h for h in hotels}

        # --- 2. Real-time feedback from MongoDB ---
        # Build a live rating vector for this user from their submitted feedback.
        live_ratings = {}  # {hotel_id: rating_0_to_1}
        live_seen = set()  # hotels the user has already rated (exclude from results)
        if db_ref is not None:
            try:
                cursor = db_ref.feedback.find({"user_id": user_id})
                async for fb in cursor:
                    hid = fb.get('hotel_id')
                    rating = fb.get('rating', 0)
                    if hid is not None:
                        live_ratings[hid] = float(rating) / 5.0  # normalise to [0,1]
                        live_seen.add(hid)
                logger.info(f"User {user_id} has {len(live_ratings)} live feedback ratings")
            except Exception as e:
                logger.warning(f"Could not load live feedback for user {user_id}: {e}")

        # --- 3. Collaborative filtering ---
        in_pickle_matrix = user_id in self.user_similarity.index

        if in_pickle_matrix:
            # --- Classic pickle-based CF ---
            similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:6]
            cf_scores = pd.Series(0.0, index=hotel_ids)

            for similar_user, similarity_score in similar_users.items():
                for hotel_id in hotel_ids:
                    if hotel_id in self.user_hotel_matrix.columns:
                        cf_scores[hotel_id] += similarity_score * self.user_hotel_matrix.loc[similar_user, hotel_id]

            # Exclude hotels the user has already interacted with (pickle matrix)
            seen_from_matrix = set(
                self.user_hotel_matrix.columns[
                    self.user_hotel_matrix.loc[user_id] > 0
                ].tolist()
            )
            exclude_ids = seen_from_matrix | live_seen
            cf_scores = cf_scores[~cf_scores.index.isin(exclude_ids)]

            if cf_scores.empty or cf_scores.max() == 0:
                # Fallback within pickle path
                cf_scores = pd.Series(
                    {h: hotel_map[h]['rating'] * 0.6 + (hotel_map[h]['avg_sentiment_score'] + 1) * 2.0
                     for h in hotel_ids if h not in exclude_ids}
                )

            if not cf_scores.empty and cf_scores.max() > 0:
                cf_scores = cf_scores / cf_scores.max()

        elif live_ratings:
            # --- New user with live feedback: build CF from similar raters ---
            # Find hotels rated by this user and compute cosine similarity with all
            # other users in the matrix who rated those same hotels.
            rated_hids = [h for h in live_ratings if h in self.user_hotel_matrix.columns]
            if rated_hids:
                user_live_vec = pd.Series(0.0, index=self.user_hotel_matrix.columns)
                for hid in rated_hids:
                    user_live_vec[hid] = live_ratings[hid]

                # Cosine similarity against all pickle users
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                live_vec_arr = user_live_vec.values.reshape(1, -1)
                matrix_arr = self.user_hotel_matrix.values
                sims = cos_sim(live_vec_arr, matrix_arr)[0]
                sim_series = pd.Series(sims, index=self.user_hotel_matrix.index)
                top_similar = sim_series.sort_values(ascending=False)[:5]

                cf_scores = pd.Series(0.0, index=hotel_ids)
                for sim_user, sim_score in top_similar.items():
                    for hotel_id in hotel_ids:
                        if hotel_id in self.user_hotel_matrix.columns:
                            cf_scores[hotel_id] += sim_score * self.user_hotel_matrix.loc[sim_user, hotel_id]

                cf_scores = cf_scores[~cf_scores.index.isin(live_seen)]
                if not cf_scores.empty and cf_scores.max() > 0:
                    cf_scores = cf_scores / cf_scores.max()
                logger.info(f"Built live CF scores for new user {user_id} from feedback")
            else:
                cf_scores = pd.Series(0.0, index=[h for h in hotel_ids if h not in live_seen])
        else:
            # --- Pure cold-start: no pickle matrix, no feedback ---
            # Personalise by the user's registered preferences.
            # Preferences that match amenity keywords boost the hotel's score.
            PREF_AMENITY_MAP = {
                'beach': ['beach', 'coast', 'sea', 'ocean', 'bay', 'lagoon'],
                'luxury': [],   # handled via high rating weight
                'wifi': [],
                'spa': ['spa', 'wellness'],
                'pool': ['pool'],
                'gym': ['gym', 'fitness'],
                'restaurant': ['restaurant', 'dining'],
                'garden': ['garden', 'nature'],
            }

            scores = {}
            prefs_lower = [p.lower() for p in (user_preferences or [])]
            for hotel in hotels:
                if hotel['hotel_id'] in live_seen:
                    continue
                sentiment = hotel['avg_sentiment_score']
                if np.isnan(sentiment) or np.isinf(sentiment):
                    sentiment = 0
                base_score = hotel['rating'] * 0.6 + (sentiment + 1) * 2.0

                # Preference boost: +0.5 per matching preference keyword found in hotel name/location
                pref_boost = 0.0
                loc_name = (hotel.get('location', '') + ' ' + hotel.get('name', '')).lower()
                for pref in prefs_lower:
                    keywords = PREF_AMENITY_MAP.get(pref, [pref])
                    if any(kw in loc_name for kw in keywords) if keywords else False:
                        pref_boost += 0.5

                # For 'luxury' preference, boost hotels with rating >= 4.5
                if 'luxury' in prefs_lower and hotel['rating'] >= 4.5:
                    pref_boost += 0.5

                scores[hotel['hotel_id']] = base_score + pref_boost

            sorted_hotels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"Cold-start for user {user_id} with prefs {prefs_lower}: returning {len(sorted_hotels)} hotels")

            # --- Per-user diversification via deterministic shuffle ---
            # Hotels with similar scores (within ±0.3 of each other) are shuffled
            # using a seed derived from the user_id hash so different users see
            # a different ordering of equally-good hotels.
            import hashlib, random as _random
            user_seed = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % (2 ** 31)
            rng = _random.Random(user_seed)

            TIER_BAND = 0.3
            diversified = []
            tier_buf = []
            tier_top = sorted_hotels[0][1] if sorted_hotels else 0

            for hotel_id, score in sorted_hotels:
                if tier_top - score <= TIER_BAND:
                    tier_buf.append((hotel_id, score))
                else:
                    rng.shuffle(tier_buf)
                    diversified.extend(tier_buf)
                    tier_buf = [(hotel_id, score)]
                    tier_top = score
            if tier_buf:
                rng.shuffle(tier_buf)
                diversified.extend(tier_buf)

            if top_n > 0 and top_n < len(diversified) and top_n < 1000:
                return [h[0] for h in diversified[:top_n]]
            return [h[0] for h in diversified]

        # --- 4. NLP sentiment scores ---
        all_scored_ids = cf_scores.index.tolist()
        nlp_scores = pd.Series({h['hotel_id']: h['avg_sentiment_score'] for h in hotels})
        nlp_scores = nlp_scores.reindex(all_scored_ids).fillna(0)

        # --- 5. VGG image similarity scores ---
        if in_pickle_matrix:
            user_hotels_seen = self.train_df[self.train_df['user_id'] == user_id]['hotel_id'].unique()
        else:
            user_hotels_seen = list(live_ratings.keys())
        user_vecs = [self.img_df[h] for h in user_hotels_seen if h in self.img_df]

        img_scores_list = []
        for hotel_id in all_scored_ids:
            if hotel_id in self.img_df and len(user_vecs) > 0:
                hotel_vec = self.img_df[hotel_id].reshape(1, -1)
                user_vecs_array = np.array(user_vecs)
                sim = cosine_similarity(hotel_vec, user_vecs_array).mean()
                img_scores_list.append(sim)
            else:
                img_scores_list.append(0)

        img_scores = pd.Series(img_scores_list, index=all_scored_ids)

        # --- 6. Hybrid final score: alpha*CF + beta*NLP + gamma*Image ---
        final_score = (
            alpha * cf_scores +
            beta * nlp_scores +
            gamma * img_scores
        )

        sorted_hotel_ids = final_score.sort_values(ascending=False).index.tolist()
        if top_n > 0 and top_n < len(sorted_hotel_ids) and top_n < 1000:
            return sorted_hotel_ids[:top_n]
        return sorted_hotel_ids

    def score_hotels_by_category(self, hotels: list, category: str, top_n: int = 12) -> list:
        """
        CNN-powered hotel category ranking.

        Algorithm (3 steps):
        ─────────────────────────────────────────────────────────────────────
        1. SEED IDENTIFICATION (rule-based)
           Pick hotels that are *unambiguously* in this category using hard
           rules — these become the "training anchors" for the CNN.

           beach   → location matches coastal keywords AND rating >= 4.0
           luxury  → rating >= 4.5  OR  derived price >= $200/night
           budget  → derived price <= $40/night AND rating >= 3.5

        2. CNN ARCHETYPE VECTOR
           Average the VGG-4096 feature vectors of all seed hotels.
           This "archetype" vector visually summarises what a beach/luxury/budget
           hotel looks like in the training image corpus.

        3. HYBRID SCORING (CNN + rating + sentiment + price)
           For every hotel in the pool compute:

           beach:
             0.35 × (rating/5) + 0.30 × (norm_sentiment) + 0.35 × CNN_sim
             Hard filter: hotel must be in a coastal location.

           luxury:
             0.40 × (rating/5) + 0.25 × (price/300 capped at 1) + 0.35 × CNN_sim

           budget:
             0.30 × (rating/5) + 0.35 × (norm_sentiment)
             + 0.20 × (1 − price/80 floored at 0) + 0.15 × CNN_sim

        Hotels without VGG vectors (not in pickle training set) degrade
        gracefully — CNN_sim = 0, score falls back to rating/price only.
        ─────────────────────────────────────────────────────────────────────
        """
        BEACH_KW = [
            'beach', 'coast', 'sea', 'ocean', 'bay', 'lagoon', 'shore',
            'hikkaduwa', 'bentota', 'beruwala', 'negombo', 'trincomalee',
            'tangalle', 'ahungalla', 'galle', 'kalutara', 'wadduwa',
            'matara', 'mirissa', 'unawatuna', 'pasikudah',
        ]
        MOUNTAIN_KW = [
            'kandy', 'nuwara eliya', 'dambulla', 'habarana', 'hills',
            'mountain', 'central province', 'hill country', 'anuradhapura',
            'polonnaruwa', 'sigiriya',
        ]

        def _is_coastal(location: str) -> bool:
            loc = location.lower()
            return (
                any(kw in loc for kw in BEACH_KW) and
                not any(kw in loc for kw in MOUNTAIN_KW)
            )

        # ── Step 1: Identify seed hotels ──────────────────────────────────
        seed_vgg_vecs = []
        for h in hotels:
            hid = int(h.get('hotel_id', 0))
            rating = float(h.get('rating', 0))
            location = h.get('location', '')
            price_info = _derive_price_range(rating, h.get('name', ''))
            price_min = price_info['min']

            is_seed = False
            if category == 'beach':
                is_seed = _is_coastal(location) and rating >= 4.0
            elif category == 'luxury':
                is_seed = rating >= 4.5 or price_min >= 200
            elif category == 'budget':
                is_seed = price_min <= 40 and rating >= 3.5

            if is_seed and hid in self.img_df:
                seed_vgg_vecs.append(self.img_df[hid])

        # ── Step 2: Build CNN archetype vector ────────────────────────────
        archetype = np.mean(seed_vgg_vecs, axis=0) if seed_vgg_vecs else None
        logger.info(
            f"[CategoryCNN] category={category} seeds={len(seed_vgg_vecs)} "
            f"archetype={'built' if archetype is not None else 'N/A (no VGG seeds)'}"
        )

        # ── Step 3: Score all hotels ──────────────────────────────────────
        results = []
        for h in hotels:
            hid = int(h.get('hotel_id', 0))
            rating = float(h.get('rating', 0))
            sentiment = float(h.get('avg_sentiment_score', 0))
            location = h.get('location', '')
            price_info = _derive_price_range(rating, h.get('name', ''))
            price_min = price_info['min']

            # Normalised feature signals (all in [0, 1])
            norm_rating    = rating / 5.0
            norm_sentiment = (sentiment + 1.0) / 2.0   # maps [-1,1] → [0,1]

            # CNN similarity to archetype
            cnn_sim = 0.0
            if archetype is not None and hid in self.img_df:
                cnn_sim = float(
                    cosine_similarity(
                        self.img_df[hid].reshape(1, -1),
                        archetype.reshape(1, -1)
                    )[0][0]
                )

            if category == 'beach':
                if not _is_coastal(location):
                    continue   # hard filter — non-coastal hotels excluded
                score = 0.35 * norm_rating + 0.30 * norm_sentiment + 0.35 * cnn_sim

            elif category == 'luxury':
                price_score = min(price_min / 300.0, 1.0)
                score = 0.40 * norm_rating + 0.25 * price_score + 0.35 * cnn_sim

            elif category == 'budget':
                # Favour lower price ↔ price_score near 1 for cheap hotels
                price_score = max(0.0, 1.0 - price_min / 80.0)
                score = (
                    0.30 * norm_rating +
                    0.35 * norm_sentiment +
                    0.20 * price_score +
                    0.15 * cnn_sim
                )
            else:
                score = norm_rating

            results.append((h, score, price_info))

        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"[CategoryCNN] category={category} returned {min(top_n, len(results))} hotels")
        return results[:top_n]


recommendation_engine = RecommendationEngine()

# ============== ROUTES ==============

@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_id = str(uuid.uuid4())
    user = {
        "_id": user_id,
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "name": user_data.name,
        "country": user_data.country,
        "age_group": user_data.age_group,
        "travel_frequency": user_data.travel_frequency,
        "preferences": user_data.preferences or [],
        "created_at": datetime.utcnow(),
        "interaction_history": []
    }
    
    await db.users.insert_one(user)
    
    # Generate token
    token = create_access_token({"sub": user_id, "email": user_data.email})
    
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name
        }
    }

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user['_id'], "email": user['email']})
    
    return {
        "token": token,
        "user": {
            "id": user['_id'],
            "email": user['email'],
            "name": user['name']
        }
    }

@api_router.get("/user/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user['_id'],
        "email": current_user['email'],
        "name": current_user['name'],
        "country": current_user.get('country'),
        "age_group": current_user.get('age_group'),
        "travel_frequency": current_user.get('travel_frequency'),
        "preferences": current_user.get('preferences', []),
        "created_at": current_user['created_at']
    }

@api_router.put("/user/profile")
async def update_profile(profile_data: dict, current_user: dict = Depends(get_current_user)):
    update_fields = {}
    allowed_fields = ['name', 'country', 'age_group', 'travel_frequency', 'preferences']
    
    for field in allowed_fields:
        if field in profile_data:
            update_fields[field] = profile_data[field]
    
    if update_fields:
        await db.users.update_one(
            {"_id": current_user['_id']},
            {"$set": update_fields}
        )
    
    return {"message": "Profile updated successfully"}

@api_router.get("/hotels")
async def get_hotels(location: Optional[str] = None, min_rating: Optional[float] = None, limit: int = 50):
    query = {}
    
    if location:
        query['location'] = {"$regex": location, "$options": "i"}
    
    if min_rating:
        query['rating'] = {"$gte": min_rating}
    
    hotels = await db.hotels.find(query).limit(limit).to_list(limit)
    base_url = os.environ.get("BACKEND_URL", "http://localhost:8000").strip().rstrip('/')
    
    return [
        {
            "id": h['_id'],
            "hotel_id": h['hotel_id'],
            "name": h['name'],
            "location": h['location'],
            "rating": h['rating'],
            "total_reviews": h['total_reviews'],
            "avg_sentiment_score": h['avg_sentiment_score'],
            "image_url": f"{base_url}/api/hotel-images/{h['hotel_id']}/proxy?index=1"
        }
        for h in hotels
    ]



@api_router.get("/hotels/search")
async def search_hotels_suggestions(q: str = "", limit: int = 8):
    """
    Live search suggestions for the search bar.
    Matches hotels by name OR location using a case-insensitive regex.
    Returns lightweight suggestion objects (hotel_id, name, location, rating).
    """
    if not q or len(q.strip()) < 2:
        return []

    query = {
        "$or": [
            {"name":     {"$regex": q.strip(), "$options": "i"}},
            {"location": {"$regex": q.strip(), "$options": "i"}},
        ]
    }
    hotels = await db.hotels.find(query).limit(limit).to_list(limit)
    base_url = os.environ.get("BACKEND_URL", "http://localhost:8000").strip().rstrip('/')
    return [
        {
            "hotel_id":  h["hotel_id"],
            "name":      h["name"],
            "location":  h["location"],
            "rating":    h["rating"],
            "image_url": f"{base_url}/api/hotel-images/{h['hotel_id']}/proxy?index=1",
        }
        for h in hotels
    ]


@api_router.get("/hotels/stats")
async def get_hotel_stats():
    """
    Returns real aggregate stats from the database:
    - total: total number of hotel documents
    - avg_rating: mean rating across all hotels (2 dp)
    """
    pipeline = [
        {
            "$group": {
                "_id":        None,
                "total":      {"$sum":  1},
                "avg_rating": {"$avg":  "$rating"},
            }
        }
    ]
    cursor = db.hotels.aggregate(pipeline)
    result = await cursor.to_list(1)
    if result:
        return {
            "total":      result[0]["total"],
            "avg_rating": round(result[0]["avg_rating"] or 0, 2),
        }
    return {"total": 0, "avg_rating": 0.0}


@api_router.get("/hotel-category")
async def get_hotels_by_category(
    type: str = "beach",
    limit: int = 12,
    location: Optional[str] = None,
):
    """
    Returns hotels ranked by CNN (VGG) + rating/sentiment/price for a given category.
    type: 'beach' | 'luxury' | 'budget'

    The scoring uses the RecommendationEngine.score_hotels_by_category() method which:
    1. Identifies seed hotels unambiguously in the category using rule-based logic
    2. Builds a VGG archetype vector (mean of seed image feature vectors)
    3. Scores all hotels by cosine similarity to the archetype + rating/price signals
    """
    valid_types = {"beach", "luxury", "budget"}
    if type not in valid_types:
        raise HTTPException(status_code=400, detail=f"type must be one of {valid_types}")

    query: dict = {}
    if location:
        query["location"] = {"$regex": location, "$options": "i"}

    hotels = await db.hotels.find(query).to_list(1000)
    if not hotels:
        return []

    ranked = recommendation_engine.score_hotels_by_category(hotels, type, top_n=limit)

    base_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    return [
        {
            "id": h["_id"],
            "hotel_id": h["hotel_id"],
            "name": h["name"],
            "location": h["location"],
            "rating": h["rating"],
            "total_reviews": h["total_reviews"],
            "avg_sentiment_score": h["avg_sentiment_score"],
            "positive_pct": _positive_pct(h),
            "price_info": price_info,
            "image_url": f"{base_url}/api/hotel-images/{h['hotel_id']}/proxy?index=1",
        }
        for h, _score, price_info in ranked   # 3-tuple: (hotel, score, price_info)
    ]

# ---- Hotel enrichment helpers ----

def _derive_amenities(name: str, location: str, reviews: list) -> list:
    """
    Detect amenities from hotel name, location keywords, and review text.
    Returns a deduplicated list of amenity labels.
    """
    amenities = set()
    text = (name + " " + location + " " + " ".join(r.get("review_text", "") for r in reviews)).lower()

    rules = [
        (["pool", "swimming", "swim"],        "Swimming Pool"),
        (["beach", "oceanfront", "seafront", "beachfront"], "Beach Access"),
        (["spa", "wellness", "massage", "ayurved"], "Spa & Wellness"),
        (["gym", "fitness", "workout"],       "Fitness Centre"),
        (["restaurant", "dining", "cuisine", "buffet"], "Restaurant"),
        (["bar", "cocktail", "lounge"],       "Bar / Lounge"),
        (["wifi", "wi-fi", "wireless", "internet"], "Free WiFi"),
        (["parking", "car park", "valet"],    "Parking"),
        (["airport", "transfer", "shuttle"], "Airport Transfer"),
        (["garden", "nature", "wildlife", "safari", "yala", "national park"], "Nature / Wildlife"),
        (["lake", "river", "waterfall"],      "Lakeside / Riverside"),
        (["rooftop", "roof top"],             "Rooftop"),
        (["conference", "meeting", "business"], "Business Facilities"),
        (["kids", "children", "family", "playground"], "Family Friendly"),
        (["pet", "dog"],                       "Pet Friendly"),
        (["view", "scenic", "mountain", "hill top", "hilltop"], "Scenic Views"),
        (["tea", "plantation"],               "Tea Plantation"),
        (["surf", "surfing"],                 "Surfing"),
        (["diving", "snorkel"],               "Water Sports"),
        (["ac", "air condition", "air-condition"], "Air Conditioning"),
        (["room service"],                    "Room Service"),
    ]
    # WiFi is basically universal — always include it
    amenities.add("Free WiFi")

    for keywords, label in rules:
        if any(kw in text for kw in keywords):
            amenities.add(label)

    return sorted(amenities)


def _derive_price_range(rating: float, name: str) -> dict:
    """Return a realistic nightly price range (USD) based on star rating."""
    name_lower = name.lower()
    # Well-known luxury brands get a premium
    luxury_brands = ["jetwing", "cinnamon", "heritance", "shangri", "hilton", "marriott",
                     "oberoi", "amangalla", "uga bay", "amanwella", "anantara"]
    is_luxury = any(b in name_lower for b in luxury_brands)

    if rating >= 4.8 or is_luxury:
        return {"min": 200, "max": 450, "currency": "USD", "tier": "Luxury"}
    elif rating >= 4.3:
        return {"min": 100, "max": 200, "currency": "USD", "tier": "Upscale"}
    elif rating >= 3.8:
        return {"min": 55, "max": 100, "currency": "USD", "tier": "Mid-range"}
    elif rating >= 3.0:
        return {"min": 25, "max": 55, "currency": "USD", "tier": "Budget-friendly"}
    else:
        return {"min": 15, "max": 25, "currency": "USD", "tier": "Economy"}


def _positive_pct(hotel: dict) -> int:
    """
    Returns the positive-review percentage (0-100).
    Fast path: read the pre-computed 'positive_pct' stored by sync_nlp_sentiment.py.
    Fallback: count reviews with sentiment_label == 'positive'.
    """
    if "positive_pct" in hotel:
        return int(hotel["positive_pct"])
    reviews = hotel.get("reviews", [])
    if not reviews:
        return 0
    positive = sum(
        1 for r in reviews
        if str(r.get("sentiment_label", "")).lower() == "positive"
    )
    return round(positive / len(reviews) * 100)


def _derive_description(hotel: dict) -> str:
    """Build a short description from actual top review snippets and sentiment."""
    reviews = hotel.get("reviews", [])
    sentiment = hotel.get("avg_sentiment_score", 0)
    name = hotel.get("name", "This property")
    location = hotel.get("location", "Sri Lanka")

    # Pick top positive review snippet (≤120 chars)
    # Sort reviews by sentiment descending and pick first short one
    positive = sorted(
        [r for r in reviews if r.get("sentiment_label") == "positive" and len(r.get("review_text", "")) > 40],
        key=lambda r: r.get("sentiment_score", 0),
        reverse=True
    )
    snippet = ""
    for r in positive[:5]:
        text = r.get("review_text", "").strip()
        # Take first sentence only
        first_sentence = text.split(".")[0]
        if 30 < len(first_sentence) <= 130:
            snippet = f'"{first_sentence.capitalize()}."'
            break

    sentiment_word = "excellent" if sentiment > 0.7 else "positive" if sentiment > 0.5 else "mixed"
    desc = (
        f"{name} is situated in {location}. "
        f"With a {hotel.get('rating', 0):.1f}-star guest rating from {hotel.get('total_reviews', 0)} reviews, "
        f"guests report {sentiment_word} overall experiences."
    )
    if snippet:
        desc += f" Guests say: {snippet}"
    return desc


# ============== GOOGLE PLACES API INTEGRATION ==============

# Map Google Places types → friendly amenity labels
_PLACES_TYPE_MAP = {
    "spa": "Spa & Wellness",
    "gym": "Fitness Centre",
    "restaurant": "Restaurant",
    "bar": "Bar / Lounge",
    "night_club": "Bar / Lounge",
    "parking": "Parking",
    "swimming_pool": "Swimming Pool",
    "beach": "Beach Access",
    "airport_shuttle": "Airport Transfer",
    "wifi": "Free WiFi",
    "meeting_room": "Business Facilities",
    "pet_friendly": "Pet Friendly",
    "rooftop": "Rooftop",
    "casino": "Casino",
    "golf_course": "Golf Course",
    "water_park": "Water Park",
    "tennis_court": "Tennis Court",
}

# Price level label mapping
_PRICE_LEVEL_LABELS = {
    1: {"label": "Budget", "tier": "Budget-friendly"},
    2: {"label": "Moderate ($$)", "tier": "Mid-range"},
    3: {"label": "Upscale ($$$)", "tier": "Upscale"},
    4: {"label": "Luxury ($$$$)", "tier": "Luxury"},
}


async def fetch_places_data(hotel_name: str, location: str, hotel_id: int) -> Optional[dict]:
    """
    Fetch hotel details from Google Places API (Text Search + Place Details).
    Results are cached in MongoDB to avoid repeated API calls.

    Returns a dict with: amenities, price_level, price_label, website, phone, editorial_summary
    or None if the lookup fails.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        logger.warning("GOOGLE_MAPS_API_KEY not set — skipping Places lookup")
        return None

    # ---- Check MongoDB cache first ----
    cache_key = f"places_{hotel_id}"
    cached = await db.places_cache.find_one({"_id": cache_key})
    if cached:
        logger.info(f"[Places] Cache hit for hotel_id={hotel_id}")
        return cached.get("data")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Step 1: Text Search to get place_id
            search_query = f"{hotel_name} {location} Sri Lanka hotel"
            text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            search_resp = await client.get(text_search_url, params={
                "query": search_query,
                "type": "lodging",
                "key": api_key,
            })
            search_data = search_resp.json()

            if search_data.get("status") != "OK" or not search_data.get("results"):
                logger.warning(f"[Places] No results for '{search_query}': {search_data.get('status')}")
                # Cache empty result so we don't retry on every request
                await db.places_cache.update_one(
                    {"_id": cache_key},
                    {"$set": {"_id": cache_key, "data": None, "cached_at": datetime.utcnow()}},
                    upsert=True,
                )
                return None

            place_id = search_data["results"][0]["place_id"]

            # Step 2: Place Details for full info
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_resp = await client.get(details_url, params={
                "place_id": place_id,
                "fields": (
                    "name,formatted_address,formatted_phone_number,"
                    "website,price_level,types,editorial_summary,"
                    "opening_hours,url"
                ),
                "key": api_key,
            })
            details_data = details_resp.json()

            if details_data.get("status") != "OK":
                logger.warning(f"[Places] Place Details failed: {details_data.get('status')}")
                return None

            result = details_data.get("result", {})

            # Map place types → amenity labels
            place_types = result.get("types", [])
            amenities = list({
                _PLACES_TYPE_MAP[t]
                for t in place_types
                if t in _PLACES_TYPE_MAP
            })
            # Always include WiFi for hotels
            if "Free WiFi" not in amenities:
                amenities.append("Free WiFi")
            amenities.sort()

            price_level = result.get("price_level")  # 1-4 or None
            price_meta = _PRICE_LEVEL_LABELS.get(price_level, {}) if price_level else {}

            editorial = result.get("editorial_summary", {}).get("overview", "")

            places_data = {
                "amenities": amenities,
                "price_level": price_level,
                "price_label": price_meta.get("label"),
                "price_tier": price_meta.get("tier"),
                "website": result.get("website"),
                "phone": result.get("formatted_phone_number"),
                "google_maps_url": result.get("url"),
                "editorial_summary": editorial,
                "place_id": place_id,
            }

            # Cache in MongoDB (TTL: 30 days — set TTL index manually or just store timestamp)
            await db.places_cache.update_one(
                {"_id": cache_key},
                {"$set": {"_id": cache_key, "data": places_data, "cached_at": datetime.utcnow()}},
                upsert=True,
            )
            logger.info(f"[Places] Cached data for hotel_id={hotel_id}: price_level={price_level}, amenities={amenities}")
            return places_data

    except Exception as e:
        logger.error(f"[Places] Error fetching data for hotel_id={hotel_id}: {e}", exc_info=True)
        return None


@api_router.get("/hotels/{hotel_id}")
async def get_hotel_details(hotel_id: int):

    hotel = await db.hotels.find_one({"hotel_id": hotel_id})
    if not hotel:
        raise HTTPException(status_code=404, detail="Hotel not found")

    rating = float(hotel.get("rating", 3.5))
    reviews = hotel.get("reviews", [])

    # Derived fallbacks (used when Places API has no data)
    derived_amenities = _derive_amenities(hotel.get("name", ""), hotel.get("location", ""), reviews)
    derived_price    = _derive_price_range(rating, hotel.get("name", ""))
    description      = _derive_description(hotel)

    # ---- Enrich with Google Places API ----
    places = await fetch_places_data(hotel.get("name", ""), hotel.get("location", ""), hotel_id)

    # Prefer Places data where available; fall back to derived
    if places:
        amenities   = places.get("amenities") or derived_amenities
        price_tier  = places.get("price_tier") or derived_price["tier"]
        price_label = places.get("price_label")
        # Keep min/max from derived (Places only gives a level, not a dollar range)
        price_info  = {
            "min": derived_price["min"],
            "max": derived_price["max"],
            "currency": "USD",
            "tier": price_tier,
            "label": price_label,
            "level": places.get("price_level"),   # 1-4
        }
        # Prefer editorial summary from Google over our generated one
        if places.get("editorial_summary"):
            description = places["editorial_summary"]
        website          = places.get("website")
        phone            = places.get("phone")
        google_maps_url  = places.get("google_maps_url")
    else:
        amenities   = derived_amenities
        price_info  = {**derived_price, "label": None, "level": None}
        website         = None
        phone           = None
        google_maps_url = None

    return {
        "id": hotel['_id'],
        "hotel_id": hotel['hotel_id'],
        "name": hotel['name'],
        "location": hotel['location'],
        "rating": rating,
        "total_reviews": hotel['total_reviews'],
        "positive_reviews": hotel.get('positive_reviews', 0),
        "negative_reviews": hotel.get('negative_reviews', 0),
        "neutral_reviews": hotel.get('neutral_reviews', 0),
        "positive_pct": hotel.get('positive_pct', _positive_pct(hotel)),
        "negative_pct": hotel.get('negative_pct', 0),
        "neutral_pct": hotel.get('neutral_pct', 0),
        "avg_sentiment_score": hotel['avg_sentiment_score'],
        "reviews": reviews,
        "amenities": amenities,
        "price_info": price_info,
        "description": description,
        "website": website,
        "phone": phone,
        "google_maps_url": google_maps_url,
        "emotion_breakdown": {
            "joy":      sum(1 for r in reviews if r.get('emotion_label') == 'joy'),
            "surprise": sum(1 for r in reviews if r.get('emotion_label') == 'surprise'),
            "neutral":  sum(1 for r in reviews if r.get('emotion_label') == 'neutral'),
            "sadness":  sum(1 for r in reviews if r.get('emotion_label') == 'sadness'),
            "anger":    sum(1 for r in reviews if r.get('emotion_label') == 'anger'),
            "fear":     sum(1 for r in reviews if r.get('emotion_label') == 'fear'),
        }
    }



@api_router.post("/recommendations")
async def get_recommendations(request: RecommendationRequest, current_user: Optional[dict] = Depends(get_current_user_optional)):
    try:
        # Get all hotels with filters
        query = {}
        if request.location:
            query['location'] = {"$regex": request.location, "$options": "i"}
        
        # Filter by minimum rating if provided
        if request.min_rating is not None:
            query['rating'] = {"$gte": request.min_rating}
            logger.info(f"Filtering by minimum rating: {request.min_rating}")
        
        hotels = await db.hotels.find(query).to_list(1000)
        
        logger.info(f"Found {len(hotels)} hotels for recommendations")
        
        if not hotels:
            return []

        # ── Resolve the effective user ────────────────────────────────────────
        # Priority 1: JWT-authenticated user (current_user from Bearer token)
        # Priority 2: MongoDB lookup by request.user_id (mobile users without JWT)
        # Priority 3: Truly anonymous — use preferences passed in the request body
        effective_user = current_user
        if not effective_user and request.user_id:
            effective_user = await db.users.find_one({"_id": request.user_id})
            if effective_user:
                logger.info(f"Resolved user by request.user_id: {request.user_id}")

        # Merge preferences: request body wins, then stored profile, then empty
        user_preferences = list(
            request.preferences or []
            or (effective_user.get('preferences', []) if effective_user else [])
        )

        # Amenities: request body > stored profile
        selected_amenities = (
            request.amenities if request.amenities
            else (effective_user.get('selected_amenities', []) if effective_user else [])
        )
        
        logger.info(f"=== RECOMMENDATION REQUEST ===")
        logger.info(f"User: {effective_user['_id'] if effective_user else request.user_id + ' (cold-start)'}")
        logger.info(f"Location: {request.location}")
        logger.info(f"Preferences: {user_preferences}")
        logger.info(f"Selected amenities: {selected_amenities}")
        logger.info(f"Hotels in pool: {len(hotels)}")
        
        # EARLY EXIT: If location doesn't support selected amenities, return empty immediately
        if selected_amenities and request.location:
            location_lower = request.location.lower()
            for amenity in selected_amenities:
                if not recommendation_engine._check_location_has_amenity(location_lower, amenity):
                    logger.warning(f"⚠️ EARLY EXIT: Location '{request.location}' doesn't support amenity '{amenity}'")
                    return []
        
        if hotels:
            sample_locations = [h.get('location', 'N/A') for h in hotels[:3]]
            logger.info(f"Sample hotel locations: {sample_locations}")
        
        # Always use the hybrid engine — it handles cold-start internally.
        # Pass effective_user's _id if available, otherwise fall back to request.user_id
        # so each user gets a deterministically different shuffle.
        engine_user_id = (
            effective_user['_id'] if effective_user else request.user_id
        )
        recommended_ids = await recommendation_engine.hybrid_recommendation_with_vgg(
            engine_user_id,
            hotels,
            selected_amenities,
            request.limit,
            user_preferences=user_preferences,
            db_ref=db
        )
        
        logger.info(f"Got {len(recommended_ids)} recommended IDs")
        
        # Fetch full hotel details in sorted order
        base_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
        recommended_hotels = []
        for hotel_id in recommended_ids:
            hotel = next((h for h in hotels if h['hotel_id'] == hotel_id), None)
            if hotel:
                hotel_rating = float(hotel['rating']) if not np.isnan(hotel['rating']) else 3.5
                
                # Double-check rating filter (in case rating was modified after query)
                if request.min_rating is not None and hotel_rating < request.min_rating:
                    logger.debug(f"Skipping hotel {hotel_id} - rating {hotel_rating} < min_rating {request.min_rating}")
                    continue
                
                recommended_hotels.append({
                    "id": hotel['_id'],
                    "hotel_id": hotel['hotel_id'],
                    "name": hotel['name'],
                    "location": hotel['location'],
                    "rating": hotel_rating,
                    "total_reviews": hotel['total_reviews'],
                    "avg_sentiment_score": float(hotel['avg_sentiment_score']) if not np.isnan(hotel['avg_sentiment_score']) else 0.0,
                    "positive_pct": _positive_pct(hotel),
                    "recommendation_score": round(hotel['rating'] * 20, 1) if not np.isnan(hotel['rating']) else 70.0,
                    "image_url": f"{base_url}/api/hotel-images/{hotel['hotel_id']}/proxy?index=1",
                    "price_info": _derive_price_range(hotel_rating, hotel.get('name', '')),
                })
        
        logger.info(f"Returning {len(recommended_hotels)} recommended hotels (sorted by match)")
        return recommended_hotels
    except Exception as e:
        logger.error(f"Error in recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Submit a favourite / rating for a hotel.
    Auth: JWT Bearer token (preferred) OR user_id in the request body.
    This mirrors the /recommendations endpoint pattern so mobile users
    who authenticate via the Flask backend can still submit favourites.
    """
    # Resolve the effective user — same priority chain as /recommendations
    effective_user = current_user
    if not effective_user and feedback.user_id:
        effective_user = await db.users.find_one({"_id": feedback.user_id})

    # Determine the real user_id to store
    resolved_user_id = (
        effective_user["_id"] if effective_user else feedback.user_id
    )
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    # Upsert: one feedback doc per (user, hotel) so repeated hearts just update rating
    await db.feedback.update_one(
        {"user_id": resolved_user_id, "hotel_id": feedback.hotel_id},
        {"$set": {
            "user_id":    resolved_user_id,
            "hotel_id":   feedback.hotel_id,
            "rating":     feedback.rating,
            "comment":    feedback.comment,
            "updated_at": datetime.utcnow(),
        }, "$setOnInsert": {"_id": str(uuid.uuid4()), "created_at": datetime.utcnow()}},
        upsert=True,
    )

    # Update user interaction history (only for known users)
    if effective_user:
        await db.users.update_one(
            {"_id": resolved_user_id},
            {"$push": {"interaction_history": {
                "hotel_id":  feedback.hotel_id,
                "rating":    feedback.rating,
                "timestamp": datetime.utcnow(),
            }}}
        )

    return {"message": "Feedback submitted successfully"}


@api_router.get("/favourites/{user_id}")
async def get_user_favourites(user_id: str):
    """
    Return the list of hotel_ids that a user has liked (rating >= 4.0).
    Used by the mobile app to restore heart state on screen load.
    """
    cursor = db.feedback.find(
        {"user_id": user_id, "rating": {"$gte": 4.0}},
        {"hotel_id": 1, "_id": 0},
    )
    docs = await cursor.to_list(500)
    return {"hotel_ids": [d["hotel_id"] for d in docs]}

# ============== HOTEL IMAGES ==============

GOOGLE_DRIVE_FOLDER_ID = "1LwQm93QxqnwWTGv75xCejyu8a3iT7SGn"

# Thread pool for running synchronous Google Drive API calls
drive_executor = ThreadPoolExecutor(max_workers=5)

# Google Drive API setup
def get_drive_service():
    """
    Initialize and return Google Drive API service.
    Supports Service Account authentication via environment variable JSON or file.
    """
    try:
        # 1. Try GOOGLE_CREDENTIALS_JSON (recommended for Railway)
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            import json
            info = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            return build('drive', 'v3', credentials=credentials)

        # 2. Try GOOGLE_APPLICATION_CREDENTIALS (file path)
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            return build('drive', 'v3', credentials=credentials)
        
        # 3. Default path
        default_path = ROOT_DIR / 'service-account-key.json'
        if default_path.exists():
            credentials = service_account.Credentials.from_service_account_file(
                str(default_path),
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            return build('drive', 'v3', credentials=credentials)

        logger.warning("Google Drive credentials not found (checked GOOGLE_CREDENTIALS_JSON, GOOGLE_APPLICATION_CREDENTIALS, and default file)")
        return None
        
    except GoogleAuthError as e:
        logger.error(f"Google Drive authentication error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing Google Drive service: {e}")
        return None

def _find_hotel_folder_id_sync(drive_service, parent_folder_id: str, hotel_id: str) -> Optional[str]:
    """
    Find the folder ID for a specific hotel within the parent Google Drive folder
    """
    try:
        # List all folders in the parent folder
        query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{hotel_id}' and trashed=false"
        
        results = drive_service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=10
        ).execute()
        
        folders = results.get('files', [])
        if folders:
            return folders[0]['id']
        
        # Also try with hotel_id as string if it's a number
        if hotel_id.isdigit():
            query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and (name='{hotel_id}' or name='{int(hotel_id)}') and trashed=false"
            results = drive_service.files().list(
                q=query,
                fields="files(id, name)",
                pageSize=10
            ).execute()
            folders = results.get('files', [])
            if folders:
                return folders[0]['id']
        
        return None
    except HttpError as e:
        logger.error(f"Error finding hotel folder: {e}")
        return None

async def find_hotel_folder_id(drive_service, parent_folder_id: str, hotel_id: str) -> Optional[str]:
    """Async wrapper for finding hotel folder"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        drive_executor,
        _find_hotel_folder_id_sync,
        drive_service,
        parent_folder_id,
        hotel_id
    )

def _get_image_file_id_sync(drive_service, folder_id: str, index: int = 1) -> Optional[str]:
    """
    Get the file ID of an image in a folder by index (synchronous)
    Images are typically named 1.jpg, 2.jpg, etc. or 1.png, 2.png, etc.
    """
    try:
        # List all image files in the folder, sorted by name
        query = f"'{folder_id}' in parents and (mimeType='image/jpeg' or mimeType='image/jpg' or mimeType='image/png' or mimeType='image/webp') and trashed=false"
        
        results = drive_service.files().list(
            q=query,
            fields="files(id, name)",
            orderBy="name",
            pageSize=100
        ).execute()
        
        files = results.get('files', [])
        if not files:
            return None
        
        # Try to find file by name pattern (1.jpg, 1.png, etc.)
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            target_name = f"{index}.{ext}"
            for file in files:
                if file['name'].lower() == target_name.lower():
                    return file['id']
        
        # If exact match not found, return file at index (0-based)
        if index > 0 and index <= len(files):
            return files[index - 1]['id']
        
        # Default to first file
        return files[0]['id'] if files else None
        
    except HttpError as e:
        logger.error(f"Error getting image file: {e}")
        return None

async def get_image_file_id(drive_service, folder_id: str, index: int = 1) -> Optional[str]:
    """Async wrapper for getting image file ID"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        drive_executor,
        _get_image_file_id_sync,
        drive_service,
        folder_id,
        index
    )

def _download_image_sync(drive_service, file_id: str) -> Optional[bytes]:
    """
    Download image file from Google Drive (synchronous)
    """
    try:
        request = drive_service.files().get_media(fileId=file_id)
        import io
        
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        return fh.getvalue()
    except HttpError as e:
        logger.error(f"Error downloading image: {e}")
        return None

async def download_image(drive_service, file_id: str) -> Optional[bytes]:
    """Async wrapper for downloading image"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        drive_executor,
        _download_image_sync,
        drive_service,
        file_id
    )

@api_router.get("/hotel-images/{hotel_id}")
async def get_hotel_image(hotel_id: str, index: int = 1):
    """
    Get hotel image from Google Drive and return JSON with image URL
    
    The Google Drive folder structure:
    - Parent folder: GOOGLE_DRIVE_FOLDER_ID
    - Hotel folders: Named with hotel_id (e.g., "1", "2", "123")
    - Images: Numbered sequentially (1.jpg, 2.jpg, etc.)
    
    Args:
        hotel_id: The hotel ID (folder name in Google Drive)
        index: Image index (defaults to 1 for first image)
    
    Returns:
        JSON with image_url pointing to proxy endpoint
    """
    if not GOOGLE_DRIVE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Google Drive API not available. Please install: pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib"
        )
    
    try:
        # Get Google Drive service
        drive_service = get_drive_service()
        if not drive_service:
            raise HTTPException(
                status_code=503,
                detail="Google Drive service not available. Please configure GOOGLE_APPLICATION_CREDENTIALS."
            )
        
        # Find the hotel folder
        hotel_folder_id = await find_hotel_folder_id(drive_service, GOOGLE_DRIVE_FOLDER_ID, hotel_id)
        if not hotel_folder_id:
            logger.warning(f"Hotel folder not found for hotel_id: {hotel_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Hotel folder not found for hotel ID: {hotel_id}"
            )
        
        # Get the image file ID
        image_file_id = await get_image_file_id(drive_service, hotel_folder_id, index)
        if not image_file_id:
            logger.warning(f"No images found in hotel folder: {hotel_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No images found for hotel ID: {hotel_id}"
            )
        
        # Return proxy URL that will serve the image directly
        # This works better with React Native Image components
        from fastapi.responses import JSONResponse
        base_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
        image_url = f"{base_url}/api/hotel-images/{hotel_id}/proxy?index={index}"
        
        return JSONResponse({
            "image_url": image_url,
            "file_id": image_file_id,
            "hotel_id": hotel_id
        })
        
    except HTTPException:
        raise
    except HttpError as e:
        logger.error(f"Google Drive API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting hotel image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get hotel image: {str(e)}"
        )

@api_router.get("/hotel-images/{hotel_id}/proxy")
async def proxy_hotel_image(hotel_id: str, index: int = 1):
    """
    Proxy hotel image from Google Drive - serves image directly for React Native
    
    This endpoint downloads the image from Google Drive and serves it directly
    with proper headers, which works better with React Native Image components.
    
    Args:
        hotel_id: The hotel ID (folder name in Google Drive)
        index: Image index (defaults to 1 for first image)
    
    Returns:
        StreamingResponse with image data
    """
    if not GOOGLE_DRIVE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Google Drive API not available"
        )
    
    try:
        # Get Google Drive service
        drive_service = get_drive_service()
        if not drive_service:
            raise HTTPException(
                status_code=503,
                detail="Google Drive service not available. Please configure GOOGLE_APPLICATION_CREDENTIALS."
            )
        
        # Find the hotel folder
        hotel_folder_id = await find_hotel_folder_id(drive_service, GOOGLE_DRIVE_FOLDER_ID, hotel_id)
        if not hotel_folder_id:
            logger.warning(f"Hotel folder not found for hotel_id: {hotel_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Hotel folder not found for hotel ID: {hotel_id}"
            )
        
        # Get the image file ID
        image_file_id = await get_image_file_id(drive_service, hotel_folder_id, index)
        if not image_file_id:
            logger.warning(f"No images found in hotel folder: {hotel_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No images found for hotel ID: {hotel_id}"
            )
        
        # Download the image
        image_data = await download_image(drive_service, image_file_id)
        if not image_data:
            raise HTTPException(
                status_code=500,
                detail="Failed to download image from Google Drive"
            )
        
        # Determine content type based on file extension
        # Get file metadata to determine MIME type
        try:
            file_metadata = drive_service.files().get(fileId=image_file_id, fields='mimeType').execute()
            content_type = file_metadata.get('mimeType', 'image/jpeg')
        except:
            content_type = 'image/jpeg'  # Default to JPEG
        
        # Return image as streaming response
        from io import BytesIO
        return StreamingResponse(
            BytesIO(image_data),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f'inline; filename="hotel_{hotel_id}_{index}.jpg"'
            }
        )
        
    except HTTPException:
        raise
    except HttpError as e:
        logger.error(f"Google Drive API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error proxying hotel image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to proxy hotel image: {str(e)}"
        )

@api_router.get("/hotel-images/{hotel_id}/list")
async def list_hotel_images(hotel_id: str, request: Request):
    """
    List all images for a hotel
    
    Returns a list of image URLs that can be used to display all images
    for a hotel in a carousel or gallery.
    
    Args:
        hotel_id: The hotel ID (folder name in Google Drive)
    
    Returns:
        JSON with list of image URLs and count
    """
    if not GOOGLE_DRIVE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Google Drive API not available"
        )
    
    try:
        # Get Google Drive service
        drive_service = get_drive_service()
        if not drive_service:
            raise HTTPException(
                status_code=503,
                detail="Google Drive service not available"
            )
        
        # Find the hotel folder
        hotel_folder_id = await find_hotel_folder_id(drive_service, GOOGLE_DRIVE_FOLDER_ID, hotel_id)
        if not hotel_folder_id:
            logger.warning(f"Hotel folder not found for hotel_id: {hotel_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Hotel folder not found for hotel ID: {hotel_id}"
            )
        
        # List all image files in the folder
        loop = asyncio.get_event_loop()
        query = f"'{hotel_folder_id}' in parents and (mimeType='image/jpeg' or mimeType='image/jpg' or mimeType='image/png' or mimeType='image/webp') and trashed=false"
        
        def _list_images_sync():
            results = drive_service.files().list(
                q=query,
                fields="files(id, name)",
                orderBy="name",
                pageSize=100
            ).execute()
            return results.get('files', [])
        
        files = await loop.run_in_executor(drive_executor, _list_images_sync)
        
        if not files:
            logger.warning(f"No images found in hotel folder: {hotel_id}")
            # Try to return at least one image by attempting to load index 1
            # This handles cases where images exist but query didn't find them
            try:
                image_file_id = await get_image_file_id(drive_service, hotel_folder_id, 1)
                if image_file_id:
                    base_url = str(request.base_url).rstrip('/')
                    image_url = f"{base_url}/api/hotel-images/{hotel_id}/proxy?index=1"
                    from fastapi.responses import JSONResponse
                    return JSONResponse({
                        "images": [{
                            "url": image_url,
                            "index": 1,
                            "file_id": image_file_id,
                            "name": "1.jpg"
                        }],
                        "count": 1,
                        "hotel_id": hotel_id
                    })
            except:
                pass
            
            # Return empty list if no images found at all
            from fastapi.responses import JSONResponse
            return JSONResponse({
                "images": [],
                "count": 0,
                "hotel_id": hotel_id
            })
        
        # Build list of image URLs
        base_url = str(request.base_url).rstrip('/')
        image_urls = []
        
        for index, file in enumerate(files, start=1):
            image_url = f"{base_url}/api/hotel-images/{hotel_id}/proxy?index={index}"
            image_urls.append({
                "url": image_url,
                "index": index,
                "file_id": file['id'],
                "name": file['name']
            })
        
        from fastapi.responses import JSONResponse
        return JSONResponse({
            "images": image_urls,
            "count": len(image_urls),
            "hotel_id": hotel_id
        })
        
    except HTTPException:
        raise
    except HttpError as e:
        logger.error(f"Google Drive API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error listing hotel images: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list hotel images: {str(e)}"
        )

@api_router.get("/search/suggestions")
async def get_search_suggestions(q: str, limit: int = 10):
    """
    Get search suggestions for autocomplete.
    Returns hotel names and locations matching the query.
    """
    if not q or len(q.strip()) < 2:
        return []
    
    query = {
        "$or": [
            {"name": {"$regex": q, "$options": "i"}},
            {"location": {"$regex": q, "$options": "i"}}
        ]
    }
    
    hotels = await db.hotels.find(query).limit(limit).to_list(limit)
    
    # Create suggestions list with unique hotel names and locations
    suggestions = []
    seen_names = set()
    seen_locations = set()
    
    for h in hotels:
        # Add hotel name suggestions
        if h['name'] not in seen_names:
            suggestions.append({
                "type": "hotel",
                "text": h['name'],
                "location": h['location'],
                "hotel_id": h['hotel_id']
            })
            seen_names.add(h['name'])
        
        # Add location suggestions
        if h['location'] not in seen_locations:
            suggestions.append({
                "type": "location",
                "text": h['location'],
                "hotel_id": None
            })
            seen_locations.add(h['location'])
    
    return suggestions[:limit]

@api_router.get("/search")
async def search_hotels(q: str, limit: int = 20):
    query = {
        "$or": [
            {"name": {"$regex": q, "$options": "i"}},
            {"location": {"$regex": q, "$options": "i"}}
        ]
    }
    
    hotels = await db.hotels.find(query).limit(limit).to_list(limit)
    base_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    
    return [
        {
            "id": h['_id'],
            "hotel_id": h['hotel_id'],
            "name": h['name'],
            "location": h['location'],
            "rating": h['rating'],
            "total_reviews": h['total_reviews'],
            "avg_sentiment_score": h['avg_sentiment_score'],
            "image_url": f"{base_url}/api/hotel-images/{h['hotel_id']}/proxy?index=1"
        }
        for h in hotels
    ]

# ============== GEMINI AI INSIGHTS ==============

def _build_gemini_prompt(hotel: dict, reviews: list) -> str:
    """
    Build a structured prompt for Gemini using real hotel data.
    """
    name = hotel.get('name', 'Unknown Hotel')
    location = hotel.get('location', 'Sri Lanka')
    rating = hotel.get('rating', 0)
    total_reviews = hotel.get('total_reviews', 0)
    sentiment = hotel.get('avg_sentiment_score', 0)
    sentiment_word = "excellent" if sentiment > 0.7 else "positive" if sentiment > 0.5 else "mixed"

    # Pick top positive and negative review snippets (max 3 each)
    positive_snippets = [
        r.get('review_text', '')[:200]
        for r in sorted(reviews, key=lambda r: r.get('sentiment_score', 0), reverse=True)
        if r.get('sentiment_label') == 'positive' and len(r.get('review_text', '')) > 30
    ][:3]

    negative_snippets = [
        r.get('review_text', '')[:200]
        for r in sorted(reviews, key=lambda r: r.get('sentiment_score', 0))
        if r.get('sentiment_label') == 'negative' and len(r.get('review_text', '')) > 30
    ][:3]

    pos_text = "\n".join(f"- {s}" for s in positive_snippets) or "- No positive reviews available"
    neg_text = "\n".join(f"- {s}" for s in negative_snippets) or "- No negative reviews available"

    prompt = f"""
You are an expert travel analyst. Analyze the following hotel and generate a concise, helpful AI insight report.

HOTEL DATA:
- Name: {name}
- Location: {location}, Sri Lanka
- Guest Rating: {rating:.1f}/5 from {total_reviews} reviews
- Overall Sentiment: {sentiment_word} (score: {sentiment:.2f})

POSITIVE REVIEW SNIPPETS:
{pos_text}

NEGATIVE REVIEW SNIPPETS:
{neg_text}

Generate a JSON response with exactly these fields (no extra text, only valid JSON):
{{
  "verdict": "One sentence overall verdict (max 20 words)",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "watch_out": ["caveat 1", "caveat 2"],
  "best_for": ["traveller type 1", "traveller type 2", "traveller type 3"],
  "best_time": "Best season/time to visit this location in Sri Lanka (1-2 sentences)",
  "guest_quote": "Most compelling single quote from a real guest review (max 30 words)",
  "ai_score": <number between 0 and 100 representing overall AI recommendation score>
}}
"""
    return prompt.strip()


@api_router.get("/ai-insights/{hotel_id}")
async def get_ai_insights(hotel_id: int):
    """
    Generate AI-powered hotel insights using Google Gemini.
    Results are cached in MongoDB for 7 days to avoid repeated API calls.
    """
    if not GEMINI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Gemini AI not available. Install google-generativeai.")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured in .env")

    # ── Check MongoDB cache (7-day TTL) ──────────────────────────────────────
    cache_key = f"ai_insights_{hotel_id}"
    cached = await db.ai_insights_cache.find_one({"_id": cache_key})
    if cached:
        cached_at = cached.get("cached_at")
        if cached_at and (datetime.utcnow() - cached_at).days < 7:
            logger.info(f"[Gemini] Cache hit for hotel_id={hotel_id}")
            return cached["data"]

    # ── Fetch hotel from MongoDB ──────────────────────────────────────────────
    hotel = await db.hotels.find_one({"hotel_id": hotel_id})
    if not hotel:
        raise HTTPException(status_code=404, detail="Hotel not found")

    reviews = hotel.get("reviews", [])

    try:
        genai.configure(api_key=api_key)
        # Use a stable model name that exists in all versions
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = _build_gemini_prompt(hotel, reviews)

        logger.info(f"[Gemini] Requesting insights for hotel_id={hotel_id} ({hotel.get('name')})")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1500,
                response_mime_type="application/json",
            )
        )

        # Parse JSON from Gemini response
        raw_text = response.text.strip()
        # Remove markdown code fences if Gemini wraps the JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        insights = json.loads(raw_text)
        insights["hotel_id"] = hotel_id
        insights["hotel_name"] = hotel.get("name", "")
        insights["generated_at"] = datetime.utcnow().isoformat()

        # ── Cache in MongoDB ─────────────────────────────────────────────────
        await db.ai_insights_cache.update_one(
            {"_id": cache_key},
            {"$set": {"_id": cache_key, "data": insights, "cached_at": datetime.utcnow()}},
            upsert=True,
        )
        logger.info(f"[Gemini] Insights generated and cached for hotel_id={hotel_id}")
        return insights

    except json.JSONDecodeError as e:
        logger.error(f"[Gemini] Failed to parse JSON response: {e}\nRaw: {raw_text[:300]}")
        raise HTTPException(status_code=502, detail="Gemini returned an invalid response. Please try again.")
    except Exception as e:
        logger.error(f"[Gemini] Error generating insights for hotel_id={hotel_id}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")


# Include router
app.include_router(api_router)

# Logging - MUST be before using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log Google Drive availability status after logger is initialized
if not GOOGLE_DRIVE_AVAILABLE:
    logger.warning("Google Drive API packages not installed. Image serving endpoint will be disabled.")
    logger.warning("To enable: pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load hotels data into MongoDB on startup if not exists"""
    try:
        count = await db.hotels.count_documents({})
        if count == 0:
            logger.info("Loading hotels data into MongoDB...")
            with open(ROOT_DIR / 'hotels_data.json', 'r') as f:
                hotels_data = json.load(f)
            
            # Add _id field
            for hotel in hotels_data:
                hotel['_id'] = str(uuid.uuid4())
            
            await db.hotels.insert_many(hotels_data)
            logger.info(f"✅ Loaded {len(hotels_data)} hotels into MongoDB")
        else:
            logger.info(f"✅ Database already contains {count} hotels")
    except Exception as e:
        logger.error(f"Error loading data: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow connections from other devices on the network
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
