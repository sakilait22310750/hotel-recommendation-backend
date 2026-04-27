import os, asyncio
import motor.motor_asyncio
import pandas as pd
from dotenv import load_dotenv

async def check():
    load_dotenv(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\.env')
    client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv('MONGO_URL'))
    db = client[os.getenv('DB_NAME')]
    
    # Let's find Uga Chena Huts to be specific
    hotel = await db.hotels.find_one({'name': {'$regex': 'Uga Chena Huts', '$options': 'i'}})
    if hotel and 'reviews' in hotel:
        print(f"Hotel: {hotel['name']}")
        print('Reviews in DB:')
        for r in hotel['reviews'][:5]:
             print(f"- Text: {r.get('review_text', '')[:30]}... | Lang: {r.get('language')}")
    else:
        print('Hotel or reviews not found in DB')
    
    # Compare with Excel
    df = pd.read_excel(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\reviews_with_nlp_features.xlsx')
    uga_df = df[df['hotel_name'].str.contains('Uga Chena Huts', case=False, na=False)]
    print(f"\nExcel Data for Uga Chena Huts ({len(uga_df)} reviews):")
    for _, row in uga_df.head(5).iterrows():
         print(f"- Text: {str(row.get('review', ''))[:30]}... | Lang: {row.get('language')}")

if __name__ == '__main__':
    asyncio.run(check())
