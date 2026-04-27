import os, asyncio
import motor.motor_asyncio
from dotenv import load_dotenv

async def check():
    load_dotenv(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\.env')
    client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv('MONGO_URL'))
    db = client[os.getenv('DB_NAME')]
    hotel = await db.hotels.find_one({'name': {'$regex': 'Uga Chena', '$options': 'i'}})
    if hotel:
        print('Reviews count:', len(hotel.get('reviews', [])))
        for r in hotel.get('reviews', [])[:5]:
            print(f"user_id: {r.get('user_id')} - text: {str(r.get('review_text', ''))[:20]}")
    else:
        print("Hotel not found")
if __name__ == '__main__':
    asyncio.run(check())
