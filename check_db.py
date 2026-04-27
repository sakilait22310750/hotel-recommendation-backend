import os
import motor.motor_asyncio
import asyncio
from dotenv import load_dotenv

async def check():
    load_dotenv(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\.env')
    client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv('MONGO_URL'))
    db = client[os.getenv('DB_NAME')]
    count = await db.hotels.count_documents({})
    print(f'Total hotels in DB: {count}')
    
    cursor = db.hotels.find({}, {'name': 1, 'positive_pct': 1}).limit(5)
    async for h in cursor:
        print(f"- {h.get('name')}: pos={h.get('positive_pct')}")

if __name__ == '__main__':
    asyncio.run(check())
