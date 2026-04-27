import os
import motor.motor_asyncio
import asyncio
from dotenv import load_dotenv

async def verify():
    load_dotenv(r'C:\Users\User\Downloads\Research_2025\Research_2025\Research-main\backend\.env')
    client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv('MONGO_URL'))
    db = client[os.getenv('DB_NAME')]
    
    # Check 5 hotels covering different sentiment ranges
    test_ids = [0, 42, 13, 27, 34] # Aditya, Jetwing Lake, Citrus Hikkaduwa, Heritance Ahungalla, Hotel J Negombo
    
    print(f"{'Hotel Name':<35} | {'Total':<5} | {'Pos%':<4} | {'Neu%':<4} | {'Neg%':<4}")
    print("-" * 65)
    
    for hid in test_ids:
        hotel = await db.hotels.find_one({'hotel_id': hid})
        if hotel:
            name = hotel.get('name', 'Unknown')
            total = hotel.get('total_reviews', 0)
            pos = hotel.get('positive_pct', 0)
            neu = hotel.get('neutral_pct', 0)
            neg = hotel.get('negative_pct', 0)
            print(f"{name[:35]:<35} | {total:<5} | {pos:<4} | {neu:<4} | {neg:<4}")
        else:
            print(f"Hotel {hid} not found")

if __name__ == '__main__':
    asyncio.run(verify())
