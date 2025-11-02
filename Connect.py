import asyncio
from google import genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key="AIzaSyDoAC6ks_fa4kSV_zhjvR4UaZUI1WRYqGM")

model = "gemini-live-2.5-flash-preview"
config = {"response_modalities": ["TEXT"]} #[]裡面可以選擇以音訊或文字來回覆

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        print("Session started")

if __name__ == "__main__":
    asyncio.run(main())