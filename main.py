from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
import openai
import os
import logging
import asyncio

# Initialize FastAPI
app = FastAPI(title="GPT WhatsApp Assistant", version="1.1")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant")

# Load API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simulated in-memory business DB
BUSINESS_PROFILES = {
    "smilebright": {
        "name": "SmileBright Dental Clinic",
        "location": "Austin, Texas",
        "services": ["Teeth cleaning", "Whitening", "Fillings", "Check-ups"],
        "hours": "Mon–Fri, 9am–5pm"
    },
    "sparkbarbers": {
        "name": "Spark Barbershop",
        "location": "Miami, Florida",
        "services": ["Haircuts", "Beard Trim", "Shave"],
        "hours": "Tue–Sat, 10am–6pm"
    }
}

# Pydantic model for cleaner validation
class ChatRequest(BaseModel):
    message: str
    business_id: str

# Helper: Generate system prompt per business
def create_prompt(business: Dict[str, str]) -> str:
    return f"""
You are a helpful assistant for {business['name']} in {business['location']}.
Services offered: {', '.join(business['services'])}.
Hours: {business['hours']}.
Your job is to respond in a professional, friendly way and assist clients in booking services or answering questions.
"""

# Helper: GPT interaction
async def get_gpt_response(message: str, system_prompt: str) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=503, detail="AI is currently unavailable.")


# POST /chat — Multi-tenant WhatsApp-style bot
@app.post("/chat", tags=["Chat"])
async def chat_endpoint(payload: ChatRequest):
    business_id = payload.business_id.lower()
    user_msg = payload.message.strip()

    if business_id not in BUSINESS_PROFILES:
        raise HTTPException(status_code=404, detail="Business profile not found.")

    prompt = create_prompt(BUSINESS_PROFILES[business_id])

    # Retry up to 2x
    for attempt in range(3):
        try:
            reply = await get_gpt_response(user_msg, prompt)
            return JSONResponse(content={"reply": reply})
        except HTTPException:
            if attempt == 2:
                raise
            await asyncio.sleep(1)
