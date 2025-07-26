from fastapi import APIRouter

from app.models.request_model import ChatRequest

router = APIRouter()

@router.get("/chat")
async def chat_api(request: ChatRequest):
    return {"response": f"你说的是:{request.query}"}
