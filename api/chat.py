from fastapi import APIRouter

from models.schemas import ChatRequest, ResponseModel
from schemas.llm_schema import AnswerResponse, QuestionRequest, KnowledgeRequest
# from services import qa_system  # 暂时注释掉，避免导入错误
from services.llm_service import ask_llm

router = APIRouter()

@router.get("/chat")
async def chat_api(request: ChatRequest):
    return ResponseModel(data={"response": f"你说的是:{request.query}"})


@router.post("/items", response_model=ResponseModel)
def create_item(item: ChatRequest):
    # 假装这里存入数据库了
    saved_item = {"id": 1, "name": item.name, "description": item.description}

    return ResponseModel(data=saved_item)
