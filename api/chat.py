from fastapi import APIRouter

from models.schemas import ChatRequest, ResponseModel
from schemas.llm_schema import AnswerResponse, QuestionRequest, KnowledgeRequest
from services import qa_system
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


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(data: QuestionRequest):
    answer = ask_llm(data.question)
    return AnswerResponse(answer=answer)

@router.post("/qa/ask")
async def ask_question(request: QuestionRequest):
    """问答接口"""
    try:
        answer = qa_system.answer_question(request.question)
        return {"answer": answer, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

@router.post("/qa/add_knowledge")
async def add_knowledge(request: KnowledgeRequest):
    """添加知识库"""
    qa_system.add_knowledge(request.documents)
    return {"message": "知识库更新成功", "status": "success"}
