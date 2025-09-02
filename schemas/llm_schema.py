from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class KnowledgeRequest(BaseModel):
    """知识库添加请求模型"""
    documents: List[str]
