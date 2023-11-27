from rag_retriver import Vector_search, GPT_completion_with_vector_search
from fastapi import FastAPI
from pydantic import BaseModel

#Pydantic object
class validation(BaseModel):
    prompt: str
    
#Fast API
app = FastAPI()


@app.post("/Gathnex_Rag_System")
async def retrival_augmented_generation(item: validation):
    rag = Vector_search(item.prompt)
    completion = GPT_completion_with_vector_search(item.prompt, rag)
    return completion
