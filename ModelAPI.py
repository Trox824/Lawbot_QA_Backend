from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qabot import AskModel
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def PreProcess(response):
    answer = response['result']
    answer = answer.replace('<|im_start|>', '')
    answer = answer.replace('system', '')
    answer = answer.replace('user','')
    answer = answer.replace('<|im_end|>','')
    answer = answer.replace('assistant','')
    resource = response['source_documents']
    return answer + "\n\nTrích nguồn: " + str(resource[0])

@app.get("/")
async def root():
    return {"response": "hello world"}

@app.get("/{question}")
async def root(question):
    return {"response": PreProcess(AskModel(question))}