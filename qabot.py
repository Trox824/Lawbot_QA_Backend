from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
# Cau hinh
model_file = "models/vinallama-7b-chat.Q5_K_M.gguf"
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
vector_db_path = "vectorstores/huggingface512_faiss"

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":1}, max_tokens_limit=512),
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt})
    return llm_chain
# Read tu VectorDB
def read_vectors_db():
    MODEL_NAME = "keepitreal/vietnamese-sbert"
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db
db = read_vectors_db()

llm = CTransformers( model=model_file, model_type="llama", max_new_tokens=1024, temperature=0.01 )
#Tao Prompt
template = """bạn là một trợ lí luật sư AI hữu ích. Dựa vào thông tin pháp luật sau đây để trả lời câu hỏi.
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.\n
Thông tin pháp luật: {context}\n Câu hỏi:\n{question}\n"""
prompt = PromptTemplate(template = template, input_variables=["context", "question"])

llm_chain  = create_qa_chain(prompt, llm, db)

def AskModel(question):
    return llm_chain.invoke({"query": question})
print(AskModel("Người dưới 18 tuổi được xem phim R18"))
# db.similarity_search("tôi muốn hỏi về quyền lợi của người lao động")