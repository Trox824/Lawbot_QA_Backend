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
vector_db_path = "vectorstores/db_faiss"
# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm
# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt
# Tao simple chain››
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":1}, max_tokens_limit=512),
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain
# Read tu VectorDB
def read_vectors_db():
    # Embeding
    MODEL_NAME = "keepitreal/vietnamese-sbert"
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db
# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)
#Tao Prompt
template = """<|im_start|>system\nbạn là một trợ lí luật sư AI hữu ích. Dựa vào thông tin pháp luật sau đây để trả lời câu hỏi.
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)
llm_chain  = create_qa_chain(prompt, llm, db)

def AskModel(question):
    return llm_chain.invoke({"query": question})
# Chay cai chain
# print(db.similarity_search(question))