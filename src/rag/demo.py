import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
#===========================#
# 載入並處理文本資料
#===========================#
def load_and_split_documents(filepath: str, chunk_size=500, chunk_overlap=50):
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

#===========================#
# 建立向量資料庫
#===========================#
def create_vector_db(splits, db_model_path):
    embedding = HuggingFaceEmbeddings(
        model_name=db_model_path,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    return FAISS.from_documents(splits, embedding)

#===========================#
# 查詢相似內容
#===========================#
def retrieve_context(db, query, k=1):
    docs = db.similarity_search(query, k=k)
    for i, doc in enumerate(docs):
        print(f"[匹配段落 {i+1}]\n{doc.page_content}\n{'='*30}")
    return "\n".join([doc.page_content for doc in docs])

#===========================#
# 建構提示語 (Prompt)
#===========================#
def build_prompt(context: str, query: str):
    return f"""
請用條列式回答，並確保語句完整。

### 內容:
{context}
### 問題:
{query}
### LLM的回應:"""

#===========================#
# 載入本地量化模型
#===========================#
def load_local_model(model_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

#===========================#
# 執行推理並輸出
#===========================#
def generate_answer(tokenizer, model, prompt: str, max_new_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
        num_return_sequences=1
    )

    for i, output in enumerate(outputs):
        generated_tokens = output[input_len:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"[回答 {i+1}]\n{answer}\n{'-'*40}")

#===========================#
# 主執行流程
#===========================#
def run_rag_pipeline(model_path: str, db_model_path: str, data_path: str, user_query: str):
    tokenizer, model = load_local_model(model_path)  # 載入本地量化模型
    splits = load_and_split_documents(data_path)  # 載入並處理文本資料
    db = create_vector_db(splits, db_model_path) # 建立向量資料庫
    context = retrieve_context(db, user_query)  # 查詢相似內容
    # context = ""  # 不使用RAG
    prompt = build_prompt(context, user_query)  # 建構提示語 (Prompt)
    generate_answer(tokenizer, model, prompt)  # 執行推理並輸出

#===========================#
# 執行範例
#===========================#
if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    model_path = os.path.join(project_root, "models", "taide--Llama-3.1-TAIDE-LX-8B-Chat")
    db_model_path = os.path.join(project_root, "models", "shibing624--text2vec-base-chinese")

    # data_path = os.path.join(project_root, "data", "merged_en.txt")
    data_path = os.path.join(project_root, "data", "merged_ch.txt")
    question = "推薦我一台外送無人機"

    # data_path = os.path.join(project_root, "data", "demo.txt")
    # question = "2025空難事件"

    print(f"📌 問題：{ question}\n")
    run_rag_pipeline(model_path, db_model_path, data_path, question)
