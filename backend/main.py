from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import redis
import json
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis 연결 (EC2 #1 프라이빗 IP)
redis_client = redis.Redis(
    host="172.31.47.216",  # EC2 #1 프라이빗 IP
    port=6379,
    decode_responses=True
)

# 모델 로딩
print("모델 로딩 중...")
MODEL_DIR = "./model_files"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print("모델 로딩 완료!")

class TextInput(BaseModel):
    text: str

def get_cache_key(text: str) -> str:
    # 텍스트를 MD5 해시로 변환해서 키로 사용
    return f"predict:{hashlib.md5(text.encode()).hexdigest()}"

@app.post("/predict")
def predict(body: TextInput):
    if len(body.text.strip()) < 10:
        return {"error": "10자 이상 입력해주세요"}

    # 1. 캐시 확인
    cache_key = get_cache_key(body.text)
    cached = redis_client.get(cache_key)
    if cached:
        result = json.loads(cached)
        result["cached"] = True  # 캐시에서 왔다는 표시
        return result

    # 2. 캐시 없으면 AI 모델 실행
    inputs = tokenizer(
        body.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    ai_prob = float(probs[1])

    result = {
        "label": "AI 생성" if ai_prob > 0.5 else "사람 작성",
        "ai_probability": round(ai_prob * 100, 1),
        "human_probability": round((1 - ai_prob) * 100, 1),
        "cached": False  # 모델에서 직접 계산
    }

    # 3. 결과를 Redis에 저장 (1시간 유효)
    redis_client.setex(cache_key, 3600, json.dumps(result))

    return result

@app.get("/health")
def health():
    # Redis 연결 상태도 같이 확인
    try:
        redis_client.ping()
        redis_status = "ok"
    except:
        redis_status = "error"

    return {
        "status": "ok",
        "model": "RoBERTa-Fake-Detector",
        "redis": redis_status
    }