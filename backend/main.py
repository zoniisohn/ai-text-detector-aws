# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

# 1. 자원 관리 최적화 (CPU 환경에서 서버 과부하 방지)
torch.set_num_threads(1)

app = FastAPI()

# 2. CORS 설정 (프론트엔드 연결 대비)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 모델 로드 (절대 경로 혹은 상대 경로 확인)
print("Loading model and tokenizer...")
MODEL_DIR = "./model_files"
device = torch.device("cpu") # AWS/로컬 환경 대응을 위해 CPU로 고정

try:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval() # 추론 모드로 설정 (드롭아웃 등 비활성화)
    print("✅ Model successfully loaded")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# 4. 데이터 규격 정의
class TextInput(BaseModel):
    text: str

# 5. 예측(Predict) 엔드포인트
@app.post("/predict")
def predict(body: TextInput):
    # 입력 검증
    clean_text = body.text.strip()
    if len(clean_text) < 10:
        return {"error": "Entered text is too short. Please enter at least 10 characters."}

    # 토크나이징 (CPU 전용 텐서 생성)
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # 예측 수행 (그래디언트 계산 비활성화로 메모리 절약)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 확률 계산 및 결과 반환
    probs = torch.softmax(logits, dim=-1)[0]
    ai_prob = float(probs[1]) # 보통 1번 인덱스가 AI 생성물

    return {
        "label": "AI generated" if ai_prob > 0.5 else "Human written",
        "ai_probability": round(ai_prob * 100, 1),
        "human_probability": round((1 - ai_prob) * 100, 1),
    }

# 6. 헬스체크 (AWS 로드밸런서/컨테이너 상태 확인용)
@app.get("/health")
def health():
    return {"status": "ok", "model": "RoBERTa-Fake-Detector"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)