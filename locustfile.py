from locust import HttpUser, task, between
import random

# 테스트용 텍스트 목록 (다양한 텍스트로 캐시 미스 유도)
AI_TEXTS = [
    "The implementation of advanced machine learning algorithms has significantly enhanced the capability of autonomous systems to process large datasets.",
    "Artificial intelligence represents a paradigm shift in computational methodology, enabling unprecedented analytical capabilities across diverse domains.",
    "The deployment of neural network architectures has demonstrated remarkable performance improvements in natural language processing tasks.",
    "Machine learning models have achieved superhuman performance on various benchmarks through sophisticated training methodologies.",
    "The integration of deep learning frameworks has revolutionized computer vision applications in medical imaging.",
]

HUMAN_TEXTS = [
    "hey so i was thinking about grabbing lunch later, you down?",
    "omg did you see what happened yesterday lol that was crazy",
    "ugh my wifi keeps cutting out its so annoying rn",
    "can't believe how long that meeting went, total waste of time tbh",
    "just got back from the gym, absolutely destroyed today",
]

ALL_TEXTS = AI_TEXTS + HUMAN_TEXTS

class AIDetectorUser(HttpUser):
    # 요청 사이 대기시간 (1~3초)
    wait_time = between(1, 3)

    @task(3)  # 가중치 3 = predict가 health보다 3배 많이 호출
    def predict(self):
        text = random.choice(ALL_TEXTS)
        self.client.post(
            "/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")