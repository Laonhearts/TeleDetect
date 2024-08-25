# 베이스 이미지로 Python 3.8을 사용
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 요구사항 파일 복사 및 설치
COPY setup.py /app/
COPY . /app/
RUN pip install --upgrade pip
RUN pip install .

# 학습 모델 복사 (필요한 경우)
# ADD path_to_your_trained_model.h5 /app/

# 스크립트를 실행할 명령어 (예: main 함수를 실행)
CMD ["deepfake_detector"]
