FROM python:3.10-slim

# 1. 파이썬 로그 즉시 출력 설정
ENV PYTHONUNBUFFERED True

# 2. 작업 디렉토리 설정
ENV APP_HOME /app
WORKDIR $APP_HOME

# 3. [중요] 시스템 패키지 설치 (패키지명 변경: libgl1-mesa-glx -> libgl1)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 4. 파일 복사
COPY . ./

# 5. 파이썬 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 6. 서버 실행
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
