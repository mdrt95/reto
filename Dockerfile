FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# The container sits behind exactly one trusted platform proxy (Render); trusting
# X-Forwarded-For from any other source would let clients spoof their own IP.
ENV FORWARDED_ALLOW_IPS="*"

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-*}\""]
