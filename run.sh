#!/bin/bash

# MKE Turnover Predictor - Launch Script

echo "🚀 MKE Turnover Predictor Projesi Başlatılıyor..."

# 1. API'nin arka planda başlatılması
echo "🔌 FastAPI API Sunucusu Başlatılıyor (Port 8000)..."
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# API'nin ayağa kalkması için kısa bir süre bekleyelim
sleep 3

# 2. Streamlit arayüzünün başlatılması
echo "🖥️ Streamlit Arayüzü Başlatılıyor..."
python3 -m streamlit run app/dashboard.py

# Arayüz kapatıldığında arka plandaki API sunucusunu da kapatıyoruz
cleanup() {
    echo "⏱️ Sunucular kapatılıyor..."
    kill $API_PID
    exit
}

# Ctrl+C veya çıkış durumlarında cleanup fonksiyonunu tetikle
trap cleanup EXIT INT TERM
