#!/bin/bash

echo "🔁 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚂 Training data"
python train_data.py

echo "🚀 Running sentiment analysis"
python sentiment.py

echo "📊 Generating sentiment report"
python sentiment_report.py

echo "✅ All done!"
