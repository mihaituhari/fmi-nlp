#!/bin/bash

echo "🔁 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Running sentiment.py..."
python3 sentiment.py

echo "📊 Generating sentiment report..."
python sentiment_report.py

echo "✅ All done!"

