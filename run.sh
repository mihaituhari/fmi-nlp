#!/bin/bash

echo "🔁 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Running clean_data.py..."
python clean_data.py

echo "🚀 Running sentiment.py..."
python sentiment.py

echo "📊 Generating sentiment report..."
python sentiment_report.py

echo "✅ All done!"

