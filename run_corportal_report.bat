@echo off
cd /d "C:\Users\Roci\northlight"
python backend\jobs\ingest_corportal.py
echo Report extraction completed at %date% %time%