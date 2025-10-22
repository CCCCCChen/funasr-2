#!/bin/bash
export PYTHONUNBUFFERED=1
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
uvicorn app:app --host 0.0.0.0 --port 8220 --reload