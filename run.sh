#!/usr/bin/env bash
# from repo root
export PYTHONPATH=.

# start backend (uvicorn)
uvicorn backend.app:app --host 0.0.0.0 --port 8000
