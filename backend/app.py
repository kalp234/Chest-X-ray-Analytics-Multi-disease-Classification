# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.inference_routes import router as inference_router
from backend.routes.explain_routes import router as explain_router

app = FastAPI(title="Chest X-Ray Ensemble API")

# ✅ Allow React app requests
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include both routers
app.include_router(inference_router, prefix="/predict")
app.include_router(explain_router, prefix="/explain")

@app.get("/")
def root():
    return {"message": "✅ Backend is running"}
