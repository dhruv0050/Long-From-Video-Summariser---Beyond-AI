from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models.database import db
from routes.video_routes import router as video_router
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await db.connect_db()
    print("🚀 Video Intelligence Pipeline API started")
    print(f"📁 Temp directory: {config.TEMP_DIR}")
    print(f"🔑 Using model: {config.MODEL}")
    
    yield
    
    # Shutdown
    await db.close_db()
    print("👋 API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Video Intelligence Pipeline API",
    description="Process long-form videos with AI for intelligent insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Intelligence Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        database = db.get_db()
        await database.command("ping")
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": config.MODEL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to False in production
    )
