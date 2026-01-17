from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from bson import ObjectId
from datetime import datetime
from typing import List

from models.database import db
from models.video_job import (
    VideoJobCreate,
    YouTubeJobCreate,
    VideoJobResponse, 
    VideoJobResult,
    VideoJob,
    ReportSummary
)
from services.pipeline import pipeline

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/process", response_model=VideoJobResponse)
async def process_video(
    job_request: VideoJobCreate,
    background_tasks: BackgroundTasks
):
    """
    Start processing a video from Google Drive
    
    Args:
        job_request: Video job request with Drive URL
        background_tasks: FastAPI background tasks
    
    Returns:
        Job response with job_id and initial status
    """
    try:
        # Create job in database
        database = db.get_db()
        
        job = VideoJob(
            drive_video_url=job_request.drive_video_url,
            video_name=job_request.video_name,
            status="pending",
            progress=0.0
        )
        
        # Insert into MongoDB
        result = await database.video_jobs.insert_one(job.dict(by_alias=True))
        job_id = str(result.inserted_id)
        
        # Start processing in background
        background_tasks.add_task(pipeline.process_video, job_id)
        
        return VideoJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            video_name=job_request.video_name,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.post("/process-youtube", response_model=VideoJobResponse)
async def process_youtube_video(
    job_request: YouTubeJobCreate,
    background_tasks: BackgroundTasks
):
    """
    Start processing a video from YouTube
    
    Args:
        job_request: YouTube job request with YouTube URL
        background_tasks: FastAPI background tasks
    
    Returns:
        Job response with job_id and initial status
    """
    try:
        # Create job in database
        database = db.get_db()
        
        job = VideoJob(
            youtube_url=job_request.youtube_url,
            video_name=job_request.video_name,
            video_source="youtube",
            status="pending",
            progress=0.0
        )
        
        # Insert into MongoDB
        result = await database.video_jobs.insert_one(job.dict(by_alias=True))
        job_id = str(result.inserted_id)
        
        # Start processing in background
        background_tasks.add_task(pipeline.process_video, job_id)
        
        return VideoJobResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            video_name=job_request.video_name,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create YouTube job: {str(e)}")


@router.get("/status/{job_id}", response_model=VideoJobResponse)
async def get_job_status(job_id: str):
    """
    Get the current status of a processing job
    
    Args:
        job_id: Job ID to check
    
    Returns:
        Job status and progress
    """
    try:
        database = db.get_db()
        
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return VideoJobResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            progress=job.get("progress", 0.0),
            video_name=job.get("video_name"),
            created_at=job.get("created_at", datetime.utcnow())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/results/{job_id}", response_model=VideoJobResult)
async def get_job_results(job_id: str):
    """
    Get the results of a completed job
    
    Args:
        job_id: Job ID to retrieve results for
    
    Returns:
        Complete job results including topics, summary, frames
    """
    try:
        database = db.get_db()
        
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("status") not in ["completed", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Job is still processing (status: {job.get('status')})"
            )
        
        if job.get("status") == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Job failed: {job.get('error_message', 'Unknown error')}"
            )
        
        return VideoJobResult(
            job_id=job_id,
            status=job.get("status"),
            video_name=job.get("video_name"),
            duration=job.get("duration"),
            executive_summary=job.get("executive_summary"),
            topics=job.get("topics", []),
            key_takeaways=job.get("key_takeaways", []),
            entities=job.get("entities", {}),
            total_frames=job.get("total_frames", 0),
            processing_cost=job.get("processing_cost"),
            completed_at=job.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job results: {str(e)}")


@router.get("/list")
async def list_jobs(limit: int = 50, skip: int = 0):
    """
    List all video processing jobs
    
    Args:
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip (for pagination)
    
    Returns:
        List of jobs with basic info
    """
    try:
        database = db.get_db()
        
        cursor = database.video_jobs.find(
            {},
            {"_id": 1, "video_name": 1, "status": 1, "progress": 1, "created_at": 1}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        jobs = await cursor.to_list(length=limit)
        
        return [
            {
                "job_id": str(job["_id"]),
                "video_name": job.get("video_name", "Unknown"),
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", 0.0),
                "created_at": job.get("created_at")
            }
            for job in jobs
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated data
    
    Args:
        job_id: Job ID to delete
    
    Returns:
        Success message
    """
    try:
        database = db.get_db()
        
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # TODO: Delete associated Drive files
        
        # Delete from database
        await database.video_jobs.delete_one({"_id": ObjectId(job_id)})
        
        return {"message": f"Job {job_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.get("/reports", response_model=List[ReportSummary])
async def get_reports(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status: str = Query(None)
):
    """
    Get all past reports with pagination
    
    Args:
        page: Page number (1-indexed)
        limit: Number of results per page
        status: Filter by status (optional)
    
    Returns:
        List of report summaries
    """
    try:
        database = db.get_db()
        
        # Build query
        query = {}
        if status:
            query["status"] = status
        
        # Calculate skip
        skip = (page - 1) * limit
        
        # Get completed jobs sorted by creation date (newest first)
        jobs = await database.video_jobs.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(length=limit)
        
        reports = []
        for job in jobs:
            reports.append(ReportSummary(
                job_id=str(job.get("_id")),
                video_name=job.get("video_name"),
                status=job.get("status"),
                duration=job.get("duration"),
                topics_count=len(job.get("topics", [])),
                created_at=job.get("created_at"),
                completed_at=job.get("completed_at"),
                executive_summary=job.get("executive_summary")
            ))
        
        return reports
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reports: {str(e)}")
