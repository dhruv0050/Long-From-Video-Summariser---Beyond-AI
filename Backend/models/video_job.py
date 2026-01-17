from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class Frame(BaseModel):
    timestamp: str
    frame_number: int
    drive_url: Optional[str] = None
    description: Optional[str] = None
    ocr_text: Optional[str] = None
    type: Optional[str] = None  # "slide", "demo", "diagram", etc.


class Topic(BaseModel):
    title: str
    timestamp_range: List[str]  # [start, end] in HH:MM:SS format
    start_seconds: float = 0.0  # For easier sorting
    end_seconds: float = 0.0
    summary: Optional[str] = None
    key_points: List[str] = []
    frames: List[Frame] = []
    quotes: List[str] = []
    visual_cues: List[str] = []


class TranscriptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None


class VideoJobCreate(BaseModel):
    drive_video_url: str
    video_name: Optional[str] = None


class YouTubeJobCreate(BaseModel):
    youtube_url: str
    video_name: Optional[str] = None


class VideoJob(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    drive_video_url: Optional[str] = None  # For Google Drive videos
    youtube_url: Optional[str] = None  # For YouTube videos
    video_source: str = "drive"  # "drive" or "youtube"
    drive_file_id: Optional[str] = None
    youtube_video_id: Optional[str] = None  # YouTube video ID
    video_name: Optional[str] = None
    status: str = "pending"  # pending, downloading, processing, completed, failed
    progress: float = 0.0
    error_message: Optional[str] = None
    
    # Storage paths
    drive_folder_id: Optional[str] = None
    audio_drive_id: Optional[str] = None
    
    # Processing results
    transcript: List[TranscriptSegment] = []
    topics: List[Topic] = []
    frames: List[Frame] = []
    executive_summary: Optional[str] = None
    key_takeaways: List[str] = []
    entities: Dict[str, List[str]] = {}
    
    # Report/Synthesis (stored for easy retrieval)
    report: Dict[str, Any] = {}  # Full synthesis result
    
    # Metadata
    duration: Optional[float] = None
    total_frames: int = 0
    processing_cost: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class VideoJobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    video_name: Optional[str] = None
    created_at: datetime
    
    class Config:
        json_encoders = {ObjectId: str}


class VideoJobResult(BaseModel):
    job_id: str
    status: str
    video_name: Optional[str] = None
    duration: Optional[float] = None
    executive_summary: Optional[str] = None
    topics: List[Topic] = []
    frames: List[Frame] = []
    key_takeaways: List[str] = []
    entities: Dict[str, List[str]] = {}
    total_frames: int = 0
    processing_cost: Optional[float] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {ObjectId: str}


class ReportSummary(BaseModel):
    """Summary of a report for listing"""
    job_id: str
    video_name: Optional[str] = None
    status: str
    duration: Optional[float] = None
    topics_count: int = 0
    created_at: datetime
    completed_at: Optional[datetime] = None
    executive_summary: Optional[str] = None
    
    class Config:
        json_encoders = {ObjectId: str}
