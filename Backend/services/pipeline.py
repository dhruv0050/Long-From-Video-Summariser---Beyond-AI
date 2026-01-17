import os
import asyncio
from datetime import datetime
from typing import Dict, Any
from bson import ObjectId

from models.database import db
from models.video_job import VideoJob, TranscriptSegment, Topic, Frame
from services.drive_service import drive_service
from services.youtube_service import youtube_service
from services.gemini_service import gemini_service
from utils.ffmpeg_utils import FFmpegUtils
import config


class ProcessingPipeline:
    """Orchestrates the video processing pipeline"""
    
    def __init__(self):
        self.ffmpeg = FFmpegUtils()
    
    async def process_video(self, job_id: str):
        """
        Main processing pipeline
        
        Steps:
        1. Download video from Drive
        2. Extract audio and frames
        3. Transcribe audio (chunked)
        4. Analyze transcript
        5. Analyze frames
        6. Synthesize results
        7. Store in MongoDB and Drive
        """
        try:
            # Get job from database
            job = await self._get_job(job_id)
            
            # Determine video source
            video_source = job.get("video_source", "drive")
            if not video_source and job.get("youtube_url"):
                video_source = "youtube"
            elif not video_source and job.get("drive_video_url"):
                video_source = "drive"
            
            # Update status
            await self._update_job(job_id, {
                "status": "downloading",
                "progress": 0.05,
                "video_source": video_source
            })
            
            # Step 1: Download video based on source
            video_path = await self._download_video(job, video_source)
            await self._update_job(job_id, {"progress": 0.1})
            
            # Get video metadata
            duration = self.ffmpeg.get_video_duration(video_path)
            await self._update_job(job_id, {"duration": duration})
            
            # Step 2: Extract audio
            await self._update_job(job_id, {
                "status": "extracting",
                "progress": 0.15
            })
            audio_path = await self._extract_audio(video_path, job_id)
            await self._update_job(job_id, {"progress": 0.25})
            
            # Step 3: Transcribe audio
            await self._update_job(job_id, {
                "status": "transcribing",
                "progress": 0.3
            })
            transcript = await self._transcribe_audio(audio_path)
            await self._update_job(job_id, {
                "transcript": [seg.model_dump() for seg in transcript],
                "progress": 0.5
            })
            
            # Step 4: Analyze transcript
            await self._update_job(job_id, {
                "status": "analyzing",
                "progress": 0.55
            })
            
            # Build transcript with timestamp context for better analysis
            transcript_segments_with_time = []
            for seg in transcript:
                start_ts = self.ffmpeg.format_timestamp(seg.start_time)
                end_ts = self.ffmpeg.format_timestamp(seg.end_time)
                transcript_segments_with_time.append(f"[{start_ts}-{end_ts}] {seg.text}")
            
            transcript_text = " ".join([seg.text for seg in transcript])
            
            # Log transcript coverage for debugging
            if transcript:
                first_seg_time = transcript[0].start_time
                last_seg_time = max(seg.end_time for seg in transcript)
                print(f"Transcript covers: {self.ffmpeg.format_timestamp(first_seg_time)} to {self.ffmpeg.format_timestamp(last_seg_time)} (video duration: {self.ffmpeg.format_timestamp(duration)})")
                print(f"Total transcript segments: {len(transcript)}, Total characters: {len(transcript_text)}")
            
            transcript_analysis = await gemini_service.analyze_transcript(
                transcript_text, 
                duration
            )
            await self._update_job(job_id, {"progress": 0.6})
            
            # Step 5: Extract frames
            await self._update_job(job_id, {"progress": 0.65})
            frames = await self._extract_frames(video_path, job_id, transcript_analysis)
            await self._update_job(job_id, {
                "total_frames": len(frames),
                "progress": 0.7
            })
            
            # Step 6: Analyze frames
            await self._update_job(job_id, {"progress": 0.75})
            frame_analyses = await self._analyze_frames(frames, job_id, transcript_analysis)
            await self._update_job(job_id, {"progress": 0.85})
            
            # Step 7: Synthesize results
            await self._update_job(job_id, {
                "status": "synthesizing",
                "progress": 0.9
            })
            synthesis = await gemini_service.synthesize_results(
                transcript_analysis,
                frame_analyses,
                duration
            )
            
            # Step 8: Build final output
            topics = await self._build_topics(
                synthesis.get("topics", []),
                frame_analyses,
                transcript
            )
            
            # Step 9: Update job with final results
            # Convert to dict - topics are Pydantic models, frames are already dicts
            topics_data = [topic.model_dump() for topic in topics]
            frames_data = frame_analyses if frame_analyses else []
            
            await self._update_job(job_id, {
                "status": "completed",
                "progress": 1.0,
                "topics": topics_data,
                "frames": frames_data,
                "executive_summary": synthesis.get("executive_summary", ""),
                "key_takeaways": synthesis.get("key_takeaways", []),
                "entities": synthesis.get("entities", {}),
                "report": synthesis,  # Store full synthesis result
                "completed_at": datetime.utcnow()
            })
            
            # Cleanup temp files
            await self._cleanup(job_id)
            
            print(f"Job {job_id} completed successfully")
            
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            await self._update_job(job_id, {
                "status": "failed",
                "error_message": str(e)
            })
    
    async def _download_video(self, job: Dict, video_source: str = "drive") -> str:
        """Download video from Google Drive or YouTube"""
        video_path = os.path.join(config.TEMP_DIR, f"{job['_id']}_video.mp4")
        
        if video_source == "youtube" or job.get("youtube_url"):
            # Download from YouTube
            youtube_url = job.get("youtube_url")
            if not youtube_url:
                raise Exception("YouTube URL not found in job")
            
            video_id = youtube_service.extract_video_id(youtube_url)
            
            # Get video metadata
            try:
                video_info = youtube_service.get_video_info(video_id)
                video_name = video_info.get("title", f"video_{job['_id']}.mp4")
                duration = video_info.get("duration", 0)
            except Exception as e:
                print(f"Warning: Could not fetch YouTube metadata: {e}")
                video_name = job.get("video_name") or f"video_{job['_id']}.mp4"
                duration = None
            
            # Update job with video info
            await self._update_job(str(job["_id"]), {
                "youtube_video_id": video_id,
                "video_name": video_name,
                "video_source": "youtube"
            })
            
            # Download video
            youtube_service.download_video(youtube_url, video_path, video_id)
            
        else:
            # Download from Google Drive (existing logic)
            drive_url = job.get("drive_video_url")
            if not drive_url:
                raise Exception("Drive video URL not found in job")
            
            file_id = drive_service.extract_file_id(drive_url)
            
            # Get file metadata
            metadata = drive_service.get_file_metadata(file_id)
            video_name = metadata.get("name", f"video_{job['_id']}.mp4")
            
            # Update job with file info
            await self._update_job(str(job["_id"]), {
                "drive_file_id": file_id,
                "video_name": video_name,
                "video_source": "drive"
            })
            
            # Download to temp directory
            drive_service.download_file(file_id, video_path)
        
        return video_path
    
    async def _extract_audio(self, video_path: str, job_id: str) -> str:
        """Extract audio from video"""
        audio_path = os.path.join(config.TEMP_DIR, f"{job_id}_audio.wav")
        self.ffmpeg.extract_audio(video_path, audio_path)
        return audio_path
    
    async def _transcribe_audio(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe audio in chunks"""
        # Split audio into chunks
        chunks = self.ffmpeg.split_audio(audio_path)
        
        all_segments = []
        
        for chunk_path, start_time, end_time in chunks:
            try:
                segments = await gemini_service.transcribe_audio(
                    chunk_path, 
                    start_time
                )
                all_segments.extend(segments)
            except Exception as e:
                print(f"Error transcribing chunk {chunk_path}: {e}")
            finally:
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        # Deduplicate overlapping segments
        deduplicated = self._deduplicate_segments(all_segments)
        
        return deduplicated
    
    def _deduplicate_segments(
        self, 
        segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Remove duplicate segments from overlapping chunks"""
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        deduplicated = [sorted_segments[0]]
        
        for seg in sorted_segments[1:]:
            last_seg = deduplicated[-1]
            
            # If this segment overlaps significantly (>70% overlap), merge or skip
            overlap_start = max(seg.start_time, last_seg.start_time)
            overlap_end = min(seg.end_time, last_seg.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            last_duration = last_seg.end_time - last_seg.start_time
            seg_duration = seg.end_time - seg.start_time
            
            # If high overlap (>70% of either segment), prefer the one with more text or merge them
            if last_duration > 0 and overlap_duration / last_duration > 0.7:
                # Merge: keep the longer segment or the one with more text
                if len(seg.text) > len(last_seg.text) or seg_duration > last_duration:
                    deduplicated[-1] = seg
                continue
            elif seg_duration > 0 and overlap_duration / seg_duration > 0.7:
                # Same check from other direction
                if len(seg.text) > len(last_seg.text) or seg_duration > last_duration:
                    deduplicated[-1] = seg
                continue
            
            # If segments are very close (< 2 seconds gap), merge them
            gap = seg.start_time - last_seg.end_time
            if gap < 2 and gap > -2:  # Very close segments, merge
                # Merge text and extend end time
                merged_text = last_seg.text + " " + seg.text
                deduplicated[-1] = TranscriptSegment(
                    text=merged_text,
                    start_time=last_seg.start_time,
                    end_time=max(last_seg.end_time, seg.end_time),
                    speaker=last_seg.speaker or seg.speaker
                )
                continue
            
            deduplicated.append(seg)
        
        return deduplicated
    
    async def _extract_frames(
        self,
        video_path: str,
        job_id: str,
        transcript_analysis: Dict
    ) -> list[tuple[str, float]]:
        """Extract keyframes from video"""
        frames_dir = os.path.join(config.TEMP_DIR, f"{job_id}_frames")
        
        # Extract frames at regular intervals
        frames = self.ffmpeg.extract_keyframes(
            video_path,
            frames_dir,
            interval=config.KEYFRAME_INTERVAL
        )
        
        # TODO: In Phase 2, add smart frame selection based on visual_cues
        # from transcript_analysis
        
        return frames
    
    async def _analyze_frames(
        self,
        frames: list[tuple[str, float]],
        job_id: str,
        transcript_analysis: Dict
    ) -> list[Dict]:
        """Analyze frames and upload to Drive"""
        # Create folder in Drive for this job
        folder_name = f"video_{job_id}_frames"
        folder_id = drive_service.create_folder(
            folder_name,
            parent_folder_id=config.DRIVE_FOLDER_ID
        )
        
        await self._update_job(job_id, {"drive_folder_id": folder_id})
        
        # Get frame paths
        frame_paths = [path for path, _ in frames]
        
        # Analyze frames with Gemini Vision
        analyses = await gemini_service.analyze_frames(frame_paths)
        
        # Upload frames to Drive and add URLs
        for i, (analysis, (frame_path, timestamp)) in enumerate(zip(analyses, frames)):
            try:
                # Upload to Drive
                uploaded = drive_service.upload_file(
                    frame_path,
                    folder_id=folder_id,
                    file_name=f"frame_{i:04d}_{int(timestamp)}s.jpg"
                )
                
                # Add Drive URL to analysis
                analysis["drive_url"] = uploaded.get("webViewLink")
                analysis["timestamp"] = timestamp
                analysis["timestamp_str"] = self.ffmpeg.format_timestamp(timestamp)
                
            except Exception as e:
                print(f"Error uploading frame {frame_path}: {e}")
        
        return analyses
    
    async def _build_topics(
        self,
        topic_data: list[Dict],
        frame_analyses: list[Dict],
        transcript: list[TranscriptSegment]
    ) -> list[Topic]:
        """Build Topic objects with frames"""
        from services.gemini_service import timestamp_to_seconds, seconds_to_timestamp
        
        topics = []
        
        for topic_info in topic_data:
            # Parse timestamp range - ensure we have valid timestamps
            ts_range = topic_info.get("timestamp_range", ["00:00:00", "00:00:00"])
            
            # Convert to seconds and back to ensure consistency
            if ts_range and len(ts_range) >= 2:
                start_seconds = timestamp_to_seconds(ts_range[0])
                end_seconds = timestamp_to_seconds(ts_range[1])
            else:
                start_seconds = 0.0
                end_seconds = 0.0
            
            # Ensure timestamps are valid
            if start_seconds < 0:
                start_seconds = 0.0
            if end_seconds < start_seconds:
                end_seconds = start_seconds + 600  # Default 10 min if end is before start
            
            # Find frames within this topic's time range
            topic_frames = []
            for analysis in frame_analyses:
                frame_ts = analysis.get("timestamp", 0)
                if isinstance(frame_ts, str):
                    frame_ts = timestamp_to_seconds(frame_ts)
                    
                if start_seconds <= frame_ts <= end_seconds:
                    frame = Frame(
                        timestamp=seconds_to_timestamp(frame_ts),
                        frame_number=len(topic_frames),
                        drive_url=analysis.get("drive_url"),
                        description=analysis.get("description"),
                        ocr_text=analysis.get("ocr_text"),
                        type=analysis.get("type", "other")
                    )
                    topic_frames.append(frame)
            
            topic = Topic(
                title=topic_info.get("title", "Untitled"),
                timestamp_range=[seconds_to_timestamp(start_seconds), seconds_to_timestamp(end_seconds)],
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                summary=topic_info.get("summary", ""),
                key_points=topic_info.get("key_points", []),
                frames=topic_frames,
                quotes=topic_info.get("quotes", []),
                visual_cues=topic_info.get("visual_cues", [])
            )
            topics.append(topic)
        
        return topics
    
    def _parse_timestamp(self, ts: str) -> float:
        """Convert HH:MM:SS to seconds"""
        from services.gemini_service import timestamp_to_seconds
        return timestamp_to_seconds(ts)
    
    async def _cleanup(self, job_id: str):
        """Clean up temporary files"""
        patterns = [
            f"{job_id}_video.mp4",
            f"{job_id}_audio.wav"
        ]
        
        for pattern in patterns:
            path = os.path.join(config.TEMP_DIR, pattern)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing {path}: {e}")
        
        # Remove frames directory
        frames_dir = os.path.join(config.TEMP_DIR, f"{job_id}_frames")
        if os.path.exists(frames_dir):
            try:
                import shutil
                shutil.rmtree(frames_dir)
            except Exception as e:
                print(f"Error removing frames directory: {e}")
    
    async def _get_job(self, job_id: str) -> Dict:
        """Get job from database"""
        database = db.get_db()
        job = await database.video_jobs.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise Exception(f"Job {job_id} not found")
        return job
    
    async def _update_job(self, job_id: str, updates: Dict):
        """Update job in database"""
        database = db.get_db()
        updates["updated_at"] = datetime.utcnow()
        await database.video_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": updates}
        )


# Singleton instance
pipeline = ProcessingPipeline()
