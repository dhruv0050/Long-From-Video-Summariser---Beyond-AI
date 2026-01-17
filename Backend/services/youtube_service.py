import os
import re
from typing import Optional
import yt_dlp
import config


class YouTubeService:
    """Service for downloading YouTube videos"""
    
    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # If no match, assume the URL is just the video ID
        if len(youtube_url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', youtube_url):
            return youtube_url
        
        raise ValueError(f"Invalid YouTube URL format: {youtube_url}")
    
    @staticmethod
    def get_video_info(video_id: str) -> dict:
        """Get video metadata without downloading"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Untitled Video'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', '')
                }
            except Exception as e:
                raise Exception(f"Failed to fetch video info: {str(e)}")
    
    @staticmethod
    def download_video(
        youtube_url: str,
        output_path: str,
        video_id: Optional[str] = None
    ) -> str:
        """
        Download YouTube video to specified path
        
        Args:
            youtube_url: YouTube video URL
            output_path: Path where video should be saved
            video_id: Optional video ID (if already extracted)
        
        Returns:
            Path to downloaded video file
        """
        if video_id is None:
            video_id = YouTubeService.extract_video_id(youtube_url)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure yt-dlp options for best quality and format
        # Remove extension from output_path for yt-dlp template
        base_output_path = output_path.rsplit('.', 1)[0] if '.' in os.path.basename(output_path) else output_path
        
        # Format preference: avoid HLS (m3u8) which can have fragment issues, prefer progressive mp4
        # Try multiple format strategies with fallbacks
        format_selectors = [
            # First try: Progressive MP4 (single file, most reliable)
            'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            # Second try: Any progressive format (no HLS)
            'best[protocol!=m3u8_native][ext=mp4]/best[protocol!=m3u8_native]',
            # Third try: Best available (including HLS as last resort)
            'best'
        ]
        
        ydl_opts = {
            'format': format_selectors[0],  # Start with best MP4
            'outtmpl': base_output_path + '.%(ext)s',  # yt-dlp will add extension
            'quiet': False,
            'no_warnings': False,
            'progress_hooks': [YouTubeService._progress_hook],
            'no_check_certificate': False,
            'prefer_insecure': False,
            'extractor_args': {
                'youtube': {
                    'skip': ['hls'],  # Try to skip HLS if possible
                    'player_client': ['android', 'web']  # Try android client first (more reliable)
                }
            },
            # Retry on fragment errors
            'fragment_retries': 3,
            'retries': 3,
        }
        
        # If the URL is just a video ID, construct full URL
        if not youtube_url.startswith('http'):
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Try download with multiple format strategies
        last_error = None
        for format_idx, format_selector in enumerate(format_selectors):
            try:
                ydl_opts['format'] = format_selector
                print(f"Attempting download with format selector {format_idx + 1}/{len(format_selectors)}: {format_selector[:50]}...")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                    
                    # yt-dlp may have added an extension, check for the actual file
                    possible_extensions = ['.mp4', '.webm', '.mkv', '.flv', '.m4a']
                    
                    for ext in possible_extensions:
                        potential_path = base_output_path + ext
                        if os.path.exists(potential_path) and os.path.getsize(potential_path) > 0:
                            # Verify file is not empty
                            file_size = os.path.getsize(potential_path)
                            print(f"Downloaded file found: {potential_path} ({file_size} bytes)")
                            
                            # If the extension doesn't match what we want, rename it
                            if potential_path != output_path:
                                os.rename(potential_path, output_path)
                            return output_path
                    
                    # If no file found with extensions, check if output_path already exists
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return output_path
                    
                    # Last resort: check if base_output_path exists
                    if os.path.exists(base_output_path) and os.path.getsize(base_output_path) > 0:
                        os.rename(base_output_path, output_path)
                        return output_path
                    
                    # If we got here, file is empty or doesn't exist
                    if format_idx < len(format_selectors) - 1:
                        print(f"Download attempt {format_idx + 1} failed: file empty or not found. Trying next format...")
                        continue
                    else:
                        raise Exception(f"Downloaded file is empty or not found after all attempts")
                        
            except Exception as e:
                last_error = str(e)
                print(f"Download attempt {format_idx + 1} failed: {last_error}")
                
                # If this was the last attempt, raise the error
                if format_idx == len(format_selectors) - 1:
                    # Clean up any empty files
                    for ext in ['.mp4', '.webm', '.mkv', '.flv', '.m4a']:
                        potential_path = base_output_path + ext
                        if os.path.exists(potential_path) and os.path.getsize(potential_path) == 0:
                            try:
                                os.remove(potential_path)
                            except:
                                pass
                    
                    raise Exception(f"Failed to download YouTube video after {len(format_selectors)} attempts. Last error: {last_error}")
                
                # Continue to next format selector
                continue
        
        # Should not reach here, but just in case
        raise Exception(f"Failed to download YouTube video: {last_error or 'Unknown error'}")
    
    @staticmethod
    def _progress_hook(d):
        """Progress hook for yt-dlp download"""
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            # Print progress (can be improved with logging)
            if percent != 'N/A':
                print(f"Download progress: {percent} at {speed}")
        elif d['status'] == 'finished':
            print(f"Download complete: {d['filename']}")


# Singleton instance
youtube_service = YouTubeService()
