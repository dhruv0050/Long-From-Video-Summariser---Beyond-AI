import os
import json
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from PIL import Image
import config
from models.video_job import TranscriptSegment, Topic, Frame

# Configure Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

def retry_with_backoff(func, max_retries=3, initial_delay=2):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            print(f"Retry {attempt + 1}/{max_retries} after {delay}s due to: {str(e)[:100]}")
            time.sleep(delay)


class GeminiService:
    def __init__(self):
        self.model_name = config.MODEL
        self.text_model = genai.GenerativeModel(self.model_name)
        # Use gemini-2.5-flash for Vision (higher quota limits)
        self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
    
    async def transcribe_audio(
        self, 
        audio_path: str,
        start_time: float = 0
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio using Gemini with retry logic
        
        Args:
            audio_path: Path to audio file
            start_time: Start time offset for this chunk
        
        Returns:
            List of transcript segments
        """
        def _transcribe():
            # Upload audio file
            print(f"Uploading audio chunk starting at {start_time}s...")
            
            # Use the correct API for google-generativeai >= 0.4.0
            with open(audio_path, 'rb') as audio_file_obj:
                audio_file = genai.upload_file(
                    path=audio_path,
                    mime_type="audio/wav"
                )
            
            prompt = """
            Transcribe this audio with speaker diarization. 
            Label speakers as Speaker A, Speaker B, etc.
            
            Return the transcription in the following JSON format:
            {
                "segments": [
                    {
                        "text": "transcribed text",
                        "start_time": 0.0,
                        "end_time": 5.2,
                        "speaker": "Speaker A"
                    }
                ]
            }
            
            Provide accurate timestamps in seconds relative to the start of this audio clip.
            """
            
            print("Sending to Gemini for transcription...")
            response = self.text_model.generate_content([prompt, audio_file])
            
            # Parse response
            result = self._parse_json_response(response.text)
            
            segments = []
            if result and "segments" in result:
                for seg in result["segments"]:
                    segments.append(TranscriptSegment(
                        text=seg.get("text", ""),
                        start_time=start_time + seg.get("start_time", 0),
                        end_time=start_time + seg.get("end_time", 0),
                        speaker=seg.get("speaker"),
                        confidence=seg.get("confidence", 0.9)
                    ))
            
            return segments
        
        try:
            return retry_with_backoff(_transcribe, max_retries=3)
        except Exception as e:
            print(f"Error transcribing audio after retries: {e}")
            # Fallback: simple transcription without timestamps
            return await self._simple_transcribe(audio_path, start_time)
    
    async def _simple_transcribe(
        self, 
        audio_path: str, 
        start_time: float
    ) -> List[TranscriptSegment]:
        """Fallback simple transcription"""
        try:
            audio_file = genai.upload_file(audio_path)
            prompt = "Transcribe this audio verbatim. Identify different speakers if possible."
            
            response = self.text_model.generate_content([prompt, audio_file])
            
            return [TranscriptSegment(
                text=response.text,
                start_time=start_time,
                end_time=start_time + 300,  # Approximate
                speaker="Speaker A",
                confidence=0.8
            )]
        except Exception as e:
            print(f"Error in simple transcription: {e}")
            return []
    
    async def analyze_transcript(
        self, 
        transcript_text: str,
        duration: float
    ) -> Dict[str, Any]:
        """
        Analyze transcript to extract topics, key moments, etc. with retry logic
        Split large transcripts into chunks to avoid timeouts
        
        Args:
            transcript_text: Full transcript text
            duration: Video duration in seconds
        
        Returns:
            Dictionary with topics, key moments, entities
        """
        # If transcript is very large (>30 min video), analyze in chunks
        max_chars = 50000  # ~12k tokens
        
        if len(transcript_text) > max_chars:
            print(f"Large transcript ({len(transcript_text)} chars), analyzing in chunks...")
            # Split into manageable chunks
            words = transcript_text.split()
            chunk_size = len(words) // 3  # Split into ~3 chunks
            chunks = [
                ' '.join(words[i:i+chunk_size]) 
                for i in range(0, len(words), chunk_size)
            ]
            
            # Analyze each chunk
            all_topics = []
            all_entities = {"people": [], "companies": [], "concepts": [], "tools": []}
            all_takeaways = []
            
            for idx, chunk in enumerate(chunks):
                print(f"Analyzing chunk {idx+1}/{len(chunks)}...")
                result = await self._analyze_transcript_chunk(chunk, duration, idx, len(chunks))
                if result:
                    all_topics.extend(result.get("topics", []))
                    # Merge entities
                    for key in all_entities:
                        all_entities[key].extend(result.get("entities", {}).get(key, []))
                    all_takeaways.extend(result.get("key_takeaways", []))
            
            # Deduplicate entities
            for key in all_entities:
                all_entities[key] = list(set(all_entities[key]))
            
            return {
                "topics": all_topics,
                "entities": all_entities,
                "key_takeaways": list(set(all_takeaways))
            }
        else:
            return await self._analyze_transcript_chunk(transcript_text, duration, 0, 1)
    
    async def _analyze_transcript_chunk(
        self,
        transcript_text: str,
        duration: float,
        chunk_idx: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """Analyze a single transcript chunk"""
        def _analyze():
            chunk_info = f" (part {chunk_idx+1}/{total_chunks})" if total_chunks > 1 else ""
            prompt = f"""
        Analyze this video transcript{chunk_info} (total duration: {duration/60:.1f} minutes) and extract:
        
        1. Topic segmentation: Break into logical chapters with start/end timestamps
        2. Key moments: Important phrases that likely reference visuals ("as shown", "this slide", etc.)
        3. Named entities: People, companies, tools, concepts mentioned
        4. Key takeaways: Main insights from the content
        
        Transcript:
        {transcript_text}
        
        Return analysis in this JSON format:
        {{
            "topics": [
                {{
                    "title": "Topic title",
                    "timestamp_range": ["00:00:00", "00:15:30"],
                    "summary": "Brief summary",
                    "key_points": ["point 1", "point 2"]
                }}
            ],
            "visual_cues": [
                {{
                    "timestamp": "00:05:23",
                    "cue_text": "as you can see on this slide",
                    "context": "surrounding context"
                }}
            ],
            "entities": {{
                "people": ["name1", "name2"],
                "companies": ["company1"],
                "concepts": ["concept1", "concept2"],
                "tools": ["tool1"]
            }},
            "key_takeaways": ["takeaway 1", "takeaway 2"]
        }}
        """
            
            print("Analyzing transcript with Gemini...")
            response = self.text_model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            return result or {}
        
        try:
            return retry_with_backoff(_analyze, max_retries=3)
        except Exception as e:
            print(f"Error analyzing transcript after retries: {e}")
            return {}
    
    async def analyze_frames(
        self, 
        frame_paths: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple frames with Gemini Vision
        
        Args:
            frame_paths: List of paths to image files
            context: Optional context about these frames
        
        Returns:
            List of frame analyses
        """
        results = []
        
        # Process in smaller batches of 2 for better reliability
        batch_size = 2
        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i:i+batch_size]
            
            try:
                batch_result = await self._analyze_frame_batch(batch, context)
                results.extend(batch_result)
            except Exception as e:
                print(f"Error analyzing frame batch at index {i}: {str(e)[:100]}")
                # Add placeholder results so job completes
                for path in batch:
                    results.append({
                        "frame_path": path,
                        "description": "Analysis failed but processing continued",
                        "ocr_text": "",
                        "type": "unknown",
                        "insights": ""
                    })
        
        return results
    
    async def _analyze_frame_batch(
        self,
        frame_paths: List[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of frames together with retry logic"""
        def _analyze():
            images = []
            for path in frame_paths:
                img = Image.open(path)
                images.append(img)
            
            context_text = f"\nContext: {context}" if context else ""
            
            prompt = f"""
        Analyze these video frames and for each frame provide:
        1. Semantic description (what's shown - slides, diagrams, people, demos, etc.)
        2. OCR: Extract all visible text
        3. Type: Classify as "slide", "diagram", "chart", "demo", "person", "other"
        4. Key insights: What information does this frame convey?
        
        {context_text}
        
        Return analysis in this JSON format (avoid trailing commas):
        {{
            "frames": [
                {{
                    "frame_index": 0,
                    "description": "Slide showing framework diagram",
                    "ocr_text": "extracted text here",
                    "type": "slide",
                    "insights": "Key concepts being presented"
                }}
            ]
        }}
        """
            
            content = [prompt] + images
            print(f"Analyzing batch of {len(images)} frames...")
            response = self.vision_model.generate_content(content)
            
            result = self._parse_json_response(response.text)
            
            analyses = []
            if result and "frames" in result:
                for i, frame_data in enumerate(result["frames"]):
                    if i < len(frame_paths):
                        analyses.append({
                            "frame_path": frame_paths[i],
                            "description": frame_data.get("description", ""),
                            "ocr_text": frame_data.get("ocr_text", ""),
                            "type": frame_data.get("type", "other"),
                            "insights": frame_data.get("insights", "")
                        })
            
            return analyses
        
        try:
            return retry_with_backoff(_analyze, max_retries=3)
        except Exception as e:
            print(f"Error analyzing frame batch after retries: {e}")
            # Return placeholder results
            return [{
                "frame_path": path,
                "description": "Analysis failed",
                "ocr_text": "",
                "type": "unknown",
                "insights": ""
            } for path in frame_paths]
    
    async def synthesize_results(
        self,
        transcript_analysis: Dict[str, Any],
        frame_analyses: List[Dict[str, Any]],
        duration: float
    ) -> Dict[str, Any]:
        """
        Synthesize transcript and frame analyses into final output with retry logic
        
        Args:
            transcript_analysis: Analysis from transcript
            frame_analyses: Analyses from frames
            duration: Video duration
        
        Returns:
            Structured output with topics, summary, etc.
        """
        def _synthesize():
            # Create a compact version for synthesis
            topics_preview = json.dumps(transcript_analysis.get("topics", [])[:5])[:2000]
            frames_preview = json.dumps(frame_analyses[:10])[:2000]
            
            prompt = f"""
        You are synthesizing analysis of a {duration/60:.1f}-minute video.
        
        Transcript Topics (excerpt):
        {topics_preview}
        
        Key Frames (excerpt, {len(frame_analyses)} total):
        {frames_preview}
        
        Create a comprehensive synthesis that:
        1. Generates an executive summary (3-5 sentences)
        2. Provides topic-wise breakdown
        3. Extracts actionable insights and key takeaways
        4. Lists entities mentioned (companies, concepts, tools)
        
        Return ONLY valid JSON (no trailing commas or newlines in strings):
        {{
            "executive_summary": "Clear summary...",
            "topics": [
                {{
                    "title": "Topic title",
                    "timestamp_range": ["00:00:00", "00:15:30"],
                    "summary": "Single line summary"
                }}
            ],
            "key_takeaways": ["takeaway 1", "takeaway 2"],
            "entities": {{
                "companies": ["name1"],
                "concepts": ["concept1"],
                "tools": ["tool1"]
            }}
        }}
        """
            
            print("Synthesizing results with Gemini...")
            response = self.text_model.generate_content(prompt)
            result = self._parse_json_response(response.text)
            
            return result or {
                "executive_summary": "Video processing completed but synthesis failed.",
                "topics": transcript_analysis.get("topics", []),
                "key_takeaways": transcript_analysis.get("key_takeaways", []),
                "entities": transcript_analysis.get("entities", {})
            }
        
        try:
            return retry_with_backoff(_synthesize, max_retries=2)
        except Exception as e:
            print(f"Error synthesizing results: {e}")
            # Return fallback with raw analysis
            return {
                "executive_summary": "Video processing completed but synthesis had errors.",
                "topics": transcript_analysis.get("topics", []),
                "key_takeaways": transcript_analysis.get("key_takeaways", []),
                "entities": transcript_analysis.get("entities", {})
            }
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from model response with aggressive error recovery"""
        import re
        
        def clean_json(s: str) -> str:
            """Clean common JSON formatting issues"""
            # Remove leading/trailing whitespace
            s = s.strip()
            
            # Remove comments (not standard JSON but LLMs sometimes add them)
            s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
            s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
            
            # Remove trailing commas before closing braces/brackets
            s = re.sub(r',\s*}', '}', s)
            s = re.sub(r',\s*]', ']', s)
            
            # Handle multiple spaces in strings
            s = re.sub(r' {2,}', ' ', s)
            
            # Escape unescaped newlines and tabs within string values
            lines = []
            in_string = False
            escaped = False
            i = 0
            while i < len(s):
                char = s[i]
                
                if char == '"' and not escaped:
                    in_string = not in_string
                    lines.append(char)
                elif char == '\\' and in_string:
                    escaped = not escaped
                    lines.append(char)
                elif in_string and not escaped:
                    # Handle special characters in strings
                    if char == '\n':
                        lines.append(' ')
                    elif char == '\r':
                        if i + 1 < len(s) and s[i+1] == '\n':
                            i += 1
                        lines.append(' ')
                    elif char == '\t':
                        lines.append(' ')
                    else:
                        lines.append(char)
                    escaped = False
                else:
                    lines.append(char)
                    escaped = False
                
                i += 1
            
            return ''.join(lines)
        
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            # Extract JSON from markdown code block
            json_text = text
            
            # Try multiple extraction patterns
            patterns = [
                (r'```json\s*(.*?)\s*```', re.DOTALL),
                (r'```\s*(.*?)\s*```', re.DOTALL),
                (r'\{.*\}', 0),  # Try to find first {...} block
                (r'\[.*\]', 0),  # Try to find first [...] array
            ]
            
            for pattern, flags in patterns:
                try:
                    match = re.search(pattern, text, flags) if flags else re.search(pattern, text)
                    if match:
                        json_text = match.group(1) if '(' in pattern else match.group(0)
                        break
                except:
                    continue
            
            # Try cleaning and parsing
            try:
                cleaned = clean_json(json_text)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Last resort: try to find valid JSON subset
                try:
                    # Find first '{' and match to '}'
                    start = cleaned.find('{')
                    if start >= 0:
                        depth = 0
                        for i, char in enumerate(cleaned[start:], start):
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    subset = cleaned[start:i+1]
                                    return json.loads(subset)
                except:
                    pass
                
                print(f"Failed to parse JSON after all attempts")
                print(f"Response preview (first 400 chars): {text[:400]}")
                return None


# Singleton instance
gemini_service = GeminiService()
