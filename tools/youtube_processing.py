import os
import tempfile
import cv2
import numpy as np
from typing import List, Dict, Any
import requests
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    pipeline, AutoTokenizer, AutoModelForSequenceClassification
)
import yt_dlp
from smolagents import Tool
import whisper
import subprocess
import time
import random


class YouTubeVideoProcessorTool(Tool):
    name = "youtube_video_processor"
    description = """
    Processes YouTube videos to answer questions about their content, including visual elements, 
    people, conversations, actions, and scenes. Takes a YouTube URL and a question as input.
    """
    inputs = {
        "url": {
            "type": "string", 
            "description": "YouTube video URL to analyze"
        },
        "questions": {
            "type": "string", 
            "description": "Question to answer about the video content"
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._setup_models()
        self._setup_yt_dlp()

    def _setup_models(self):
        """Initialize AI models for video analysis"""
        # Visual question answering model
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
        
        # Audio transcription model
        self.whisper_model = whisper.load_model("base")
        
        # Text analysis pipeline
        self.text_analyzer = pipeline(
            "question-answering", 
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )
    def _get_random_user_agent(self):
        user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        ]   
        return random.choice(user_agents)        

    def _setup_yt_dlp(self):
        """Configure yt-dlp with anti-blocking measures"""
        self.ydl_opts = {
            #'format': 'best[height<=720]',  # Limit quality to avoid large downloads
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'wav',
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # Anti-blocking measures
            'sleep_interval': 2,
            'max_sleep_interval': 3,
            'sleep_interval_requests': 2,
            'sleep_interval_subtitles': 2,
            'extractor_retries': 3,
            'fragment_retries': 3,
            #'http_headers': {
            #    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            # Headers rotation
            'http_headers': {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip,deflate',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                'Connection': 'keep-alive',
            },            
            # Use proxy rotation if available
            'proxy': self._get_random_proxy() if self._has_proxies() else None,
        }

    def _has_proxies(self) -> bool:
        """Check if proxy list is available"""
        proxy_file = os.environ.get('PROXY_LIST_FILE', 'proxies.txt')
        return os.path.exists(proxy_file)

    def _get_random_proxy(self) -> str:
        """Get random proxy from list"""
        try:
            proxy_file = os.environ.get('PROXY_LIST_FILE', 'proxies.txt')
            with open(proxy_file, 'r') as f:
                proxies = f.read().strip().split('\n')
            return random.choice(proxies) if proxies else None
        except:
            return None

    def _download_video(self, url: str, temp_dir: str) -> Dict[str, str]:
        """Download video and audio with anti-blocking measures"""
        video_path = None
        audio_path = None
        
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:                
                # Extract video info first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                
                # Download video
                video_opts = self.ydl_opts.copy()
                video_opts['outtmpl'] = os.path.join(temp_dir, f'{title}_video.%(ext)s')
                print("Video info extracted, starting download...")
                max_retries = 3
                try:
                    for attempt in range(max_retries):
                        print(f"Attempt {attempt + 1} of {max_retries} to download video...")
                        with yt_dlp.YoutubeDL(video_opts) as video_ydl:
                            video_ydl.download([url])
                        # If download is successful, break the loop
                        break
                except Exception as e:                    
                    # If all attempts fail, return None
                    if attempt == max_retries - 1:
                        return {"video": None, "audio": None}
                    
                #with yt_dlp.YoutubeDL(video_opts) as video_ydl:
                #    video_ydl.download([url])
                    
                # Find downloaded video file
                for file in os.listdir(temp_dir):
                    if 'video' in file and any(ext in file for ext in ['.mp4', '.webm', '.mkv']):
                        video_path = os.path.join(temp_dir, file)
                        break
                print(f"Video downloaded: {video_path}")                
                # Extract audio separately
                audio_opts = self.ydl_opts.copy()
                audio_opts.update({
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(temp_dir, f'{title}_audio.%(ext)s')
                })
                print(f"Starting audio extraction...")
                with yt_dlp.YoutubeDL(audio_opts) as audio_ydl:
                    audio_ydl.download([url])
                
                # Find audio file
                for file in os.listdir(temp_dir):
                    if 'audio' in file and file.endswith('.wav'):
                        audio_path = os.path.join(temp_dir, file)
                        break
                print(f"Audio extracted: {audio_path}")
                        
        except Exception as e:
            print(f"Trying fallback download method")
            # Fallback: try alternative extraction method
            return self._fallback_download(url, temp_dir)
            
        return {"video": video_path, "audio": audio_path}

    def _fallback_download(self, url: str, temp_dir: str) -> Dict[str, str]:
        """Fallback download method using different approach"""
        try:
            # Use streamlink as fallback if available
            video_path = os.path.join(temp_dir, "fallback_video.mp4")
            cmd = f'streamlink "{url}" best -o "{video_path}"'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            
            # Extract audio from video
            audio_path = os.path.join(temp_dir, "fallback_audio.wav")
            cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}"'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            
            return {"video": video_path, "audio": audio_path}
        except:
            return {"video": None, "audio": None}

    def _extract_frames(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """Extract key frames from video"""
        frames = []
        if not video_path or not os.path.exists(video_path):
            return frames
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return frames
        
        # Extract frames at regular intervals
        interval = max(1, total_frames // num_frames)
        
        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                if len(frames) >= num_frames:
                    break
        
        cap.release()
        return frames

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper"""
        if not audio_path or not os.path.exists(audio_path):
            return ""
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return ""

    def _analyze_frames_with_question(self, frames: List[np.ndarray], question: str) -> List[str]:
        """Analyze frames using visual question answering"""
        answers = []
        
        for frame in frames:
            try:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Process with BLIP model
                inputs = self.blip_processor(pil_image, question, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.blip_model.generate(**inputs, max_length=50)
                
                answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                if answer and answer.lower() not in ['no', 'none', 'nothing']:
                    answers.append(answer)
                    
            except Exception as e:
                print(f"Frame analysis error: {str(e)}")
                continue
        
        return answers

    def _answer_from_transcript(self, transcript: str, question: str) -> str:
        """Answer question using transcript analysis"""
        if not transcript:
            return ""
        
        try:
            # Split transcript into chunks if too long
            max_length = 512
            chunks = [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]
            
            best_answer = ""
            best_score = 0
            
            for chunk in chunks:
                try:
                    result = self.text_analyzer(question=question, context=chunk)
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_answer = result['answer']
                except:
                    continue
                    
            return best_answer if best_score > 0.1 else ""
            
        except Exception as e:
            print(f"Transcript analysis error: {str(e)}")
            return ""

    def forward(self, url: str, questions: str) -> str:
        """Main processing function"""
        if not url or not questions:
            return "Error: URL and questions are required"
        
        # Validate URL
        if 'youtube.com' not in url and 'youtu.be' not in url:
            return "Error: Invalid YouTube URL"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download video and audio
                print("Downloading video...")
                paths = self._download_video(url, temp_dir)
                
                if not paths["video"] and not paths["audio"]:
                    return "Error: Could not download video. YouTube may be blocking requests or the video is unavailable."
                
                # Extract visual information
                visual_answers = []
                if paths["video"]:
                    print("Processing video frames...")
                    frames = self._extract_frames(paths["video"])
                    if frames:
                        visual_answers = self._analyze_frames_with_question(frames, questions)
                
                # Extract and analyze audio
                transcript = ""
                audio_answer = ""
                if paths["audio"]:
                    print("Transcribing audio...")
                    transcript = self._transcribe_audio(paths["audio"])
                    if transcript:
                        audio_answer = self._answer_from_transcript(transcript, questions)
                
                # Combine results
                result_parts = []
                
                if audio_answer:
                    result_parts.append(f"From transcript: {audio_answer}")
                
                if visual_answers:
                    unique_visual = list(set(visual_answers))
                    result_parts.append(f"From visual analysis: {', '.join(unique_visual[:3])}")
                
                if transcript and not audio_answer:
                    # Include relevant transcript snippet
                    words = transcript.split()
                    if len(words) > 50:
                        transcript_snippet = ' '.join(words[:50]) + "..."
                    else:
                        transcript_snippet = transcript
                    result_parts.append(f"Transcript excerpt: {transcript_snippet}")
                
                if not result_parts:
                    return "Could not extract sufficient information from the video to answer the question."
                
                return "\n\n".join(result_parts)
                
            except Exception as e:
                return f"Error processing video: {str(e)[:200]}"