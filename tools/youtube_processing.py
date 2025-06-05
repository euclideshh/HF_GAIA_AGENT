import re
from typing import Dict, Any, List
from smolagents import Tool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.text_splitter import RecursiveCharacterTextSplitter

class YouTubeVideoProcessorTool(Tool):
    name = "youtube_video_processor"
    description = "Process YouTube videos to extract transcript content using LangChain."
    inputs = {'url': {'type': 'string', 'description': 'The YouTube video URL to process.'}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Initialize text formatter and splitter
        self.formatter = TextFormatter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def forward(self, url: str) -> str:
        """
        Process a YouTube video URL and return transcript content
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            str: Transcript text content from the video
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(url)
            if not video_id:
                return "Error: Could not extract video ID from URL"
            
            # Get transcript using youtube-transcript-api
            transcript_text = self._get_transcript(video_id)
            
            if not transcript_text:
                return "Error: Could not extract transcript from video. Video may not have captions available or may be private/restricted."
            
            # Format and return transcript
            return self._format_transcript(transcript_text, url, video_id)
                
        except Exception as e:
            return f"Error processing YouTube video: {str(e)}"
    
    def _extract_video_id(self, url: str) -> str:
        """
        Extract video ID from various YouTube URL formats
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            str: Video ID or None if not found
        """
        # Different YouTube URL patterns
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)'
            r'([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
            r'youtu\.be\/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _get_transcript(self, video_id: str) -> str:
        """
        Get transcript from YouTube video using youtube-transcript-api
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            str: Transcript text, or None if failed
        """
        try:
            # Try to get transcript in multiple languages
            # Priority: English, Spanish, then any available language
            languages_to_try = ['en', 'es', 'en-US', 'en-GB', 'es-ES', 'es-MX']
            
            transcript_list = None
            
            # First try with preferred languages
            for lang in languages_to_try:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(
                        video_id, 
                        languages=[lang]
                    )
                    break
                except:
                    continue
            
            # If no preferred language found, try any available language
            if not transcript_list:
                try:
                    # Get list of available transcripts
                    available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    
                    # Try to get any available transcript
                    for transcript in available_transcripts:
                        try:
                            transcript_list = transcript.fetch()
                            break
                        except:
                            continue
                            
                except Exception as e:
                    print(f"Error getting available transcripts: {e}")
            
            if not transcript_list:
                return None
            
            # Format transcript using TextFormatter
            formatted_transcript = self.formatter.format_transcript(transcript_list)
            
            return formatted_transcript.strip()
            
        except Exception as e:
            print(f"Error getting transcript: {e}")
            return None
    
    def _format_transcript(self, transcript: str, url: str, video_id: str) -> str:
        """
        Format the transcript with metadata
        
        Args:
            transcript (str): Raw transcript text
            url (str): Original YouTube URL
            video_id (str): YouTube video ID
            
        Returns:
            str: Formatted transcript with metadata
        """
        # Clean up transcript text
        transcript = self._clean_transcript(transcript)
        
        # Split transcript if it's too long
        if len(transcript) > 8000:
            chunks = self.text_splitter.split_text(transcript)
            transcript = chunks[0] + "\n\n[Note: Transcript truncated for processing. Full content available.]"
        
        formatted_text = f"""
        YouTube Video Content Analysis
        Source URL: {url}
        Video ID: {video_id}
        Content Length: {len(transcript)} characters

        --- TRANSCRIPT ---
        {transcript}
        --- END TRANSCRIPT ---

        This transcript is ready for summarization, analysis, or answering questions about the video content.
        """.strip()
        
        return formatted_text
    
    def _clean_transcript(self, transcript: str) -> str:
        """
        Clean up transcript text by removing extra whitespace and formatting
        
        Args:
            transcript (str): Raw transcript text
            
        Returns:
            str: Cleaned transcript text
        """
        # Remove extra whitespace and newlines
        transcript = re.sub(r'\s+', ' ', transcript)
        
        # Remove common transcript artifacts
        transcript = re.sub(r'\[.*?\]', '', transcript)  # Remove [Music], [Applause], etc.
        transcript = re.sub(r'\(.*?\)', '', transcript)  # Remove (inaudible), etc.
        
        # Clean up punctuation spacing
        transcript = re.sub(r'\s+([,.!?])', r'\1', transcript)
        
        return transcript.strip()