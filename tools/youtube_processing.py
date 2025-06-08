import os
import re
import tempfile
import logging
from typing import Dict, Any, Optional, List
import torch

from smolagents import Tool

# Importaciones con manejo de errores
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class YouTubeVideoProcessorTool(Tool):
    name = "youtube_processing"
    description = "Process YouTube videos to extract transcript content using multiple methods (subtitles, Whisper). Optimized for limited resources and works in both local and HuggingFace Space environments."
    inputs = {
        'url': {
            'type': 'string', 
            'description': 'The YouTube video URL to process.'
        },
        'language': {
            'type': 'string', 
            'description': 'Preferred language for transcript (e.g., "en", "es"). If not specified, defaults to English or first available.', 
            'nullable': True
        },
        'translate': {
            'type': 'boolean', 
            'description': 'Whether to translate non-English transcripts to English using Whisper.', 
            'nullable': True
        },
        'max_duration': {
            'type': 'integer',
            'description': 'Maximum video duration in seconds to process (default: 600s = 10min)',
            'nullable': True
        },
        'method': {
            'type': 'string',
            'description': 'Processing method: "auto" (try subtitles first), "subtitles" (only), "whisper" (only)',
            'nullable': True
        }
    }
    output_type = "object"

    def __init__(self):
        """Initialize the YouTube processor with optimal settings for limited resources."""
        super().__init__()
        
        # Configuración para recursos limitados
        self.max_file_size_mb = 50  # Límite de archivo de audio
        self.default_max_duration = 600  # 10 minutos por defecto
        self.audio_quality = '64'  # Calidad baja para ahorrar recursos
        
        # Modelos Whisper optimizados
        self.whisper_model = None
        self.transformers_pipeline = None
        self.current_model_type = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Verificar dependencias disponibles
        self._check_dependencies()
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Verificar qué dependencias están disponibles."""
        dependencies = {
            'transformers': TRANSFORMERS_AVAILABLE,
            'whisper': WHISPER_AVAILABLE,
            'yt_dlp': YT_DLP_AVAILABLE,
            'transcript_api': TRANSCRIPT_API_AVAILABLE,
            'langchain': LANGCHAIN_AVAILABLE
        }
        
        self.logger.info(f"Dependencias disponibles: {dependencies}")
        return dependencies
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extraer el ID del video de YouTube de la URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match and len(match.group(1)) == 11:
                return match.group(1)
        return None
    
    def _get_video_info(self, url: str) -> Dict[str, Any]:
        """Obtener información básica del video sin descargarlo."""
        if not YT_DLP_AVAILABLE:
            return {'error': 'yt-dlp no disponible'}
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'uploader': info.get('uploader', ''),
                    'view_count': info.get('view_count', 0),
                    'language': info.get('language', 'unknown')
                }
        except Exception as e:
            return {'error': f'Error obteniendo info del video: {str(e)}'}
    
    def _get_subtitles_transcript(self, video_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Intentar obtener transcript usando subtítulos disponibles."""
        if not TRANSCRIPT_API_AVAILABLE:
            return {'success': False, 'error': 'youtube-transcript-api no disponible'}
        
        try:
            # Definir idiomas a intentar
            languages_to_try = []
            if language:
                languages_to_try.append(language)
            
            # Agregar idiomas comunes como fallback
            common_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
            for lang in common_languages:
                if lang not in languages_to_try:
                    languages_to_try.append(lang)
            
            # Intentar obtener transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages_to_try)
            
            # Formatear transcript
            formatter = TextFormatter()
            full_text = formatter.format_transcript(transcript)
            
            # Obtener metadata del transcript
            detected_language = transcript[0].get('language_code', 'unknown') if transcript else 'unknown'
            
            return {
                'success': True,
                'method': 'subtitles',
                'transcript': full_text,
                'raw_transcript': transcript,
                'detected_language': detected_language,
                'total_segments': len(transcript)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'No se pudieron obtener subtítulos: {str(e)}'
            }
    
    def _setup_whisper_model(self, model_type: str = 'transformers') -> bool:
        """Configurar modelo Whisper optimizado para recursos limitados."""
        try:
            if model_type == 'transformers' and TRANSFORMERS_AVAILABLE:
                if self.transformers_pipeline is None or self.current_model_type != 'transformers':
                    self.logger.info("Cargando modelo Whisper con Transformers...")
                    
                    # Detectar dispositivo disponible
                    device = 0 if torch.cuda.is_available() else -1
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    
                    self.transformers_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model="openai/whisper-small",  # Balance entre tamaño y precisión
                        torch_dtype=torch_dtype,
                        device=device,
                        return_timestamps=True
                    )
                    self.current_model_type = 'transformers'
                return True
                
            elif model_type == 'openai' and WHISPER_AVAILABLE:
                if self.whisper_model is None or self.current_model_type != 'openai':
                    self.logger.info("Cargando modelo Whisper OpenAI...")
                    # Usar modelo pequeño para recursos limitados
                    self.whisper_model = whisper.load_model("small")
                    self.current_model_type = 'openai'
                return True
                
        except Exception as e:
            self.logger.error(f"Error cargando modelo Whisper: {str(e)}")
            return False
        
        return False
    
    def _download_audio(self, url: str, max_duration: int) -> Dict[str, Any]:
        """Descargar audio del video de YouTube con límites de recursos."""
        if not YT_DLP_AVAILABLE:
            return {'success': False, 'error': 'yt-dlp no disponible'}
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Configuración optimizada para recursos limitados
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best[height<=480]',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': self.audio_quality,
                }],
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                # Filtros para limitar recursos
                'match_filter': self._duration_filter(max_duration)
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extraer info primero para verificar duración
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                
                if duration > max_duration:
                    return {
                        'success': False,
                        'error': f'Video demasiado largo: {duration}s > {max_duration}s'
                    }
                
                # Descargar audio
                ydl.download([url])
            
            # Encontrar archivo descargado
            audio_files = []
            for file in os.listdir(temp_dir):
                if file.endswith(('.wav', '.mp3', '.m4a')):
                    audio_files.append(file)
            
            if not audio_files:
                return {'success': False, 'error': 'No se pudo descargar el audio'}
            
            audio_path = os.path.join(temp_dir, audio_files[0])
            
            # Verificar tamaño del archivo
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return {
                    'success': False,
                    'error': f'Archivo demasiado grande: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB'
                }
            
            return {
                'success': True,
                'audio_path': audio_path,
                'temp_dir': temp_dir,
                'file_size_mb': round(file_size_mb, 1),
                'duration': duration
            }
            
        except Exception as e:
            # Limpiar directorio temporal en caso de error
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            return {'success': False, 'error': f'Error descargando audio: {str(e)}'}
    
    def _duration_filter(self, max_duration: int):
        """Filtro para limitar duración de videos."""
        def filter_func(info_dict):
            duration = info_dict.get('duration')
            if duration and duration > max_duration:
                return f"Video demasiado largo: {duration}s"
            return None
        return filter_func
    
    def _transcribe_with_whisper(self, audio_path: str, translate: bool = False) -> Dict[str, Any]:
        """Transcribir audio usando Whisper."""
        try:
            # Intentar con Transformers primero (más eficiente)
            if self._setup_whisper_model('transformers'):
                result = self.transformers_pipeline(
                    audio_path,
                    generate_kwargs={
                        "task": "translate" if translate else "transcribe",
                        "language": None  # Auto-detect
                    }
                )
                
                return {
                    'success': True,
                    'method': 'whisper_transformers',
                    'transcript': result['text'],
                    'chunks': result.get('chunks', []),
                    'detected_language': 'auto-detected'
                }
            
            # Fallback a OpenAI Whisper
            elif self._setup_whisper_model('openai'):
                options = {
                    "task": "translate" if translate else "transcribe",
                    "fp16": torch.cuda.is_available()
                }
                
                result = self.whisper_model.transcribe(audio_path, **options)
                
                return {
                    'success': True,
                    'method': 'whisper_openai',
                    'transcript': result['text'],
                    'segments': result.get('segments', []),
                    'detected_language': result.get('language', 'unknown')
                }
            
            else:
                return {
                    'success': False,
                    'error': 'No hay modelos Whisper disponibles'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en transcripción Whisper: {str(e)}'
            }
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Dividir texto en chunks usando LangChain si está disponible."""
        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return text_splitter.split_text(text)
        else:
            # Fallback simple
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > 1000 and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(word)
                current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
    
    def forward(self, url: str, language: Optional[str] = None, translate: Optional[bool] = None, 
               max_duration: Optional[int] = None, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesar video de YouTube para extraer transcript.
        
        Args:
            url: URL del video de YouTube
            language: Idioma preferido para transcript
            translate: Si traducir a inglés
            max_duration: Duración máxima en segundos
            method: Método de procesamiento ('auto', 'subtitles', 'whisper')
        
        Returns:
            Dict con resultado del procesamiento
        """
        try:
            # Validar URL y extraer video ID
            video_id = self._extract_video_id(url)
            if not video_id:
                return {
                    'success': False,
                    'error': 'URL de YouTube inválida',
                    'url': url
                }
            
            # Configurar parámetros por defecto
            max_duration = max_duration or self.default_max_duration
            translate = translate or False
            method = method or 'auto'
            
            # Obtener información del video
            video_info = self._get_video_info(url)
            if 'error' in video_info:
                return {
                    'success': False,
                    'error': video_info['error'],
                    'url': url
                }
            
            # Verificar duración
            if video_info.get('duration', 0) > max_duration:
                return {
                    'success': False,
                    'error': f"Video demasiado largo: {video_info['duration']}s > {max_duration}s",
                    'video_info': video_info
                }
            
            result = {
                'success': False,
                'video_id': video_id,
                'video_info': video_info,
                'processing_method': method
            }
            
            # Método 1: Intentar subtítulos primero (si method es 'auto' o 'subtitles')
            if method in ['auto', 'subtitles']:
                subtitle_result = self._get_subtitles_transcript(video_id, language)
                
                if subtitle_result['success']:
                    transcript = subtitle_result['transcript']
                    chunks = self._split_text_into_chunks(transcript)
                    
                    result.update({
                        'success': True,
                        'method_used': 'subtitles',
                        'transcript': transcript,
                        'chunks': chunks,
                        'detected_language': subtitle_result.get('detected_language'),
                        'total_segments': subtitle_result.get('total_segments', 0),
                        'raw_data': subtitle_result.get('raw_transcript', [])
                    })
                    
                    return result
                
                # Si solo se pidieron subtítulos y fallaron
                elif method == 'subtitles':
                    result.update({
                        'success': False,
                        'error': subtitle_result['error'],
                        'method_attempted': 'subtitles'
                    })
                    return result
            
            # Método 2: Usar Whisper (si method es 'auto', 'whisper' o fallback)
            if method in ['auto', 'whisper']:
                # Descargar audio
                self.logger.info("Descargando audio para transcripción con Whisper...")
                audio_result = self._download_audio(url, max_duration)
                
                if not audio_result['success']:
                    result.update({
                        'success': False,
                        'error': audio_result['error'],
                        'method_attempted': 'whisper'
                    })
                    return result
                
                # Transcribir con Whisper
                self.logger.info("Transcribiendo con Whisper...")
                whisper_result = self._transcribe_with_whisper(
                    audio_result['audio_path'], 
                    translate=translate
                )
                
                # Limpiar archivos temporales
                import shutil
                try:
                    shutil.rmtree(audio_result['temp_dir'])
                except:
                    pass
                
                if whisper_result['success']:
                    transcript = whisper_result['transcript']
                    chunks = self._split_text_into_chunks(transcript)
                    
                    result.update({
                        'success': True,
                        'method_used': whisper_result['method'],
                        'transcript': transcript,
                        'chunks': chunks,
                        'detected_language': whisper_result.get('detected_language'),
                        'audio_info': {
                            'file_size_mb': audio_result['file_size_mb'],
                            'duration': audio_result['duration']
                        },
                        'translation_applied': translate
                    })
                    
                    return result
                else:
                    result.update({
                        'success': False,
                        'error': whisper_result['error'],
                        'method_attempted': 'whisper'
                    })
                    return result
            
            # Si llegamos aquí, ningún método funcionó
            result.update({
                'success': False,
                'error': 'No se pudo procesar el video con ningún método disponible',
                'methods_attempted': [method]
            })
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error inesperado: {str(e)}',
                'url': url
            }