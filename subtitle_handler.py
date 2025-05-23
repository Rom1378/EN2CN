import re
import logging
import tempfile
import os
import yt_dlp
import shutil
import time
from urllib.error import URLError
import socket

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def download_subtitles(url, max_retries=3, retry_delay=5):
    """Download video and subtitles from URL using yt-dlp with retries
    
    Args:
        url (str): The URL of the video to download
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
    """
    logger.debug(f"Starting download for URL: {url}")
    
    # Create a permanent temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {temp_dir}")
    
    for attempt in range(max_retries):
        try:
            # First, get available formats
            with yt_dlp.YoutubeDL({
                'quiet': True,
                'socket_timeout': 30,
                'retries': 10,
                'fragment_retries': 10,
                'extractor_retries': 10,
                'ignoreerrors': True
            }) as ydl:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}: Getting available formats")
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise ValueError("Could not extract video information")
                    
                formats = info.get('formats', [])
                logger.debug(f"Available formats: {[f.get('format_id') for f in formats]}")
                
                # Find the best format that includes both video and audio
                best_format = None
                for f in formats:
                    if (f.get('height', 0) <= 720 and 
                        f.get('vcodec', 'none') != 'none' and 
                        f.get('acodec', 'none') != 'none'):
                        best_format = f.get('format_id')
                        break
                
                if not best_format:
                    # If no combined format found, try to get best video + best audio
                    best_format = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            
            # First download the video
            video_opts = {
                'format': best_format,
                'merge_output_format': 'mp4',
                'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'verbose': False,
                'socket_timeout': 30,
                'retries': 10,
                'fragment_retries': 10,
                'extractor_retries': 10,
                'ignoreerrors': True
            }
            
            logger.debug(f"Using format: {best_format}")
            logger.debug("Downloading video...")
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                video_info = ydl.extract_info(url, download=True)
                if not video_info:
                    raise ValueError("Failed to download video")
                logger.debug(f"Video downloaded: {video_info.get('title', 'Unknown title')}")
            
            # Then try to download subtitles separately
            subtitle_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,  # Skip video download since we already have it
                'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'verbose': False,
                'socket_timeout': 30,
                'retries': 10,
                'fragment_retries': 10,
                'extractor_retries': 10,
                'ignoreerrors': True
            }
            
            logger.debug("Downloading subtitles...")
            try:
                with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                    sub_info = ydl.extract_info(url, download=True)
                    logger.debug("Subtitles downloaded successfully")
            except Exception as e:
                logger.warning(f"Error downloading subtitles: {str(e)}")
                # Try alternative subtitle download method
                try:
                    subtitle_opts['writeautomaticsub'] = False
                    with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                        sub_info = ydl.extract_info(url, download=True)
                        logger.debug("Subtitles downloaded with alternative method")
                except Exception as e2:
                    logger.error(f"Failed to download subtitles with alternative method: {str(e2)}")
                    # Continue without subtitles
            
            # Get the video file path
            video_path = os.path.join(temp_dir, 'video.mp4')
                
            # Get the subtitle file path
            subtitle_path = os.path.join(temp_dir, 'video.en.vtt')
            if not os.path.exists(subtitle_path):
                subtitle_path = os.path.join(temp_dir, 'video.en.srt')
            
            if os.path.exists(video_path):
                logger.debug(f"Video file exists at: {video_path}")
                if os.path.exists(subtitle_path):
                    logger.debug(f"Subtitle file exists at: {subtitle_path}")
                    return subtitle_path, video_path, video_info, temp_dir
                else:
                    logger.warning("No subtitle file found, continuing with video only")
                    return None, video_path, video_info, temp_dir
            else:
                logger.error(f"Video file not found at: {video_path}")
                logger.debug(f"Directory contents: {os.listdir(temp_dir)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return None, None, None, temp_dir
                    
        except (URLError, socket.gaierror) as e:
            logger.error(f"Network error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return None, None, None, temp_dir
        except Exception as e:
            logger.error(f"Error downloading video and subtitles: {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return None, None, None, temp_dir
    
    return None, None, None, temp_dir

def convert_vtt_to_srt(vtt_content):
    """Convert VTT format to SRT format"""
    # Remove VTT header and WEBVTT line
    lines = vtt_content.split('\n')
    while lines and (lines[0].startswith('WEBVTT') or not lines[0].strip()):
        lines.pop(0)
    
    # Convert timestamps and format
    srt_lines = []
    counter = 1
    i = 0
    
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
            
        # Timestamp line
        if '-->' in lines[i]:
            timestamp = lines[i].replace('.', ',')
            srt_lines.append(str(counter))
            srt_lines.append(timestamp)
            counter += 1
            i += 1
            
            # Text lines
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            srt_lines.append('\n'.join(text_lines))
            srt_lines.append('')
        else:
            i += 1
    
    return '\n'.join(srt_lines)

def convert_srt_to_vtt(srt_content):
    """Convert SRT format to VTT format"""
    # Add VTT header
    vtt_content = "WEBVTT\n\n"
    
    # Convert each subtitle entry
    entries = parse_srt(srt_content)
    for match in entries:
        # Convert timestamp format from SRT to VTT
        start_time = match.group(2).replace(',', '.')
        end_time = match.group(3).replace(',', '.')
        text = match.group(4).strip()
        
        # Add the entry to VTT content
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    
    return vtt_content

def parse_srt(content):
    """Parse SRT content and return list of subtitle entries"""
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
    entries = list(re.finditer(pattern, content, re.DOTALL))
    return entries 