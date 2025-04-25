import os
import subprocess
import logging
import shutil
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        # Try to find ffmpeg in PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            logger.info(f"Found FFmpeg at: {ffmpeg_path}")
            return True
        
        # Check common Windows installation locations
        windows_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
        ]
        
        for path in windows_paths:
            if os.path.exists(path):
                logger.info(f"Found FFmpeg at: {path}")
                return True
        
        return False
    except Exception:
        return False

def get_video_info(video_path: str) -> dict:
    """Get video information using FFprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return None

def add_subtitles_to_video(video_path: str, subtitle_path: str, output_path: str) -> None:
    """
    Add subtitles to a video file using FFmpeg.
    
    Args:
        video_path: Path to the input video file
        subtitle_path: Path to the SRT subtitle file
        output_path: Path where the output video will be saved
    """
    try:
        # Check if FFmpeg is installed
        if not check_ffmpeg():
            logger.error("""FFmpeg is not installed or not found in PATH. Please install FFmpeg:
            
For Windows:
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract the downloaded zip file
3. Add the 'bin' folder to your system PATH
   OR
4. Use Chocolatey: choco install ffmpeg

For Linux:
sudo apt-get install ffmpeg

For macOS:
brew install ffmpeg""")
            return

        # Convert paths to absolute paths
        video_path = os.path.abspath(video_path)
        subtitle_path = os.path.abspath(subtitle_path)
        output_path = os.path.abspath(output_path)
        
        # Check if input files exist
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Adding subtitles to video: {video_path}")
        logger.info(f"Using subtitle file: {subtitle_path}")
        
        # Get video information
        video_info = get_video_info(video_path)
        if not video_info:
            raise Exception("Could not get video information")
        
        # Find video stream
        video_stream = next((s for s in video_info['streams'] if s['codec_type'] == 'video'), None)
        if not video_stream:
            raise Exception("No video stream found")
        
        width = video_stream['width']
        height = video_stream['height']
        
        # FFmpeg command to add subtitles
        # We need to re-encode the video to burn in subtitles
        # Use proper escaping for paths with spaces and special characters
        subtitle_filter = f"subtitles='{subtitle_path.replace('\\', '/')}':force_style='FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF&':original_size={width}x{height}"
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', subtitle_filter,
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'medium',  # Encoding preset (faster than 'slow' but still good quality)
            '-crf', '23',  # Constant Rate Factor (18-28 is good, lower is better quality)
            '-c:a', 'aac',  # Audio codec
            '-b:a', '192k',  # Audio bitrate
            output_path
        ]
        
        # Run FFmpeg command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log progress
        logger.info("Processing video...")
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Successfully added subtitles. Output saved to: {output_path}")
        else:
            logger.error(f"Error adding subtitles: {stderr}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Add subtitles to a video file')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('subtitle_path', help='Path to the SRT subtitle file')
    parser.add_argument('output_path', help='Path where the output video will be saved')
    
    args = parser.parse_args()
    
    add_subtitles_to_video(args.video_path, args.subtitle_path, args.output_path)

if __name__ == '__main__':
    main() 