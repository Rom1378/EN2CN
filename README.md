﻿# EN2CN - Subtitle Translator Project

This project provides tools for downloading, translating, and handling subtitle files. It includes a Streamlit web app for interactive subtitle translation with support for multiple languages.

## Quick Start Guide

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Subtitle Translator App
```bash
streamlit run streamlit_translator.py
```
- This launches a web UI for downloading YouTube videos and translating their subtitles
- Select your target language from the dropdown menu
- Enter a YouTube video URL
- The app will download the video and subtitles, then translate them to your chosen language

## Features

- Download videos and subtitles from YouTube
- Support for multiple target languages
- Automatic subtitle format conversion (VTT ↔ SRT)
- Interactive web UI with video preview
- Side-by-side comparison of original and translated subtitles
- Download options for both original and translated subtitles
- Robust error handling and retry mechanisms
- Network resilience with automatic retries


## Technical Details

### Video Download
- Uses yt-dlp for reliable video and subtitle downloads
- Automatically selects optimal video quality (up to 720p)
- Includes both video and audio streams
- Handles network issues with automatic retries

### Subtitle Processing
- Supports both VTT and SRT subtitle formats
- Automatic format conversion as needed
- Maintains subtitle timing and formatting
- Preserves line breaks and styling

### Translation
- Uses Google's Gemini API for high-quality translations
- Maintains context between subtitle segments
- Preserves formatting and timing constraints
- Handles special characters and formatting

## Error Handling

The application includes robust error handling for:
- Network connectivity issues
- Download failures
- Subtitle format conversion
- Translation errors
- File system operations

## Project Structure

- `streamlit_translator.py`: Main Streamlit application
- `gemini_translator.py`: Translation service implementation
- `subtitle_handler.py`: Subtitle download and processing utilities
- `requirements.txt`: Project dependencies

## Dependencies

- streamlit>=1.32.0
- yt-dlp>=2024.3.10
- google-generativeai>=0.3.2
- python-dotenv>=1.0.1
- requests>=2.31.0

## Future Plans

- Add support for more languages
- Implement batch processing for multiple videos
- Add support for custom subtitle files
- Improve translation quality with better context handling
- Add support for more video platforms

## Notes

- A valid Gemini API key is required for translation
- The application requires an active internet connection
- Video downloads may take some time depending on the video size and your connection speed
- Some videos may not have available subtitles
