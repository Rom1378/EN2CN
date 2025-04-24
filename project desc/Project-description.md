# YouTube Subtitle Enhancement and Translation Project

In the folder [srt-files](./srt-files), you can find a bunch of video srt files downloaded from YouTube.

In fact, you can see the id of those videos in their file names, e.g. for `2015 NBA Finals Mini-Movie, Games 1-6 + More [3pxKO_-Cp-o].srt`, the id is `3pxKO_-Cp-o`. Hence, its corresponding video is `https://www.youtube.com/watch?v=3pxKO_-Cp-o`.

You can download the srt files as well as the video files with the tool `yt-dlp`, or `you-get`. Or, simply Google "download youtube videos online".

## Current Challenges

Problems that I have identified:
- Some of the srt in English might not be good enough. Though most should be fine. One perticular keyword, for example, `foreign`, can be very misleading. Or, sometimes, the original srt file downloaded from YouTube might not be good enough.
- If we translate the srt files into Chinese/French with non-LLM AI tools (e.g. `Helsinki-NLP/opus-mt-en-zh`), the result might not be good, sometimes very bad. See [week-4/practice-2-output-srt-files](../week-4/practice-2-output-srt-files/). From my first experiences of trial and error, there is no reason to expect better results with other non-LLM tools.

## Project Objectives

I would like you guys to:
- identify things that can be pitfalls in the srt files (e.g. the keyword `foreign`).
- explore different methods to improve the quality of the srt files. For example, maybe youtube srt files could be enhanced by Whisper, it will be like information fusion from two sources. [might be too CPU-intensive and time-consuming, hence just simple exploration and you can stop]
- it's important to emphasize that having a high quality EN srt file is 50% of the objective of the work.
- explore different LLM models to translate the srt files into Chinese/French (maybe Local (preferred), or, if local is not enough, Cloud LLM API). 
- identify the minimum viable LLM models (e.g. `Qwen2.5:3B`) to translate the srt files for the videos with acceptable quality.
- how to make the translation result better (e.g. prompt engineering, context awareness, continuity checks to ensure logical sentence flow across subtitle segments, feed into the information of the video (video title, description, channel title, channel description, etc.), etc.).
- how much time will it take to translate the srt files for all videos (let's say 1000 srt files). Propose an efficient processing pipeline to minimize/balance/optimize time and resource consumption.

However, if your subtitle files (both in English and the translated Chinese/French) are of sufficient quality, you may choose to skip the above objectives.

And yes, your project should be implemented in Python code.

The project can be quite open, it can be research oriented, or application oriented, or a blend of both, depending on your taste and styling. I have no idea what will come out from this project yet. Hence, just play hard, work harder, have fun and surprise me.

## [Optional] Human-in-the-Loop Validation with Streamlit
   - Implement targeted human review for segments flagged as potentially problematic
   - Create an efficient interface for reviewers to quickly approve or correct translations
   - Develop a feedback mechanism to improve system performance based on corrections



