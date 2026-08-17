This is a tool designed to use FFMPEG to convert all various media formats into AV1 format, both 1080p/30 and 4k/30. 

It will detect the format, frames, cropping, and all aspects of the input video, then calibrate itself to only render active lines.
It will upscale anything below 1080p to 1080p. 
It will pass through footage already 1080p.
It will pass through footage already 4k (if set). or down convert to 1080p. 
It will convert SDR to HDR 10bit if selected. 
It will pull metadata, and create an info sidecar.
It will sort audio tracks to only English and japaneese (for anime), automagically putting English first. 
It will generate a boosted audio English track 1 from any complex audio tracks. 
It will generate subtitles in both English and translated to japaneese automagically. 

Still very much in alpha. 
Designed for 5000+ RTX nvidia GPU's with CUDA.

Enjoy. 
