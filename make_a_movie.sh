#! /bin/sh

ffmpeg -y -vcodec png -r 3 -start_number 1 -i movies/MDQN_modern/MsPacman/1/%05d0.png -frames:v 5858 -c:v libx264 -vf fps=60 -pix_fmt yuv420p -crf 17 -preset veryslow MsPacman.mp4 