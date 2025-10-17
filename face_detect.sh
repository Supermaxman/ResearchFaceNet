#!/usr/bin/env bash
source ~/dev/bin/activate
cd ~/ResearchFaceNet
python3 face_detect.py -cx 640 -cy 480 -ux 512 -uy 384 -pi --threshold 0.925 -fs 
