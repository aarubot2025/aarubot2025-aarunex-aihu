#!/bin/bash
echo "Starting GitHub push..."
git init
git remote add origin https://github.com/aarubot2025/aarunex-aihu.git
git add .
git commit -m 'Push from working local version of AaruNex'
git push -u origin main
