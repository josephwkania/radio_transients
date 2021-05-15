#!/bin/bash

# This file auto-updates the build dates in Singularity, Singularity.cpu and Singularity.gpu
# the commits the new files, pushed to github
# Singularity hub should then rebuild the containers

# makes changes so git sees the files are new
today=$(date +'%d-%b-%Y')
sed -i -e "s|Build-date.*|Build-date ${today}|g" Singularity
sed -i -e "s|Build-date.*|Build-date ${today}|g" Singularity.cpu
sed -i -e "s|Build-date.*|Build-date ${today}|g" Singularity.gpu
sed -i -e "s|They where last built on.*|They where last built on ${today}|g" README.md

# push to github where Singularity hub will see the files
# and rebuild the containers
git add .
git commit -m "${today} auto rebuild"
git push origin master
