#!/bin/bash
curl -L -o ./Dataset/guitar-chords-v3.zip\
  https://www.kaggle.com/api/v1/datasets/download/fabianavinci/guitar-chords-v3

unzip -o ./Dataset/guitar-chords-v3.zip -d ./Dataset
rm ./Dataset/guitar-chords-v3.zip