#!/bin/bash
curl -L -o ./dataset/guitar-chords-v3.zip\
  https://www.kaggle.com/api/v1/datasets/download/fabianavinci/guitar-chords-v3

unzip -o ./dataset/guitar-chords-v3.zip -d ./dataset
rm ./dataset/guitar-chords-v3.zip