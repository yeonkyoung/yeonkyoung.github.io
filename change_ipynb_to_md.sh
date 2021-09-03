#!/bin/bash/
# 전체 경로명이 아니라 확장자를 제외한 파일명만 입력할 것 

file_path=$1
file_path=${file_path#*./_notebooks}
input=${file_path%%.*}
echo ${file_name}
function converter() {
    jupyter nbconvert --to markdown `pwd`/_notebooks/${input} --output-dir=`pwd`/_posts --ExtractOutputPreprocessor.enabled=False
    echo "Success"
}
converter