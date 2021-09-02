#!/bin/bash/
# 전체 경로명이 아니라 확장자를 제외한 파일명만 입력할 것 

input=$1
file_name=${input%%.*}
echo ${file_name}
function converter() {
    jupyter nbconvert --to markdown `pwd`/_notebooks/${file_name} --output-dir=`pwd`/_posts
    echo "Success"
}
converter