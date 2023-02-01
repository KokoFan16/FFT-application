#!/bin/bash

while getopts f: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;
    esac
done

nfile=$(ls -l . | grep $filename\_.*.csv | wc -l);
echo "File name: $filename, $nfile";

if [[ $nfile -gt 0 ]] 
then
    cat $filename\_*csv > $filename.csv;
    rm -rf $filename\_*csv 
else
    echo "merge nothing"; 
fi

