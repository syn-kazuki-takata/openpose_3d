#!/bin/bash

IFS_BK=$IFS
IFS_BR=$'\n'
IFS=$IFS_BR

pairs=(
  "007835552747 007521652647"
  "007521652647 001174160447"
  "001174160447 006847352647"
  "006847352647 007835552747"
)


for pair in ${pairs[@]}; do
  IFS=$IFS_BK
  ../bin/calib_relative $pair
  IFS=$IFS_BR
done

IFS=$IFS_BK

echo ${#pairs[@]}
