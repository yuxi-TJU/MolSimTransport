#!/bin/bash

mkdir -p output

for i in {0..200}; do
  if [ -f "output/$i/output.log" ]; then
    echo "Skipping $i (already completed)"
    continue
  fi

  mkdir -p "output/$i"
  cd "output/$i" || exit 1
  
  L1_EHT -f "../../${i}.xyz" -L 36 -R 78 -C 0.2 --Erange -12 -8.5 --Enum 300 > output.log 2>&1
  
  cd - > /dev/null
done
