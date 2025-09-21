#!/bin/sh

cp ../pdf_minimizer/minimizer.py .

rm -r minned/

python3 minimizer.py pdf_files/ decomp/ minned/ 

