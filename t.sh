#!/bin/sh

# cp ../pdf_minimizer/minimizer.py .

rm -r minned/

mkdir minned/

python3 minimizer.py in/ decomp/ minned/ 

