all: paper build

build: plot/correlator.pdf
	python src/main.py

paper: build latex/main.tex
	cp -t latex plot/*.pdf
	pdflatex -interaction=nonstopmode -output-directory=./latex latex/main.tex 
