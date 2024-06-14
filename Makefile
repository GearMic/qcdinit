all: build

build: latex/main.tex
	cp -t latex plot/*.pdf
	pdflatex -interaction=nonstopmode -output-directory=./latex latex/main.tex 
