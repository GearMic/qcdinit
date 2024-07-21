all: paper build

build: src/main.py
	python src/main.py
	touch build

paper: build latex/main.tex
	cp -t latex plot/*.pdf
	pdflatex -interaction=nonstopmode -output-directory=./latex latex/main.tex 
