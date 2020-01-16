CC=gcc
INTELINCLUDE=-I "$(INTELOCLSDKROOT)/include"
INTELLINK=-L "$(INTELOCLSDKROOT)/bin/icd/x64" -lOpenCL

all: program.exe

program.exe: main.o
	$(CC) load_file.o lu.o main.o $(INTELLINK) -fopenmp -o program.exe

main.o: main.c lu.c lu.h load_file.c load_file.h
	$(CC) -std=c11 $(INTELINCLUDE) -c $(CPUFLAGS) main.c lu.c load_file.c

run-cpu: app-CPU
	./app-CPU

clean: 
	rm -f *.o *.optrpt