#Targets
par_target = mpi-k-folds.c knnomp.c 
ser_target = k-folds.c knnomp.c 
file_reader_target = file-reader.c
compare_target = compare_knn.c

#Executables
ser_out = k-folds
par_out = k-folds-complete

#Flags
op = -O3

all: 
	make gccnearly
	make gcccomplete
	make iccnearly
	make icccomplete

gccnearly: $(ser_target) $(file_reader_target)
	gcc -fopenmp -std=c99 $(op) $(ser_target) $(file_reader_target) -o $(ser_out)-gcc -lm

gcccomplete: $(par_target) $(file_reader_target)
	mpicc -fopenmp -std=c99 $(op) $(par_target) $(file_reader_target) -o $(par_out)-gcc -lm

iccnearly: $(ser_target) $(file_reader_target)
	icc -qopenmp -std=c99 $(op) $(ser_target) $(file_reader_target) -o $(ser_out)-icc

icccomplete: $(par_target) $(file_reader_target)
	mpiicc -qopenmp -std=c99 $(op) $(par_target) $(file_reader_target) -o $(par_out)-icc

debug: ${par_target} $(file_reader_target)
	mpicc -g -fopenmp -O0 -std=c99 $(par_target) $(file_reader_target) -o $(par_out)-db -lm

iccompreport: $(par_target) $(file_reader_target)
	icc -qopenmp -std=c99 $(op) $(par_target) $(file_reader_target) -o $(par_out)-icc -qopt-report3

compare: $(compare_target) $(file_reader_target)
	gcc -std=c99 $(op) $(compare_target) $(file_reader_target) -o compare -lm

comparedb: $(compare_target) $(file_reader_target)
	gcc -g -std=c99 $(compare_target) $(file_reader_target) -o compare-db -lm

clear:
	rm $(ser_out)-gcc
	rm $(par_out)-gcc
	rm $(ser_out)-icc
	rm $(par_out)-icc