EXPNAME := SJ1798_MOPSgly17aa_arab02pc_001
#TIFFS := ./TIFF/$(EXPNAME)*.tif
PARAMPROCESSING := roles/params_processing.yaml
ND2 := ../data/$(EXPNAME).nd2

# targets
TIFF := target_tiff
COMPILE := target_compile

all: $(TIFF)

#$(TIFF): $(PARAMPROCESSING)
$(TIFF): $(ND2)
	python mm3/mm3_nd2ToTIFF.py -f $(PARAMPROCESSING) $(ND2)
	touch $(TIFF)

$(COMPILE): $(TIFF) $(PARAMPROCESSING)
	python mm3/mm3_Compile.py -f $(PARAMPROCESSING) -j 2
	touch $(COMPILE)

### TO DELETE
#OBJS = $(SRC:.cpp=.o)
#PROG = ../bin/prog
#PROG_DEBUG = ../bin/prog_debug
#all: $(PROG)
#debug: $(PROG_DEBUG)
#
#$(PROG): $(OBJS)
#	 $(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

#main.o: utils.h cell.h global.h main.cpp
#	$(CC) $(CFLAGS) -c main.cpp
#
#cell.o: cell.h cell.cpp
#	$(CC) $(CFLAGS) -c cell.cpp
#
#utils.o: utils.h utils.tpp utils.cpp
#	$(CC) $(CFLAGS) -c utils.cpp
#
#linalg.o: linalg.h linalg.cpp
#	$(CC) $(CFLAGS) -c linalg.cpp
#
#physics.o: physics.h physics.cpp
#	$(CC) $(CFLAGS) -c physics.cpp
#
#stepper.o: stepper.h stepper.cpp
#	$(CC) $(CFLAGS) -c stepper.cpp
#
#clean:
#	rm -f *~ *.o $(PROG) core a.out
#
#$(PROG_DEBUG): main_debug.o cell.o utils.o linalg.o physics.o stepper.o
#	 $(CC) $(CFLAGS) main_debug.o cell.o utils.o linalg.o physics.o stepper.o -o $@ $(LDFLAGS) $(LDLIBS)
#
#main_debug.o: utils.h cell.h global.h main_debug.cpp
#	$(CC) $(CFLAGS) -c main_debug.cpp

