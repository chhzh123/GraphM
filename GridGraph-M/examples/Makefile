ROOT_DIR = ../

CXX = g++
CXXFLAGS = -O3 -std=c++11 -g -fopenmp -I$(ROOT_DIR)

ALL = Homo1 Homo2 Heter M-BFS M-SSSP

all: $(ALL)

% : %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: Heter
	./Heter ../../../Dataset/cit-Patents-grid 2 12 6 149 1

clean:
	rm -rf $(ALL)