CXX		= CC
CXXFLAGS	= -O3 -g -march=native
OMP		= -fopenmp

assign	: assign.o tipsy.o gpufft.o
	$(CXX) $(OMP) -o assign assign.o gpufft.o tipsy.o -lfftw3f -lfftw3f_mpi -lcufft -lcudart

assign.o : assign.cxx tipsy.h gpufft.h
	$(CXX) $(OMP) $(CXXFLAGS) -O3 -std=c++17 -c -o assign.o assign.cxx -I /store/uzh/uzh8/packages/include

gpufft.o : gpufft.cu gpufft.h
	nvcc  -O3 -std=c++17 -c -o gpufft.o gpufft.cu -I /store/uzh/uzh8/packages/include

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(CXXFLAGS) -O3 -std=c++17 -c -o tipsy.o tipsy.cxx -I /store/uzh/uzh8/packages/include


clean:
	rm -f assign assign.o tipsy.o gpufft.o fft10-cpu fft10-gpu
