CXX = g++ -g -O3 -Wall -std=c++11 -lm -fopenmp

testsubm: test_submodular.cpp Graph.hpp Slave.hpp
	$(CXX) -o test_submodular test_submodular.cpp
