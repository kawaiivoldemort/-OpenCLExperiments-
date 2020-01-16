compiler_include = -I "$(AMDAPPSDKROOT)/include"
linker_options = -L "$(AMDAPPSDKROOT)/lib/x86_64" -l OpenCL

build: main.cpp
	g++ -std=c++11 $(compiler_include) main.cpp $(linker_options)