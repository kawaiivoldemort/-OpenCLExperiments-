#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <CL/cl.hpp>

std::string load_file(std::string filename) {
	std::ifstream ifs(filename);
	return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));

}

int* multiply(
	         int* h_a,
	         int* h_b,
	unsigned int  i,
	unsigned int  j,
	unsigned int  k
) {
	// Get Platforms
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	cl::Platform platform = all_platforms[0];
	if (all_platforms.size() == 0) {
		// error
	}
	// Get the devices on the platform
	std::vector<cl::Device> all_devices;
	platform.getDevices(
		CL_DEVICE_TYPE_GPU,
		&all_devices
	);
	cl::Device device = all_devices[0];
	std::cout << "Using Device " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
	// Create the context to execute in
	const std::vector<cl::Device> devices({ device });
	cl::Context context(devices);
	// Create a command queue to execute on the device
	cl::CommandQueue queue(context, device);
	// Create the program
	std::string source_code = load_file("vector_mul.cl");
	cl::Program::Sources sources(1, std::make_pair(source_code.c_str(), source_code.length()));
	cl::Program program(context, sources);
	// Build the program
	if (program.build({ device }) != CL_SUCCESS) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
	}
	// Allocate buffers for read and write and copy
	cl::Buffer d_a(context, CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY, sizeof(int) * i * j, h_a);
	cl::Buffer d_b(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * j * k, h_b);
	cl::Buffer d_c(context, CL_MEM_READ_ONLY, sizeof(int) * i * k);
	// Create the kernel and set arguments
	cl::Kernel kernel(program, "vector_mul");
	kernel.setArg(0, d_a);
	kernel.setArg(1, d_b);
	kernel.setArg(2, d_c);
	kernel.setArg(3, sizeof(int), &j);
	kernel.setArg(4, sizeof(int), &k);
	// Set the work size	
	size_t global_work_size[] = { i };
	// Enqueue the kernel as a command
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(i));
	/* cl::KernelFunctor vmul(
		cl::Kernel(program, "vector_mul"),
		queue,
		cl::NullRange,
		cl::NDRange(i)
	);
	vmul(d_a, d_b, d_c, j, k);*/
	// Complete the command
	queue.finish();
	// Copy back
	int* h_c = (int*) malloc(i * k * sizeof(int));
	queue.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(int) * i * k, h_c);
	return h_c;
	// remember async work group copy and async work group strided copy
	// remember how cpus dont have local memory but gpus do, so using a lot can make the kernel slower on cpus
	// also, gpus have on chip caches that can provide the benefit of local memory without any explicit programming
}

int main() {
	srand(time(NULL));
	unsigned int i = rand() % 3 + 1;
	unsigned int j = rand() % 3 + 1;
	unsigned int k = rand() % 3 + 1;
	int d = 5;
	int* a = (int*) malloc(i * j * sizeof(int));
	int* b = (int*) malloc(j * k * sizeof(int));
	for(int l = 0; l < i; l++) {
		for(int m = 0; m < j; m++) {
			a[l * j + m] = rand() % 10;
		}
	}
	for(int l = 0; l < j; l++) {
		for(int m = 0; m < k; m++) {
			b[l * k + m] = rand() % 10;
		}
	}
	int* c = multiply(a, b, i, j, k);
	std::cout << "\nA:\n";
	for(int l = 0; l < i; l++) {
		std::cout << "[ " << a[l * j];
		for(int m = 1; m < j; m++) {
			std::cout << ", " << a[l * j + m];
		}
		std::cout << " ]\n";
	}
	std::cout << "\nB:\n";
	for(int l = 0; l < j; l++) {
		std::cout << "[ " << b[l * k];
		for(int m = 0; m < k; m++) {
			std::cout << ", " << b[l * k + m];
		}
		std::cout << " ]\n";
	}
	std::cout << "\nC:\n";
	for(int l = 0; l < i; l++) {
		std::cout << "[ " << c[l * k];
		for(int m = 0; m < k; m++) {
			std::cout << ", " << c[l * k + m];
		}
		std::cout << " ]\n";
	}
	std::cout << "\n";
	free(a);
	free(b);
	free(c);
	return 0;
}