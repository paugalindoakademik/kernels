// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#ifdef INT
#include "GEN_kernel_integer.hcl"
#else
#include "GEN_kernel.hcl"
#endif

//#define BLOCK_SIZE 16
void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}
void init(GraficObject *device_object, int platform ,int device, char* device_name){
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platformB
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[device];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device.getInfo<CL_DEVICE_NAME>().c_str() );
    // context
    device_object->context = new cl::Context(default_device);
    device_object->queue = new cl::CommandQueue(*device_object->context,default_device,CL_QUEUE_PROFILING_ENABLE);
    device_object->default_device = default_device;
    
    // events
    device_object->evt = new cl::Event; 
    device_object->evt_copyA = new cl::Event;
    device_object->evt_copyB = new cl::Event;
    device_object->evt_copyC = new cl::Event;
    
}

bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
   device_object->d_A = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_matrix);
   device_object->d_B = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_b_matrix);
   #ifdef INT
    // if int don't add the copy of the filters
   #else
   device_object->low_filter = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*LOWPASSFILTERSIZE);
   device_object->high_filter = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*HIGHPASSFILTERSIZE);
   #endif
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a){
	// copy memory host -> device
	//TODO Errors check
    device_object->queue->enqueueWriteBuffer(*device_object->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, device_object->evt_copyA);
    #ifdef INT
    // if int don't add the copy of the filters
    #else
    device_object->queue->enqueueWriteBuffer(*device_object->low_filter,CL_TRUE,0,sizeof(bench_t)*LOWPASSFILTERSIZE, lowpass_filter, NULL, device_object->evt_copyB);
    device_object->queue->enqueueWriteBuffer(*device_object->high_filter,CL_TRUE,0,sizeof(bench_t)*HIGHPASSFILTERSIZE, highpass_filter, NULL, device_object->evt_copyC);
    #endif
}


void execute_kernel(GraficObject *device_object, unsigned int n){
    const unsigned int x_local= BLOCK_SIZE * BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;
    if (n < BLOCK_SIZE * BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(n);
    }
    else
    {
        local = cl::NDRange(x_local);
        global = cl::NDRange(n);
    }
    

    cl::Program::Sources sources;
    device_object->evt = new cl::Event;
    // load kernel from file
    kernel_code = type_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(*device_object->context,sources);
    if(program.build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }
    #ifdef INT
    device_object->evt_int = new cl::Event;
    cl::Kernel kernel_wave=cl::Kernel(program,"wavelet_transform");
    
    kernel_wave.setArg(0,*device_object->d_A);
    kernel_wave.setArg(1,*device_object->d_B);
    kernel_wave.setArg(2,n);
    device_object->queue->enqueueNDRangeKernel(kernel_wave,cl::NullRange,global,local, NULL, device_object->evt);

    cl::Kernel kernel_wave_low=cl::Kernel(program,"wavelet_transform_low");
    kernel_wave_low.setArg(0,*device_object->d_A);
    kernel_wave_low.setArg(1,*device_object->d_B);
    kernel_wave_low.setArg(2,n);
    device_object->queue->enqueueNDRangeKernel(kernel_wave_low,cl::NullRange,global,local, NULL, device_object->evt_int);
    device_object->queue->finish();
    
    #else
    cl::Kernel kernel_wave=cl::Kernel(program,"wavelet_transform");
    kernel_wave.setArg(0,*device_object->d_A);
    kernel_wave.setArg(1,*device_object->d_B);
    kernel_wave.setArg(2,n);
    kernel_wave.setArg(3,*device_object->low_filter);
    kernel_wave.setArg(4,*device_object->high_filter);
    device_object->queue->enqueueNDRangeKernel(kernel_wave,cl::NullRange,global,local, NULL, device_object->evt);
    device_object->queue->finish();
    #endif

}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    device_object->queue->enqueueReadBuffer(*device_object->d_B,CL_TRUE,0,sizeof(bench_t)*size,h_C, NULL, device_object->evt_copyC);
}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    device_object->evt_copyC->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = device_object->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    elapsed = device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    #ifdef INT
    elapsed += device_object->evt_int->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_int->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    #endif
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
    elapsed_d_h = device_object->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );


    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f miliseconds\n", elapsed / 1000000.0);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

void clean(GraficObject *device_object){
    // pointers clean
    delete device_object->context;
    delete device_object->queue;
    // pointer to memory
    delete device_object->d_A;
    delete device_object->d_B;
    #ifdef INT
    delete device_object->evt_int;
    #else
    delete device_object->low_filter;
    delete device_object->high_filter;
    #endif
    delete device_object->evt;
    delete device_object->evt_copyA;
    delete device_object->evt_copyB;
    delete device_object->evt_copyC;
}
