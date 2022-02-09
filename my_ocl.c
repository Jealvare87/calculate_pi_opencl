#include <stdio.h>
#include "my_ocl.h"
#include "common.c"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

double calc_piOCL(int n)
{
	double pi, *area;
	area = malloc(sizeof(double));
	double * areaGPU;
	int *m;
	m = malloc(sizeof(int));*m=0;
	int *mgpu;


	*area = 0.0;
	//printf("Not Implemented yet!!\n");

	// OpenCL host variables
	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[1];


	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source
	
	/*--------------------------------------------------------------------------------*/
	// read the kernel
		fp = fopen("calcpi.cl","r");
		fseek(fp,0L, SEEK_END);
		filelen = ftell(fp);
		rewind(fp);

		kernel_src = malloc(sizeof(char)*(filelen+1));
		readlen = fread(kernel_src,1,filelen,fp);
		if(readlen!= filelen)
		{
			printf("error reading file\n");
			exit(1);
		}
		
		// ensure the string is NULL terminated
		kernel_src[filelen]='\0';


	/*--------------------------------------------------------------------------------*/

	/* OpenCL Declarations */
	//1.-
	// Set up platform and GPU device

		cl_uint numPlatforms;

	// Find number of platforms
		err = clGetPlatformIDs(0, NULL, &numPlatforms);
		if (err != CL_SUCCESS || numPlatforms <= 0)
		{
			printf("Error: Failed to find a platform!\n%s\n",err_code(err));
			return EXIT_FAILURE;
		}

	// Get all platforms
		cl_platform_id Platform[numPlatforms];
		err = clGetPlatformIDs(numPlatforms, Platform, NULL);
		if (err != CL_SUCCESS || numPlatforms <= 0)
		{
			printf("Error: Failed to get the platform!\n%s\n",err_code(err));
			return EXIT_FAILURE;
		}

	// Secure a GPU
		int i;
		for (i = 0; i < numPlatforms; i++)
		{
			err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
			//if (err == CL_SUCCESS)
			if(i == 1)
			{
				break;
			}
		}

		if (device_id == NULL)
		{
			printf("Error: Failed to create a device group!\n%s\n",err_code(err));
			return EXIT_FAILURE;
		}

		err = output_device_info(device_id);

	// Create a compute context 
		context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
		if (!context)
		{
			printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
			return EXIT_FAILURE;
		}

	// Create a command queue
		cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
		if (!commands)
		{
			printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
			return EXIT_FAILURE;
		}

	// create command queue 
		command_queue = clCreateCommandQueue(context,device_id, 0, &err);
		if (err != CL_SUCCESS)
		{	
			printf("Unable to create command queue. Error Code=%d\n",err);
			exit(1);
		}
	//
	//2.-
	// create program object from source. 
	// kernel_src contains source read from file earlier
		program = clCreateProgramWithSource(context, 1 ,(const char **)
	                                          &kernel_src, NULL, &err);
		if (err != CL_SUCCESS)
		{	
			printf("Unable to create program object. Error Code=%d\n",err);
			exit(1);
		}  

		err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		if (err != CL_SUCCESS)
		{
	        	printf("Build failed. Error Code=%d\n", err);

			size_t len;
			char buffer[2048];
			// get the build log
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
	                                  sizeof(buffer), buffer, &len);
			printf("--- Build Log -- \n %s\n",buffer);
			exit(1);
		}

	//
	//3.- y 4.-
		kernel = clCreateKernel(program, "calc_pi", &err);
		if (err != CL_SUCCESS)
		{	
			printf("Unable to create kernel object. Error Code=%d\n",err);
			exit(1);
		}
	
	// create buffer objects to input and output args of kernel function
		areaGPU       	=   clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double), NULL, NULL);
		mgpu	       	=   clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(int), NULL, NULL);
	// Write a and b vectors into compute device memory 
    	clEnqueueWriteBuffer(commands, areaGPU, CL_TRUE, 0, sizeof(double), area, 0, NULL, NULL);
    	clEnqueueWriteBuffer(commands, mgpu, CL_TRUE, 0, sizeof(int), m, 0, NULL, NULL);
	// set the kernel arguments
		if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &areaGPU) ||
			 clSetKernelArg(kernel, 1, sizeof(cl_mem), &mgpu) ||
	         clSetKernelArg(kernel, 2, sizeof(cl_int), &n) != CL_SUCCESS)
		{
			printf("Unable to set kernel arguments. Error Code=%d\n",err);
			exit(1);
		}
	//
	//5.-
	// set the global work dimension size
		global[0] = n;

		double t0d = getMicroSeconds();
		err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
	                                   global, NULL, 0, NULL, NULL);
		double t1d = getMicroSeconds();

		if (err != CL_SUCCESS)
		{	
			printf("Unable to enqueue kernel command. Error Code=%d\n",err);
			exit(1);
		}else{
			printf("\nReduce Noise Device tDevice=%f (s.)\n", (t1d-t0d)/1000000);
		}

	// wait for the command to finish
		clFinish(command_queue);

	// read the output back to host memory
		err = clEnqueueReadBuffer(commands, areaGPU, CL_TRUE, 0, sizeof(double), area, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{	
			printf("Error enqueuing read buffer command. Error Code=%d\n",err);
			exit(1);
		}

	// clean up
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(command_queue);
		clReleaseContext(context);
		free(kernel_src);

		pi = (*area)/n;
		free(area);
		return (pi);
}
