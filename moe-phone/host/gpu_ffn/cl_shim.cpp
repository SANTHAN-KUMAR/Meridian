// cl_shim.cpp — the OpenCL entry points the test/bench programs call, forwarded to the device's own
// libOpenCL.so through dlopen. The NDK ships no OpenCL import library; zcbench solved the same problem
// the same way. Linked into the Android builds of gx_test, gx_pool_test and gx_bench only; libgx itself
// (gx.cpp) has its own loader and does not need this.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void * sym(const char * name) {
    static void * h = nullptr;
    if (!h) {
        const char * libs[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "libOpenCL.so.1"};
        for (const char * l : libs)
            if ((h = dlopen(l, RTLD_NOW | RTLD_LOCAL))) break;
        if (!h) { fprintf(stderr, "cl_shim: cannot dlopen libOpenCL\n"); abort(); }
    }
    void * f = dlsym(h, name);
    if (!f) { fprintf(stderr, "cl_shim: missing %s\n", name); abort(); }
    return f;
}

#define FWD(ret, name, params, args)                                  \
    extern "C" CL_API_ENTRY ret CL_API_CALL name params {             \
        static auto fwd_ = (ret(CL_API_CALL *) params) sym(#name);    \
        return fwd_ args;                                             \
    }

FWD(cl_int, clGetPlatformIDs, (cl_uint n, cl_platform_id * p, cl_uint * np), (n, p, np))
FWD(cl_int, clGetDeviceIDs, (cl_platform_id p, cl_device_type t, cl_uint n, cl_device_id * d, cl_uint * nd), (p, t, n, d, nd))
FWD(cl_int, clGetDeviceInfo, (cl_device_id d, cl_device_info i, size_t s, void * v, size_t * r), (d, i, s, v, r))
FWD(cl_context, clCreateContext,
    (const cl_context_properties * p, cl_uint n, const cl_device_id * d,
     void(CL_CALLBACK * cb)(const char *, const void *, size_t, void *), void * u, cl_int * e),
    (p, n, d, cb, u, e))
FWD(cl_command_queue, clCreateCommandQueue, (cl_context c, cl_device_id d, cl_command_queue_properties p, cl_int * e), (c, d, p, e))
FWD(cl_program, clCreateProgramWithSource, (cl_context c, cl_uint n, const char ** s, const size_t * l, cl_int * e), (c, n, s, l, e))
FWD(cl_int, clBuildProgram,
    (cl_program p, cl_uint n, const cl_device_id * d, const char * o, void(CL_CALLBACK * cb)(cl_program, void *), void * u),
    (p, n, d, o, cb, u))
FWD(cl_int, clGetProgramBuildInfo, (cl_program p, cl_device_id d, cl_program_build_info i, size_t s, void * v, size_t * r),
    (p, d, i, s, v, r))
FWD(cl_kernel, clCreateKernel, (cl_program p, const char * n, cl_int * e), (p, n, e))
FWD(cl_mem, clCreateBuffer, (cl_context c, cl_mem_flags f, size_t s, void * h, cl_int * e), (c, f, s, h, e))
FWD(cl_int, clSetKernelArg, (cl_kernel k, cl_uint i, size_t s, const void * v), (k, i, s, v))
FWD(cl_int, clEnqueueNDRangeKernel,
    (cl_command_queue q, cl_kernel k, cl_uint d, const size_t * o, const size_t * g, const size_t * l, cl_uint n,
     const cl_event * w, cl_event * ev),
    (q, k, d, o, g, l, n, w, ev))
FWD(cl_int, clEnqueueReadBuffer,
    (cl_command_queue q, cl_mem b, cl_bool bl, size_t o, size_t s, void * p, cl_uint n, const cl_event * w, cl_event * ev),
    (q, b, bl, o, s, p, n, w, ev))
FWD(void *, clEnqueueMapBuffer,
    (cl_command_queue q, cl_mem b, cl_bool bl, cl_map_flags f, size_t o, size_t s, cl_uint n, const cl_event * w, cl_event * ev,
     cl_int * e),
    (q, b, bl, f, o, s, n, w, ev, e))
FWD(cl_int, clEnqueueUnmapMemObject, (cl_command_queue q, cl_mem m, void * p, cl_uint n, const cl_event * w, cl_event * ev),
    (q, m, p, n, w, ev))
FWD(cl_int, clFinish, (cl_command_queue q), (q))
FWD(cl_int, clReleaseMemObject, (cl_mem m), (m))
FWD(cl_int, clReleaseKernel, (cl_kernel k), (k))
FWD(cl_int, clReleaseProgram, (cl_program p), (p))
