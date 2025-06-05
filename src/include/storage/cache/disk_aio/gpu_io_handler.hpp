#ifndef GPU_IO_HANDLER_H
#define GPU_IO_HANDLER_H

#include "storage/cache/common.h"
#include <cufile.h>
#include <sys/stat.h>

class GpuIoHandler {
    public:
    GpuIoHandler() : file_descriptor(-1), is_reserved(false), delete_when_close(false) {
        file_mmap = NULL;
        cu_file_handle = NULL;
    }

    ~GpuIoHandler() {
        if (cu_file_handle) {
            cuFileClose(cu_file_handle);
        }
    }

    // Initialize GDS
    bool InitGds() {
        CUfileError_t err = cuFileDriverOpen();
        if (err != CU_FILE_SUCCESS) {
            return false;
        }
        return true;
    }

    // Register file for GDS
    bool RegisterFile(int fd) {
        if (cu_file_handle) {
            cuFileClose(cu_file_handle);
        }
        
        CUfileError_t err = cuFileHandleRegister(&cu_file_handle, fd);
        if (err != CU_FILE_SUCCESS) {
            return false;
        }
        file_descriptor = fd;
        return true;
    }

    // Synchronous read from file to GPU memory
    bool Read(void* gpu_buffer, size_t size, off_t file_offset, off_t buffer_offset) {
        if (!cu_file_handle) return false;

        size_t read_size = size;
        int bytes_read;
        CUfileError_t err = cuFileRead(cu_file_handle, 
                                     gpu_buffer,
                                     &read_size,
                                     &file_offset,
                                     &buffer_offset,
                                     &bytes_read);
        return (err == CU_FILE_SUCCESS);
    }

    // Synchronous write from GPU memory to file
    bool Write(void* gpu_buffer, size_t size, off_t file_offset, off_t buffer_offset) {
        if (!cu_file_handle) return false;

        size_t write_size = size;
        int bytes_written;
        CUfileError_t err = cuFileWrite(cu_file_handle,
                                      gpu_buffer,
                                      &write_size,
                                      &file_offset,
                                      &buffer_offset,
                                      &bytes_written);
        return (err == CU_FILE_SUCCESS);
    }

    // Get file size
    size_t GetFileSize() {
        if (!cu_file_handle) return 0;
        
        struct stat st;
        if (fstat(file_descriptor, &st) != 0) {
            return 0;
        }
        return st.st_size;
    }

    private:
    int file_descriptor;
    bool is_reserved;
    bool delete_when_close;
    void* file_mmap;
    CUfileHandle_t cu_file_handle;  // GDS file handle
};

#endif
