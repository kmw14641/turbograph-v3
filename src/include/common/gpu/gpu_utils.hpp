#pragma once

static std::string makeNvrtcArchFlag_()
{
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         dev);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "--gpu-architecture=compute_%d%d", major,
                  minor);
    return std::string(buf);
}