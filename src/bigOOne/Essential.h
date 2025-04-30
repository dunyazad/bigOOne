#pragma once

#pragma warning(disable : 4098)

#include <chrono>
#include <iostream>
#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#pragma comment (lib, "glfw3.lib")

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include "helper_math.h"

struct States
{
	unsigned int windowWidth = 3840;
	unsigned int windowHeight = 2160;

    unsigned int textureWidth = 3840;
    unsigned int textureHeight = 2160;
};

class bigOOne
{
public:
    // Delete copy constructor and assignment operator to enforce singleton
    bigOOne(const bigOOne&) = delete;
    bigOOne& operator=(const bigOOne&) = delete;

    // Accessor to the singleton instance
    static bigOOne& getInstance()
    {
        static bigOOne instance; // Guaranteed to be thread-safe in C++11 and later
        return instance;
    }

    // Public member functions here
    void doSomething() {}

    States states;
private:
    // Private constructor to prevent external instantiation
    bigOOne() {}
};

#define O1 bigOOne::getInstance()
