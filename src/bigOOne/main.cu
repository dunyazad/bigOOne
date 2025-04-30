#include "Essential.h"

#define INIT_WIDTH 3840
#define INIT_HEIGHT 2160

int width = INIT_WIDTH;
int height = INIT_HEIGHT;

GLuint tex = 0;
cudaGraphicsResource_t cuda_tex_res = nullptr;
uchar4* dev_ptr = nullptr;

// --------------------------------------------
// CUDA Error Checking
// --------------------------------------------
void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// --------------------------------------------
// CUDA Kernel
// --------------------------------------------
__global__ void fillKernel(uchar4* ptr, int w, int h, unsigned char xOffset, unsigned char yOffset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    ptr[idx] = make_uchar4((x + xOffset) % 256, (y + yOffset) % 256, 128, 255);
}

// --------------------------------------------
// Resize Resources
// --------------------------------------------
void resizeResources(int newWidth, int newHeight) {
    if (cuda_tex_res) {
        checkCuda(cudaGraphicsUnregisterResource(cuda_tex_res));
        cuda_tex_res = nullptr;
    }
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
    if (dev_ptr) {
        checkCuda(cudaFree(dev_ptr));
        dev_ptr = nullptr;
    }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, newWidth, newHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    checkCuda(cudaGraphicsGLRegisterImage(&cuda_tex_res, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
    checkCuda(cudaMalloc(&dev_ptr, newWidth * newHeight * sizeof(uchar4)));
}

// --------------------------------------------
// GLFW Resize Callback
// --------------------------------------------
void framebuffer_size_callback(GLFWwindow*, int newWidth, int newHeight) {
    glViewport(0, 0, newWidth, newHeight);
    width = newWidth;
    height = newHeight;
    resizeResources(width, height);
}

// --------------------------------------------
// CUDA Texture Fill
// --------------------------------------------
void fillTextureWithCUDA(unsigned char xOffset, unsigned char yOffset) {
    if (!cuda_tex_res || !dev_ptr) return;

    cudaArray_t array;
    checkCuda(cudaGraphicsMapResources(1, &cuda_tex_res));
    checkCuda(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_res, 0, 0));

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);
    fillKernel << <grid, block >> > (dev_ptr, width, height, xOffset, yOffset);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy2DToArray(array, 0, 0, dev_ptr, width * sizeof(uchar4),
        width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice));
    checkCuda(cudaGraphicsUnmapResources(1, &cuda_tex_res));
}

// --------------------------------------------
// Shader Sources
// --------------------------------------------
const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D tex;
void main() {
    FragColor = texture(tex, TexCoord);
}
)";

// --------------------------------------------
// Compile Shader
// --------------------------------------------
GLuint createShaderProgram() {
    auto compile = [](const char* src, GLenum type) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compile error:\n" << infoLog << std::endl;
        }
        return shader;
    };

    GLuint vs = compile(vertexShaderSrc, GL_VERTEX_SHADER);
    GLuint fs = compile(fragmentShaderSrc, GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// --------------------------------------------
// Create Quad VAO
// --------------------------------------------
GLuint createFullscreenQuad() {
    float vertices[] = {
        -1, -1, 0, 0,
         1, -1, 1, 0,
         1,  1, 1, 1,
        -1,  1, 0, 1
    };

    unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    return VAO;
}

// --------------------------------------------
// Main Entry
// --------------------------------------------
int main() {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width / 2, height / 2, "CUDA + OpenGL", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0); // VSync off

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    resizeResources(width, height); // Initial texture/cuda alloc
    GLuint program = createShaderProgram();
    GLuint quadVAO = createFullscreenQuad();

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_time).count();
        if (elapsed >= 1.0f) {
            std::string title = "CUDA OpenGL - " + std::to_string(frame_count) + " FPS";
            glfwSetWindowTitle(window, title.c_str());
            frame_count = 0;
            last_time = now;
        }

        fillTextureWithCUDA((unsigned char)(elapsed * 255.0f), (unsigned char)(elapsed * 255.0f));

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(quadVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);
    }

    checkCuda(cudaFree(dev_ptr));
    checkCuda(cudaGraphicsUnregisterResource(cuda_tex_res));
    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
