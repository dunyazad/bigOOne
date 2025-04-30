#include "Essential.h"

GLuint tex = 0;
cudaGraphicsResource_t cuda_tex_res = nullptr;
uchar4* dev_ptr = nullptr;

// ------------------------------------------------
// CUDA Error Checking
// ------------------------------------------------
void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// ------------------------------------------------
// SDF & Ray March CUDA Kernel
// ------------------------------------------------
__device__ float sdf_sphere(float3 p, float3 center, float radius) {
    return length(p - center) - radius;
}

__device__ float sdf_plane(float3 p, float3 normal, float d) {
    return dot(p, normal) + d;
}

__device__ float scene_sdf(float3 p) {
    float d_sphere = sdf_sphere(p, make_float3(0.0f, 0.5f, 3.0f), 0.5f);
    float d_plane = sdf_plane(p, make_float3(0.0f, 1.0f, 0.0f), 0.0f);
    return fminf(d_sphere, d_plane);
}

__device__ float3 getNormal(float3 p) {
    float eps = 0.001f;
    return normalize(make_float3(
        scene_sdf(p + make_float3(eps, 0, 0)) - scene_sdf(p - make_float3(eps, 0, 0)),
        scene_sdf(p + make_float3(0, eps, 0)) - scene_sdf(p - make_float3(0, eps, 0)),
        scene_sdf(p + make_float3(0, 0, eps)) - scene_sdf(p - make_float3(0, 0, eps))
    ));
}

__device__ uchar4 shadePixel(float3 ro, float3 rd, float t) {
    float3 color = make_float3(0.0f);

    if (t > 0.0f) {
        float3 p = ro + t * rd;
        float3 n = getNormal(p);
        float3 lightDir = normalize(make_float3(-0.5f, 1.0f, -0.5f));
        float diff = fmaxf(dot(n, lightDir), 0.0f);

        color = make_float3(0.4f, 0.6f, 1.0f) * diff;
    }

    return make_uchar4(color.x * 255, color.y * 255, color.z * 255, 255);
}

__device__ float ray_march(float3 ro, float3 rd, float max_dist = 20.0f, int max_steps = 128) {
    float t = 0.0f;
    for (int i = 0; i < max_steps; ++i) {
        float3 p = ro + t * rd;
        float dist = scene_sdf(p);
        if (dist < 0.001f) return t;
        t += dist;
        if (t > max_dist) break;
    }
    return -1.0f;
}

__global__ void sdfRaymarchKernel(uchar4* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;

    float u = (float)x / (float)w * 2.0f - 1.0f;
    float v = (float)y / (float)h * 2.0f - 1.0f;

    float3 ro = make_float3(0.0f, 1.0f, -3.0f);
    float3 rd = normalize(make_float3(u, -v, 1.5f));

    float t = ray_march(ro, rd);
    output[idx] = shadePixel(ro, rd, t);
}

// ------------------------------------------------
// CUDA Texture Fill
// ------------------------------------------------
void fillTextureWithCUDA() {
    if (!cuda_tex_res || !dev_ptr) return;

    cudaArray_t array;
    checkCuda(cudaGraphicsMapResources(1, &cuda_tex_res));
    checkCuda(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex_res, 0, 0));

    dim3 block(16, 16);
    dim3 grid((O1.states.textureWidth + 15) / 16, (O1.states.textureHeight + 15) / 16);
    sdfRaymarchKernel << <grid, block >> > (dev_ptr, O1.states.textureWidth, O1.states.textureHeight);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy2DToArray(array, 0, 0, dev_ptr, O1.states.textureWidth * sizeof(uchar4),
        O1.states.textureWidth * sizeof(uchar4), O1.states.textureHeight, cudaMemcpyDeviceToDevice));
    checkCuda(cudaGraphicsUnmapResources(1, &cuda_tex_res));
}

// ------------------------------------------------
// Resize Resources
// ------------------------------------------------
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

// ------------------------------------------------
// GLFW Resize Callback
// ------------------------------------------------
void framebuffer_size_callback(GLFWwindow* w, int newWidth, int newHeight) {
    glViewport(0, 0, newWidth, newHeight);
    O1.states.windowWidth = newWidth;
    O1.states.windowHeight = newHeight;
    O1.states.textureWidth = newWidth;
    O1.states.textureHeight = newHeight;
    resizeResources(newWidth, newHeight);
}

// ------------------------------------------------
// Shaders
// ------------------------------------------------
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

// ------------------------------------------------
// Fullscreen Quad
// ------------------------------------------------
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

// ------------------------------------------------
// Main Entry
// ------------------------------------------------
int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(O1.states.windowWidth, O1.states.windowHeight, "CUDA SDF Raymarch", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0); // Disable VSync
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    glViewport(0, 0, O1.states.windowWidth, O1.states.windowHeight);
    resizeResources(O1.states.textureWidth, O1.states.textureHeight);
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
            std::string title = "CUDA SDF - " + std::to_string(frame_count) + " FPS";
            glfwSetWindowTitle(window, title.c_str());
            frame_count = 0;
            last_time = now;
        }

        fillTextureWithCUDA();

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
