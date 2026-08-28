#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "app_state.h"

struct SceneInstance
{
    uint64_t blas_idx{};
    float    transform[12]{};
};

struct SceneCamera
{
    glm::vec3 eye{0.0f};
    glm::vec3 center{0.0f, 1.0f, 0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    bool      invert_y{false};
    std::string preset_name;
};

struct SceneData
{
    std::vector<std::vector<glm::vec3>> blas_vertices;
    std::vector<SceneInstance>          instances;
    SceneCamera                         camera;
    SceneStats                          stats;
};

bool LoadSceneFromRra(const char* rra_file_name, AppState* app_state, SceneData* out_scene);
SceneData BuildFallbackCubeScene();
