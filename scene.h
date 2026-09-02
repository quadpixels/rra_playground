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

struct SceneBlasBvhNode
{
    std::vector<uint32_t> children;
    uint32_t              primitive_start{0};
    uint32_t              primitive_count{0};
    bool                  is_leaf{false};
};

struct SceneBlasTopology
{
    std::vector<SceneBlasBvhNode> nodes;
    uint32_t                      root_index{0};
};

struct SceneTlasNode
{
    std::vector<uint32_t> children;
    uint32_t              instance_index{0};
    bool                  is_leaf{false};
};

struct SceneTlasTopology
{
    std::vector<SceneTlasNode> nodes;
    uint32_t                   root_index{0};
    bool                       valid{false};
};

struct SceneRay
{
    glm::vec3 origin{0.0f};
    float     tmin{0.001f};
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    float     tmax{10000.0f};
    uint32_t  ray_flags{0};
    uint32_t  instance_inclusion_mask{0xFF};
    uint32_t  sbt_record_offset{0};
    uint32_t  sbt_record_stride{0};
    uint32_t  miss_index{0};
};

struct SceneDispatchRays
{
    std::vector<SceneRay> rays;
    std::vector<uint32_t> ray_offsets;
    glm::uvec3            dispatch_dims{0};
    std::string           name;
    uint32_t              num_invocations{0};
};

struct SceneData
{
    std::vector<std::vector<glm::vec3>> blas_vertices;
    std::vector<SceneBlasTopology>      blas_topologies;
    std::vector<SceneInstance>          instances;
    SceneTlasTopology                   tlas_topology;
    std::vector<SceneDispatchRays>      dispatches;
    SceneCamera                         camera;
    SceneStats                          stats;
};

bool LoadSceneFromRra(const char* rra_file_name, AppState* app_state, SceneData* out_scene);
SceneData BuildFallbackCubeScene();
