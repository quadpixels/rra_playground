#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include <glm/glm.hpp>

enum class SceneLoadStage
{
    kIdle,
    kLoadingTrace,
    kExtractingDispatchRays,
    kExtractingBlas,
    kExtractingTlas,
    kBuildingGpuBlas,
    kBuildingGpuTlas,
    kBuildingCpuBvh,
    kReady,
    kFailed,
};

struct SceneStats
{
    std::string source_name;
    std::string camera_preset;
    uint64_t    tlas_count{0};
    uint64_t    blas_count{0};
    uint64_t    instance_count{0};
    uint64_t    total_triangle_count{0};
    glm::vec3   scene_aabb_min{0.0f};
    glm::vec3   scene_aabb_max{0.0f};
};

struct CpuRenderStats
{
    uint64_t rays{0};
    uint64_t bvh_steps{0};
    uint64_t box_nodes{0};
    uint64_t ray_box_tests{0};
    uint64_t tri_nodes{0};
    uint64_t ray_triangle_tests{0};
    uint64_t tlas_to_blas{0};
};

struct AppState
{
    std::atomic<SceneLoadStage> scene_stage{SceneLoadStage::kIdle};
    std::atomic<uint32_t>       blas_completed{0};
    std::atomic<uint32_t>       blas_total{0};
    std::atomic<uint32_t>       tlas_completed{0};
    std::atomic<uint32_t>       tlas_total{0};
    std::atomic<uint32_t>       dispatch_completed{0};
    std::atomic<uint32_t>       dispatch_total{0};
    std::atomic<bool>           as_built{false};
    std::atomic<bool>           scene_loaded{false};

    std::mutex   details_mutex;
    SceneStats   scene_stats;
    CpuRenderStats cpu_render_stats;
    std::string  last_error;
    std::string  status_line;
    float        last_gpu_frame_ms{0.0f};
    float        last_gpu_compact_dispatch_rays_ms{0.0f};
    uint64_t     cpu_bvh_node_count{0};
    uint64_t     cpu_bvh_primitive_count{0};
    uint32_t     cpu_bvh_fanout{2};
    uint32_t     cpu_bvh_primitive_node_triangle_capacity{3};
    bool         cpu_bvh_uses_rra_topology{false};
    bool         cpu_bvh_uses_binned_sah{false};

    void SetStatus(std::string value)
    {
        std::lock_guard<std::mutex> lock(details_mutex);
        status_line = std::move(value);
    }

    void SetError(std::string value)
    {
        scene_stage.store(SceneLoadStage::kFailed);
        std::lock_guard<std::mutex> lock(details_mutex);
        last_error = std::move(value);
    }

    void SetSceneStats(const SceneStats& value)
    {
        std::lock_guard<std::mutex> lock(details_mutex);
        scene_stats = value;
    }

    void SetCpuRenderStats(const CpuRenderStats& value)
    {
        std::lock_guard<std::mutex> lock(details_mutex);
        cpu_render_stats = value;
    }
};
