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
    kExtractingBlas,
    kExtractingTlas,
    kBuildingGpuBlas,
    kBuildingGpuTlas,
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

struct AppState
{
    std::atomic<SceneLoadStage> scene_stage{SceneLoadStage::kIdle};
    std::atomic<uint32_t>       blas_completed{0};
    std::atomic<uint32_t>       blas_total{0};
    std::atomic<uint32_t>       tlas_completed{0};
    std::atomic<uint32_t>       tlas_total{0};
    std::atomic<bool>           as_built{false};
    std::atomic<bool>           scene_loaded{false};

    std::mutex   details_mutex;
    SceneStats   scene_stats;
    std::string  last_error;
    std::string  status_line;
    float        last_gpu_frame_ms{0.0f};

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
};
