#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include <glm/glm.hpp>

#include "app_state.h"
#include "scene.h"

enum class RenderBackend
{
    kDxr = 0,
    kCpuBruteForce = 1,
    kCpuBvh = 2,
};

enum class CpuBvhBuildMode
{
    kWideMedian = 0,
    kTranscribedRra = 1,
};

enum class CpuBvhSplitMode
{
    kEqualCounts = 0,
    kBinnedSah = 1,
};

struct CpuBvhSettings
{
    uint32_t        fanout{2};
    CpuBvhBuildMode build_mode{CpuBvhBuildMode::kWideMedian};
    CpuBvhSplitMode split_mode{CpuBvhSplitMode::kEqualCounts};
    uint32_t        primitive_node_triangle_capacity{3};
};

struct CpuRay
{
    glm::vec3 origin{0.0f};
    float     tmin{0.001f};
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    float     tmax{10000.0f};
    uint32_t  ray_flags{0};
    uint32_t  instance_inclusion_mask{0xFF};
};

struct CpuRenderRequest
{
    uint32_t          width{0};
    uint32_t          height{0};
    uint32_t          thread_count{1};
    glm::mat4         inverse_view{1.0f};
    glm::mat4         inverse_proj{1.0f};
    bool              invert_y{false};
    bool              use_external_rays{false};
    RenderBackend     backend{RenderBackend::kCpuBruteForce};
    uint64_t          generation{0};
    bool              rebuild_bvh{false};
    bool              render_after_build{true};
    CpuBvhSettings    bvh_settings{};
    std::vector<CpuRay> external_rays;
    std::vector<uint32_t> external_ray_offsets;
};

struct CpuRenderResult
{
    uint32_t             width{0};
    uint32_t             height{0};
    uint32_t             tiles_completed{0};
    uint32_t             tiles_total{0};
    uint64_t             generation{0};
    bool                 complete{false};
    RenderBackend        backend{RenderBackend::kCpuBruteForce};
    CpuRenderStats       stats{};
    std::vector<uint8_t> rgba;
};

class CpuPrimaryRayRenderer
{
public:
    struct Aabb
    {
        glm::vec3 min{0.0f};
        glm::vec3 max{0.0f};
    };

    struct InstanceData
    {
        uint32_t mesh_index{0};
        glm::mat4 object_to_world{1.0f};
        glm::mat4 world_to_object{1.0f};
        Aabb     world_bounds{};
    };

    struct BuildRef
    {
        uint32_t payload{0};
        Aabb     bounds{};
        glm::vec3 centroid{0.0f};
    };

    struct BvhNode
    {
        enum class Kind
        {
            kInternal,
            kTriangle,
        };

        enum class SubnodeType
        {
            kEmpty,
            kBox,
            kTriangle,
            kInstance,
            kProcedural,
        };

        struct ChildNode
        {
            SubnodeType type{SubnodeType::kEmpty};
            Aabb        aabb{};
            uint32_t    index{0};
        };

        struct TrianglePrimitive
        {
            glm::vec3 v0{0.0f};
            glm::vec3 v1{0.0f};
            glm::vec3 v2{0.0f};
            uint32_t  first_vertex{0};
        };

        Kind                       kind{Kind::kInternal};
        Aabb                       bounds{};
        std::array<ChildNode, 16>  children{};
        uint32_t                   child_count{0};
        std::array<TrianglePrimitive, 3> triangles{};
        uint32_t                   triangle_count{0};

        bool IsLeaf() const
        {
            return kind != Kind::kInternal;
        }
    };

    struct CpuBlas
    {
        std::vector<BuildRef> primitive_refs;
        std::vector<BvhNode>  nodes;
        uint32_t              root_index{0};
    };

    void BuildFromScene(const SceneData& scene, const CpuBvhSettings& settings, AppState* app_state);
    void Render(const CpuRenderRequest& request,
                CpuRenderResult* out_result,
                const std::function<void(const CpuRenderResult&)>& publish_partial = {});
    bool IsReady() const;

    std::vector<std::vector<glm::vec3>> mesh_vertices_;
    std::vector<InstanceData>           instances_;
    std::vector<CpuBlas>                blases_;
    std::vector<BuildRef>               tlas_refs_;
    std::vector<BvhNode>                tlas_nodes_;
    uint32_t                            tlas_root_index_{0};
    uint32_t                            bvh_fanout_{2};
    uint32_t                            primitive_node_triangle_capacity_{3};
    bool                                bvh_uses_rra_topology_{false};
    bool                                bvh_uses_binned_sah_{false};

    static glm::vec3 TransformPosition(const glm::mat4& m, const glm::vec3& x);
    static glm::vec3 TransformDirection(const glm::mat4& m, const glm::vec3& x);
};
