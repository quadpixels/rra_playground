#include "scene.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <map>

#include "public/rra_blas.h"
#include "public/rra_bvh.h"
#include "public/rra_ray_history.h"
#include "public/rra_tlas.h"
#include "public/rra_trace_loader.h"

namespace
{
struct CamParams
{
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    bool      invert_y;
};

const std::map<std::string, CamParams> kCameraParams = {
    {"SolarBay", {glm::vec3(5.964f, 1.691f, 5.374f), glm::vec3(2.921f, 1.691f, 2.120f), glm::vec3(0, 1, 0), false}},
    {"PortRoyal", {glm::vec3(-7.2252469f, 0.8361527f, 25.2023430f), glm::vec3(-6.8860960f, 0.8613553f, 24.2284565f), glm::vec3(0, 1, 0), true}},
    {"DXRFeatureTest", {glm::vec3(-6.1447086f, 2.7448003f, -11.9588842f), glm::vec3(-6.1102533f, 2.7394657f, -11.9192486f), glm::vec3(0, 1, 0), true}},
    {"Cyberpunk2077", {glm::vec3(667.6618652f, -804.2122192f, 128.7313995f), glm::vec3(666.0505371f, -802.7095947f, 128.0240326f), glm::vec3(0, 0, 1), true}},
    {"RealTimeDenoisedAmbientOcclusion", {glm::vec3(-43.5119209f, 24.3670177f, -29.0387344f), glm::vec3(-43.2385712f, 24.1981163f, -28.8011036f), glm::vec3(0, 1, 0), true}},
    {"b1-Win64-Shipping", {glm::vec3(-32269.8417969f, 9393.68f, -1515.189f), glm::vec3(-32869.87f, 9697.102f, -1436.413f), glm::vec3(0, 0, 1), true}},
    {"VictorStones", {glm::vec3(54.20388f, -360.680725f, 20.8701935f), glm::vec3(93.000f, -316.9831f, 29.51616f), glm::vec3(0, 0, 1), true}},
    {"AncientGame", {glm::vec3(-290.0213013f, 230.9532928f, 341.0099792f), glm::vec3(-344.7103882f, 227.3368835f, 347.0361633f), glm::vec3(0, 0, 1), true}},
    {"ThreeTriangles", {glm::vec3(60.1231461f, 90.5544434f, 25.7323875f), glm::vec3(60.0977364f, 86.9188461f, 25.4270802f), glm::vec3(0, 0, 1), true}},
};

void SetStatus(AppState* app_state, const std::string& text)
{
    if (app_state != nullptr)
    {
        app_state->SetStatus(text);
    }
}

SceneCamera ChooseCameraPreset(const std::string& file_name)
{
    SceneCamera camera;
    for (const auto& entry : kCameraParams)
    {
        if (file_name.find(entry.first) != std::string::npos)
        {
            camera.eye         = entry.second.eye;
            camera.center      = entry.second.center;
            camera.up          = entry.second.up;
            camera.invert_y    = entry.second.invert_y;
            camera.preset_name = entry.first;
            return camera;
        }
    }
    return camera;
}
}  // namespace

bool LoadSceneFromRra(const char* rra_file_name, AppState* app_state, SceneData* out_scene)
{
    if (out_scene == nullptr)
    {
        return false;
    }

    if (!std::filesystem::exists(rra_file_name))
    {
        if (app_state != nullptr)
        {
            app_state->SetError(std::string(rra_file_name) + " does not exist.");
        }
        return false;
    }

    if (app_state != nullptr)
    {
        app_state->scene_stage.store(SceneLoadStage::kLoadingTrace);
        app_state->scene_loaded.store(false);
    }
    SetStatus(app_state, "Loading RRA trace");

    const RraErrorCode ec = RraTraceLoaderLoad(rra_file_name);
    std::printf("Error: %d\n", static_cast<int>(ec));
    if (ec != kRraOk)
    {
        if (app_state != nullptr)
        {
            app_state->SetError("RraTraceLoaderLoad failed.");
        }
        std::printf("Error encountered, quitting.\n");
        return false;
    }

    SceneData scene;
    scene.stats.source_name = rra_file_name;

    time_t ct = RraTraceLoaderGetCreateTime();
    std::tm tm;
    localtime_s(&tm, &ct);
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%a, %Y-%m-%d %H:%M:%S", &tm);
    std::printf("Trace create time: %s\n", time_buffer);

    uint64_t tlas_count = 0;
    uint64_t blas_count = 0;
    RraBvhGetTlasCount(&tlas_count);
    RraBvhGetBlasCount(&blas_count);
    std::printf("Trace has %llu TLASs and %llu BLASs\n", tlas_count, blas_count);

    scene.stats.tlas_count = tlas_count;
    scene.stats.blas_count = blas_count;
    if (app_state != nullptr)
    {
        app_state->blas_total.store(static_cast<uint32_t>(blas_count + 1));
        app_state->tlas_total.store(static_cast<uint32_t>(std::min<uint64_t>(1, tlas_count)));
    }

    for (unsigned i = 1; i <= blas_count; i++)
    {
        uint32_t cnt = 0;
        uint32_t cnt1 = 0;
        uint32_t cnt2 = 0;
        uint32_t cnt3 = 0;
        uint64_t addr = 0;
        RraBlasGetGeometryCount(i, &cnt);
        RraBlasGetProceduralNodeCount(i, &cnt1);
        RraBlasGetTriangleNodeCount(i, &cnt2);
        RraBlasGetUniqueTriangleCount(i, &cnt3);
        RraBlasGetBaseAddress(i, &addr);
        std::printf("  BLAS[%u] (%llx) has %u geometries, %u proc nodes, %u tri nodes, %u uniq tris\n", i, addr, cnt, cnt1, cnt2, cnt3);
    }

    {
        uint32_t dispatch_count = 0;
        RraRayGetDispatchCount(&dispatch_count);
        std::printf("dispatch_count=%u\n", dispatch_count);

        for (uint32_t d = 0; d < dispatch_count; d++)
        {
            uint32_t x = 0;
            uint32_t y = 0;
            uint32_t z = 0;
            if (RraRayGetDispatchDimensions(d, &x, &y, &z) != kRraOk)
            {
                continue;
            }
            std::printf("  dispatch[%u], dim=(%u,%u,%u)\n", d, x, y, z);
        }
    }

    scene.blas_vertices.reserve(blas_count + 1);
    uint32_t total_tri_count = 0;
    scene.stats.scene_aabb_min = glm::vec3(1e20f);
    scene.stats.scene_aabb_max = glm::vec3(-1e20f);

    if (app_state != nullptr)
    {
        app_state->scene_stage.store(SceneLoadStage::kExtractingBlas);
    }
    for (unsigned i = 0; i <= blas_count; i++)
    {
        SetStatus(app_state, "Extracting BLAS triangles");

        std::vector<glm::vec3> geom_verts;

        uint32_t root_node = 0;
        RraBvhGetRootNodePtr(&root_node);
        std::deque<uint32_t> nodes_to_visit = {root_node};

        if (i > 0)
        {
            float sa = 0.0f;
            RraBlasGetSurfaceArea(i, root_node, &sa);
            if (sa <= 0.0f)
            {
                throw std::exception();
            }
        }

        uint32_t num_tris = 0;
        while (!nodes_to_visit.empty())
        {
            const uint32_t node = nodes_to_visit.front();
            nodes_to_visit.pop_front();

            uint32_t child_count = 0;
            RraBlasGetChildNodeCount(i, node, &child_count);
            std::vector<uint32_t> children(child_count);
            RraBlasGetChildNodes(i, node, children.data());

            for (uint32_t child : children)
            {
                if (RraBvhIsBoxNode(child))
                {
                    nodes_to_visit.push_back(child);
                    continue;
                }
                if (!RraBvhIsTriangleNode(child))
                {
                    continue;
                }

                float sa = 0.0f;
                RraBlasGetSurfaceArea(i, child, &sa);
                if (sa <= 0.0f)
                {
                    std::printf("BLAS[%u]'s node %08X's surface area is zero\n", i, child);
                }

                uint32_t tri_count = 0;
                if (RraBlasGetNodeTriangleCount(i, child, &tri_count) != kRraOk)
                {
                    continue;
                }

                std::vector<VertexPosition> verts(tri_count == 1 ? 3 : 4);
                if (RraBlasGetNodeVertices(i, child, verts.data()) != kRraOk)
                {
                    continue;
                }

                if (sa > 0.0f)
                {
                    num_tris += tri_count;
                    if (tri_count >= 1)
                    {
                        geom_verts.push_back(glm::vec3(verts[0].x, verts[0].y, verts[0].z));
                        geom_verts.push_back(glm::vec3(verts[1].x, verts[1].y, verts[1].z));
                        geom_verts.push_back(glm::vec3(verts[2].x, verts[2].y, verts[2].z));
                    }
                    if (tri_count >= 2)
                    {
                        geom_verts.push_back(glm::vec3(verts[1].x, verts[1].y, verts[1].z));
                        geom_verts.push_back(glm::vec3(verts[3].x, verts[3].y, verts[3].z));
                        geom_verts.push_back(glm::vec3(verts[2].x, verts[2].y, verts[2].z));
                    }
                }
            }
        }

        total_tri_count += num_tris;
        scene.blas_vertices.push_back(std::move(geom_verts));
        if (app_state != nullptr)
        {
            app_state->blas_completed.store(i + 1);
        }
    }
    scene.stats.total_triangle_count = total_tri_count;

    if (tlas_count > 1)
    {
        std::printf("%zu TLAS detected. Will only make use of the first TLAS.\n", static_cast<size_t>(tlas_count));
    }

    if (app_state != nullptr)
    {
        app_state->scene_stage.store(SceneLoadStage::kExtractingTlas);
    }
    for (unsigned i = 0; i < std::min(1, static_cast<int>(tlas_count)); i++)
    {
        SetStatus(app_state, "Extracting TLAS instances");

        uint64_t node_count = 0;
        uint32_t inst_count = 0;
        RraTlasGetBoxNodeCount(i, &node_count);

        uint32_t root_node = 0;
        RraBvhGetRootNodePtr(&root_node);
        std::deque<uint32_t> nodes_to_visit = {root_node};
        std::vector<SceneInstance> instance_infos;

        while (!nodes_to_visit.empty())
        {
            const uint32_t node = nodes_to_visit.front();
            nodes_to_visit.pop_front();

            uint32_t child_count = 0;
            RraTlasGetChildNodeCount(i, node, &child_count);
            std::vector<uint32_t> children(child_count);
            RraTlasGetChildNodes(i, node, children.data());

            for (uint32_t child : children)
            {
                if (RraBvhIsBoxNode(child))
                {
                    nodes_to_visit.push_back(child);
                    continue;
                }
                if (!RraBvhIsInstanceNode(child))
                {
                    continue;
                }

                SceneInstance instance_info{};
                RraTlasGetOriginalInstanceNodeTransform(i, child, instance_info.transform);
                RraTlasGetBlasIndexFromInstanceNode(i, child, &(instance_info.blas_idx));

                uint32_t instance_index = 0;
                RraTlasGetInstanceIndexFromInstanceNode(i, child, &instance_index);
                if (instance_infos.size() < instance_index + 1)
                {
                    instance_infos.resize(instance_index + 1);
                }
                instance_infos[instance_index] = instance_info;

                const std::vector<glm::vec3>& verts = scene.blas_vertices.at(instance_info.blas_idx);
                for (const glm::vec3& p : verts)
                {
                    const float* t = instance_info.transform;
                    glm::vec3 vt{};
                    vt.x = t[3] + t[0] * p.x + t[1] * p.y + t[2] * p.z;
                    vt.y = t[7] + t[4] * p.x + t[5] * p.y + t[6] * p.z;
                    vt.z = t[11] + t[8] * p.x + t[9] * p.y + t[10] * p.z;
                    scene.stats.scene_aabb_min = glm::min(scene.stats.scene_aabb_min, vt);
                    scene.stats.scene_aabb_max = glm::max(scene.stats.scene_aabb_max, vt);
                }
            }
        }

        inst_count            = static_cast<uint32_t>(instance_infos.size());
        scene.stats.instance_count = inst_count;
        scene.instances       = std::move(instance_infos);
        std::printf("TLAS %u: %llu nodes, %u insts\n", i, node_count, inst_count);
        if (app_state != nullptr)
        {
            app_state->tlas_completed.store(i + 1);
        }
    }

    std::printf("Scene AABB: (%g,%g,%g)-(%g,%g,%g)\n",
                scene.stats.scene_aabb_min.x,
                scene.stats.scene_aabb_min.y,
                scene.stats.scene_aabb_min.z,
                scene.stats.scene_aabb_max.x,
                scene.stats.scene_aabb_max.y,
                scene.stats.scene_aabb_max.z);

    scene.camera                 = ChooseCameraPreset(scene.stats.source_name);
    scene.stats.camera_preset    = scene.camera.preset_name;

    if (app_state != nullptr)
    {
        app_state->SetSceneStats(scene.stats);
        app_state->scene_loaded.store(true);
    }

    *out_scene = std::move(scene);
    return true;
}

SceneData BuildFallbackCubeScene()
{
    SceneData scene;

    scene.blas_vertices = {{
        {-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {1.0f, -1.0f, -1.0f},
        {-1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, 1.0f}, {-1.0f, -1.0f, 1.0f},
        {1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 1.0f},
        {1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, -1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f}, {-1.0f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f}, {-1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, -1.0f},
    }};

    SceneInstance instance{};
    instance.blas_idx      = 0;
    instance.transform[0]  = 1.0f;
    instance.transform[5]  = 1.0f;
    instance.transform[10] = 1.0f;
    scene.instances.push_back(instance);

    scene.stats.source_name        = "FallbackCube";
    scene.stats.camera_preset      = "FallbackCube";
    scene.stats.tlas_count         = 1;
    scene.stats.blas_count         = 1;
    scene.stats.instance_count     = 1;
    scene.stats.total_triangle_count = 12;
    scene.stats.scene_aabb_min     = glm::vec3(-1.0f);
    scene.stats.scene_aabb_max     = glm::vec3(1.0f);

    scene.camera.eye         = glm::vec3(0.0f, 0.0f, 5.0f);
    scene.camera.center      = glm::vec3(0.0f, 0.0f, 0.0f);
    scene.camera.up          = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.camera.invert_y    = false;
    scene.camera.preset_name = "FallbackCube";

    return scene;
}
