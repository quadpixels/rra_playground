#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include <GLFW/glfw3.h>
#include <glfw/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <dxcapi.h>
#include <DirectXMath.h>

#include <imgui.h>
#include <backends/imgui_impl_dx12.h>
#include <backends/imgui_impl_glfw.h>

#include "app_state.h"
#include "cpu_renderer.h"
#include "scene.h"

#undef min
#undef max

using RayInPixDumpFileMinimal = SceneRay;

enum class DispatchRayLayoutMode
{
    kClampToViewport = 0,
    kReflowBlocks = 1,
};

struct GpuRayInPix
{
    float    origin[3]{};
    float    tmin{0.001f};
    float    direction[3]{};
    float    tmax{10000.0f};
    uint32_t ray_flags{0};
    uint32_t instance_inclusion_mask{0xFF};
};

static_assert(sizeof(GpuRayInPix) == 40);

std::vector<RayInPixDumpFileMinimal> g_rays_in_pix_dumpfile_minimal;
glm::uvec3                         g_ray_in_pix_dispatch_dims;
std::vector<RayInPixDumpFileMinimal> g_display_ray_buffer;
std::vector<uint32_t>                g_display_ray_offsets;
uint32_t                             g_display_ray_active_pixels{0};
uint32_t                             g_display_ray_max_rays_per_pixel{0};
std::vector<RayInPixDumpFileMinimal> g_gpu_dispatch_ray_buffer;
std::vector<uint32_t>                g_gpu_dispatch_ray_offsets;
glm::uvec3                           g_gpu_dispatch_ray_dims{0};
uint32_t                             g_gpu_dispatch_ray_active_pixels{0};
uint32_t                             g_gpu_dispatch_ray_max_rays_per_pixel{0};
int                                  g_selected_dispatch_index{0};
int                                  g_dispatch_ray_layout_mode{0};
int                                  g_dispatch_reflow_block_w{16};
int                                  g_dispatch_reflow_block_h{16};
bool                                 g_dispatch_reflow_skip_empty{true};
bool                                 g_dispatch_ray_mapping_dirty{true};
bool                                 g_dispatch_ray_gpu_dirty{true};
bool                                 g_use_ray_in_pix{false};
bool                                 g_use_gpu_compact_dispatch_rays{false};

struct CompactDispatchReplayData
{
    struct BatchPixelRange
    {
        uint32_t pixel{0};
        uint32_t begin{0};
        uint32_t end{0};
        uint32_t pad{0};
    };

    std::vector<RayInPixDumpFileMinimal> rays;
    std::vector<uint32_t>                pixel_indices;
    std::vector<uint32_t>                batch_offsets;
    std::vector<BatchPixelRange>         batch_pixel_ranges;
    std::vector<uint32_t>                batch_pixel_range_offsets;
    std::vector<uint32_t>                pixel_compact_indices;
    uint32_t                             active_pixels{0};
    uint32_t                             max_rays_per_pixel{0};
};

CompactDispatchReplayData g_compact_dispatch_replay;
bool                      g_compact_dispatch_replay_dirty{true};

struct FrameTime
{
    std::vector<float> samples;  // elapsed, frame_time
    float                                last_secs{0};
    float                                curr_frametime{0};
    float                                update_interval = 0.75f; // seconds

    float GetFrameTime()
    {
        float curr_secs = glfwGetTime();
        if (curr_secs - last_secs > update_interval)
        {
            float sum = 0;
            for (float s : samples)
            {
                sum += s;
            }
            curr_frametime = sum * 1.0f / int(samples.size());
            samples.clear();
            last_secs = curr_secs;
        }
        return curr_frametime;
    }

    void AddSample(float x)
    {
        samples.push_back(x);
    }

    bool ShouldUpdate()
    {
        return (glfwGetTime() - last_secs > update_interval);
    }
};

FrameTime g_frame_time;
FrameTime g_dispatch_rays_time;
FrameTime g_compact_dispatch_rays_time;
FrameTime g_compact_scatter_time;
FrameTime g_compact_reduce_time;

struct Vertex
{
    DirectX::XMFLOAT3 position;
};
struct InstanceInfo
{
    uint64_t blas_idx{};
    float    transform[12];  // Row Major
};
struct RayGenCB
{
    DirectX::XMMATRIX inverse_view;
    DirectX::XMMATRIX inverse_proj;
    bool              invert_y;
    int               use_ray_binning;
    int               ao_samples;
    float             ao_radius;
    uint32_t          load_ray_from_buffer;
    uint32_t          buffer_w;
    uint32_t          buffer_h;
    uint32_t          buffer_d;
    uint32_t          rt_w;
    uint32_t          rt_h;
};

struct CompactReplayRootConstants
{
    uint32_t batch_base{0};
    uint32_t compact_mode{0};
    uint32_t pixel_offset_base{0};
    uint32_t pixel_index_base{0};
};

int WIN_W = 1280, WIN_H = 720;
constexpr const int FRAME_COUNT = 2;

int RT_W = 1280, RT_H = 720;  // Off-screen RT rendering width and height
int g_rt_width_input = RT_W;
int g_rt_height_input = RT_H;

GLFWwindow*      g_window;
bool             g_use_debug_layer{false};
ID3D12Device5*   g_device12;
IDXGIFactory4*   g_factory;
IDXGISwapChain3* g_swapchain;

ID3D12CommandQueue*         g_command_queue;
ID3D12CommandAllocator*     g_command_allocator;   // For rendering
ID3D12CommandAllocator*     g_command_allocator1;  // For building AS
ID3D12GraphicsCommandList4* g_command_list;   // For rendering
ID3D12GraphicsCommandList4* g_command_list1;  // For building AS

ID3D12RootSignature*         g_global_rootsig{};
ID3D12StateObject*           g_rt_state_object;
ID3D12StateObjectProperties* g_rt_state_object_props;

ID3D12RootSignature*         g_global_rootsig_ao{};
ID3D12StateObject*           g_rt_state_object_ao;
ID3D12StateObjectProperties* g_rt_state_object_props_ao;
ID3D12RootSignature*         g_compact_reduce_rootsig{};
ID3D12PipelineState*         g_compact_reduce_pso{};

ID3D12RootSignature* g_rootsig_fsquad{};
ID3D12PipelineState* g_pipeline_fsquad{};
ID3D12DescriptorHeap* g_srv_uav_cbv_heap_fsquad{};
ID3D12DescriptorHeap* g_imgui_srv_heap{};
ID3D12Resource*       g_fsquad_vb;
D3D12_VERTEX_BUFFER_VIEW g_fsquad_vbv;
ID3D12Resource*       g_cpu_rt_upload{};
D3D12_PLACED_SUBRESOURCE_FOOTPRINT g_cpu_rt_upload_footprint{};
UINT g_cpu_rt_upload_num_rows{};
UINT64 g_cpu_rt_upload_row_size{};
UINT64 g_cpu_rt_upload_total_size{};

glm::mat4 g_inv_view;
glm::mat4 g_inv_proj;
bool      g_invert_y = false;
bool      g_set_steady_power_state = false;

ID3D12DescriptorHeap* g_rtv_heap;
ID3D12DescriptorHeap* g_srv_uav_cbv_heap;
int                   g_srv_uav_cbv_descriptor_size;
int                   g_rtv_descriptor_size;
ID3D12Resource*       g_rendertargets[FRAME_COUNT];

ID3D12Resource* g_rt_output_resource;
ID3D12Resource* g_raygen_cb;

ID3D12Resource* g_raygen_sbt_storage;
ID3D12Resource* g_hit_sbt_storage;
ID3D12Resource* g_miss_sbt_storage;

ID3D12Resource* g_hitpos_ao;
ID3D12Resource* g_hitpos_ao_readback;
ID3D12Resource* g_ray_mapping, *g_ray_mapping_upload;  // For ray-binning experiments
ID3D12Resource* g_aoray_dirs, *g_aoray_dirs_upload;   // For ray-binning experiments
ID3D12Resource* g_raygen_sbt_storage_ao;
ID3D12Resource* g_hit_sbt_storage_ao;
ID3D12Resource* g_miss_sbt_storage_ao;
int             g_ao_sample_count{1};
bool            g_force_hitpos_dirty{false};
bool            g_hitpos_dirty{true};

ID3D12QueryHeap* g_query_heap;
ID3D12Resource*  g_query_readback_buffer;

ID3D12Resource* g_rays_in_pix_buffer;
ID3D12Resource* g_rays_in_pix_buffer_upload;
ID3D12Resource* g_ray_entry_offsets_buffer;
ID3D12Resource* g_ray_entry_offsets_buffer_upload;
ID3D12Resource* g_compact_ray_pixel_indices_buffer;
ID3D12Resource* g_compact_batch_pixel_offsets_buffer;
ID3D12Resource* g_compact_pixel_compact_indices_buffer;
ID3D12Resource* g_compact_ray_results_buffer;
ID3D12Resource* g_compact_accum_color_buffer;
ID3D12Resource* g_compact_accum_count_buffer;
std::string    g_adapter_name{"Unknown adapter"};

bool g_use_ao{false};
bool g_ray_mapping_dirty{true};

ID3D12Fence* g_fence;
int          g_fence_value;
HANDLE       g_fence_event;
int          g_frame_index;

const char* g_rra_file_name;

AppState  g_app_state;
SceneData g_scene_data;
CpuPrimaryRayRenderer g_cpu_renderer;
RenderBackend         g_render_backend{RenderBackend::kDxr};
int                   g_cpu_thread_count{static_cast<int>(std::min(4u, std::thread::hardware_concurrency()))};
bool                  g_cpu_refresh_requested{true};
std::mutex            g_cpu_renderer_mutex;
CpuBvhSettings        g_cpu_bvh_settings{};
std::atomic<bool>     g_cpu_bvh_rebuild_requested{true};
int                   g_cpu_bvh_fanout_index{0};
bool                  g_cpu_use_rra_topology{false};
int                   g_cpu_bvh_split_mode_index{1};
int                   g_cpu_primitive_node_triangle_capacity{3};

bool CopyToClipboard(const std::string& text)
{
    if (!OpenClipboard(NULL))
    {
        return false;
    }
    if (!EmptyClipboard())
    {
        CloseClipboard();
        return false;
    }
    size_t size = (text.size() + 1) * sizeof(char);
    HGLOBAL hGlobal = GlobalAlloc(GMEM_MOVEABLE, size);
    if (hGlobal == NULL)
    {
        CloseClipboard();
        return false;
    }

    char* pGlobal = (char*)GlobalLock(hGlobal);
    memcpy(pGlobal, text.c_str(), size);
    GlobalUnlock(hGlobal);

    if (SetClipboardData(CF_TEXT, hGlobal) == NULL)
    {
        GlobalFree(hGlobal);
        CloseClipboard();
        return false;
    }
    CloseClipboard();
    return true;
}

enum class CpuWorkerStage
{
    kIdle,
    kBuildingBvh,
    kRendering,
};

std::atomic<CpuWorkerStage> g_cpu_worker_stage{CpuWorkerStage::kIdle};
std::mutex            g_cpu_worker_mutex;
std::condition_variable g_cpu_worker_cv;
std::thread           g_cpu_worker_thread;
bool                  g_cpu_worker_exit{false};
bool                  g_cpu_request_pending{false};
std::atomic<bool>     g_cpu_worker_busy{false};
uint64_t              g_cpu_request_generation{0};
uint64_t              g_cpu_display_generation{0};
uint32_t              g_cpu_display_tiles_completed{0};
uint32_t              g_cpu_display_tiles_total{0};
std::optional<CpuRenderRequest> g_pending_cpu_request;
std::optional<CpuRenderResult>  g_latest_cpu_result;

void RebuildDisplayDispatchRays();
void RebuildGpuDispatchRays();
void BuildCompactDispatchReplay();
void ApplySceneCamera(const SceneData& scene);
void ApplyRenderTargetSize(int width, int height);
void WaitForPreviousFrame();

glm::vec3 g_scene_aabb_min{1e20, 1e20, 1e20}, g_scene_aabb_max{-1e20, -1e20, -1e20};
float     g_ao_radius{10000};
glm::vec3 g_cam_pos{};

enum BenchmarkState
{
    NOT_STARTED,
    BENCHMARKING,
};
BenchmarkState g_benchmarkState{NOT_STARTED};
int g_bmk_ft_count      = 0;
std::vector<float> g_bmk_frametimes;
const int          BMK_AO_SAMPLE_COUNT_LIMIT = 32;

static glm::vec3 Constrain(glm::vec3 x)
{
    x.x = std::max(0.0f, std::min(1.0f, x.x));
    x.y = std::max(0.0f, std::min(1.0f, x.y));
    x.z = std::max(0.0f, std::min(1.0f, x.z));
    return x;
}

static glm::vec3 TransformPosition(const glm::mat4& m, const glm::vec3& x)
{
    glm::vec4 x4(x, 0.0f);
    x4 = m * x4;
    x4.x += m[3][0];
    x4.y += m[3][1];
    x4.z += m[3][2];
    return glm::vec3(x4);
}

static glm::vec3 TransformDirection(const glm::mat4& m, const glm::vec3& x)
{
    glm::vec4 x4(x, 0.0f);
    x4 = m * x4;
    return glm::vec3(x4);
}

static glm::vec2 OctWrap(const glm::vec2& v)
{
    glm::vec2 ret(1.0f, 1.0f);
    ret -= glm::vec2(abs(v.y), abs(v.x));
    ret.x *= (v.x >= 0 ? 1 : -1);
    ret.y *= (v.y >= 0 ? 1 : -1);
    return ret;
}

static glm::vec2 OctEncode(glm::vec3 n)
{
    n = glm::normalize(n);
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0)
    {
        glm::vec2 xy = OctWrap(glm::vec2(n.x, n.y));
        n.x        = xy.x;
        n.y        = xy.y;
    }
    n.x = n.x * 0.5 + 0.5;
    n.y = n.y * 0.5 + 0.5;
    return glm::vec2(n.x, n.y);
}

glm::uvec2 TEA(unsigned int val0, unsigned int val1, unsigned int N)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return glm::uvec2(v0, v1);
}

unsigned LCG(int& seed)
{
    const unsigned int LCG_A = 1103515245u;
    const unsigned int LCG_C = 12345u;
    const unsigned int LCG_M = 0x00FFFFFFu;
    seed                     = (LCG_A * seed + LCG_C);
    return seed & LCG_M;
}

float RandF(int& seed)
{
    return float(LCG(seed)) / float(0x01000000);
}

glm::vec3 SampleHemisphereCosine(glm::vec3 n, int& seed)
{
    float phi         = 2.0f * 3.14159 * RandF(seed);
    float sinThetaSqr = RandF(seed);
    float sinTheta    = sqrt(sinThetaSqr);

    glm::vec3 axis = abs(n.x) > 0.001f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 t    = glm::cross(axis, n);
    t              = normalize(t);
    glm::vec3 s    = glm::cross(n, t);

    return glm::normalize(s * cos(phi) * sinTheta + t * sin(phi) * sinTheta + n * sqrt(1.0f - sinThetaSqr));
}

void CE(HRESULT x)
{
    if (FAILED(x))
    {
        printf("ERROR: %X\n", x);
        throw std::exception();
    }
}

void MarkDispatchRayMappingDirty()
{
    g_dispatch_ray_mapping_dirty = true;
    g_dispatch_ray_gpu_dirty = true;
    g_compact_dispatch_replay_dirty = true;
}

void ApplyStablePowerState(bool enabled)
{
    if (g_device12 == nullptr)
    {
        g_set_steady_power_state = enabled;
        return;
    }

    const HRESULT hr = g_device12->SetStablePowerState(enabled);
    if (FAILED(hr))
    {
        g_set_steady_power_state = !enabled;
        char message[128]{};
        snprintf(message, sizeof(message), "SetStablePowerState(%d) failed: 0x%08X", enabled ? 1 : 0, static_cast<unsigned int>(hr));
        g_app_state.SetStatus(message);
        return;
    }

    g_set_steady_power_state = enabled;
    g_app_state.SetStatus(enabled ? "Stable power state enabled" : "Stable power state disabled");
}

void ReleaseResource(ID3D12Resource** resource)
{
    if (resource != nullptr && *resource != nullptr)
    {
        (*resource)->Release();
        *resource = nullptr;
    }
}

void CreateStructuredBufferSrv(ID3D12Resource** resource,
                               const void* data,
                               size_t element_size,
                               uint32_t element_count,
                               uint32_t descriptor_index)
{
    ReleaseResource(resource);

    const uint32_t safe_element_count = std::max(1u, element_count);
    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment          = 0;
    desc.Width              = static_cast<UINT64>(safe_element_count) * element_size;
    desc.Height             = 1;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(resource)));

    void* mapped = nullptr;
    (*resource)->Map(0, nullptr, &mapped);
    if (data != nullptr && element_count > 0)
    {
        memcpy(mapped, data, static_cast<size_t>(element_count) * element_size);
    }
    else
    {
        memset(mapped, 0, static_cast<size_t>(safe_element_count) * element_size);
    }
    (*resource)->Unmap(0, nullptr);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Buffer.FirstElement        = 0;
    srv_desc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Buffer.NumElements         = safe_element_count;
    srv_desc.Buffer.StructureByteStride = static_cast<UINT>(element_size);
    srv_desc.Format                     = DXGI_FORMAT_UNKNOWN;
    srv_desc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;

    D3D12_CPU_DESCRIPTOR_HANDLE handle(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    handle.ptr += descriptor_index * g_srv_uav_cbv_descriptor_size;
    g_device12->CreateShaderResourceView(*resource, &srv_desc, handle);
}

std::vector<GpuRayInPix> BuildGpuRayUploadBuffer(const std::vector<RayInPixDumpFileMinimal>& rays)
{
    std::vector<GpuRayInPix> upload_rays;
    upload_rays.reserve(rays.size());
    for (const RayInPixDumpFileMinimal& ray : rays)
    {
        GpuRayInPix out_ray{};
        out_ray.origin[0] = ray.origin.x;
        out_ray.origin[1] = ray.origin.y;
        out_ray.origin[2] = ray.origin.z;
        out_ray.tmin = ray.tmin;
        out_ray.direction[0] = ray.direction.x;
        out_ray.direction[1] = ray.direction.y;
        out_ray.direction[2] = ray.direction.z;
        out_ray.tmax = ray.tmax;
        out_ray.ray_flags = ray.ray_flags;
        out_ray.instance_inclusion_mask = ray.instance_inclusion_mask;
        upload_rays.push_back(out_ray);
    }
    return upload_rays;
}

void CreateStructuredBufferUav(ID3D12Resource** resource, size_t element_size, uint32_t element_count, uint32_t descriptor_index)
{
    ReleaseResource(resource);

    const uint32_t safe_element_count = std::max(1u, element_count);
    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment          = 0;
    desc.Width              = static_cast<UINT64>(safe_element_count) * element_size;
    desc.Height             = 1;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    CE(g_device12->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(resource)));

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.FirstElement         = 0;
    uav_desc.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_NONE;
    uav_desc.Buffer.NumElements          = safe_element_count;
    uav_desc.Buffer.StructureByteStride  = static_cast<UINT>(element_size);

    D3D12_CPU_DESCRIPTOR_HANDLE handle(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    handle.ptr += descriptor_index * g_srv_uav_cbv_descriptor_size;
    g_device12->CreateUnorderedAccessView(*resource, nullptr, &uav_desc, handle);
}

void UpdateDispatchRayGpuBuffers()
{
    if (g_srv_uav_cbv_heap == nullptr || g_device12 == nullptr)
    {
        return;
    }
    if (g_dispatch_ray_mapping_dirty)
    {
        RebuildDisplayDispatchRays();
    }
    if (g_dispatch_ray_gpu_dirty)
    {
        RebuildGpuDispatchRays();
    }

    std::vector<GpuRayInPix> upload_rays = BuildGpuRayUploadBuffer(g_gpu_dispatch_ray_buffer);
    CreateStructuredBufferSrv(&g_rays_in_pix_buffer,
                              upload_rays.empty() ? nullptr : upload_rays.data(),
                              sizeof(GpuRayInPix),
                              static_cast<uint32_t>(upload_rays.size()),
                              8);
    CreateStructuredBufferSrv(&g_ray_entry_offsets_buffer,
                              g_gpu_dispatch_ray_offsets.empty() ? nullptr : g_gpu_dispatch_ray_offsets.data(),
                              sizeof(uint32_t),
                              static_cast<uint32_t>(g_gpu_dispatch_ray_offsets.size()),
                              9);
    g_dispatch_ray_gpu_dirty = false;
}

void UpdateCompactDispatchReplayGpuBuffers()
{
    if (g_srv_uav_cbv_heap == nullptr || g_device12 == nullptr)
    {
        return;
    }
    if (g_compact_dispatch_replay_dirty)
    {
        BuildCompactDispatchReplay();
    }

    std::vector<GpuRayInPix> upload_rays = BuildGpuRayUploadBuffer(g_compact_dispatch_replay.rays);
    CreateStructuredBufferSrv(&g_rays_in_pix_buffer,
                              upload_rays.empty() ? nullptr : upload_rays.data(),
                              sizeof(GpuRayInPix),
                              static_cast<uint32_t>(upload_rays.size()),
                              8);
    CreateStructuredBufferSrv(&g_compact_batch_pixel_offsets_buffer,
                              g_compact_dispatch_replay.batch_pixel_ranges.empty() ? nullptr : g_compact_dispatch_replay.batch_pixel_ranges.data(),
                              sizeof(CompactDispatchReplayData::BatchPixelRange),
                              static_cast<uint32_t>(g_compact_dispatch_replay.batch_pixel_ranges.size()),
                              13);
    CreateStructuredBufferSrv(&g_compact_pixel_compact_indices_buffer,
                              g_compact_dispatch_replay.pixel_compact_indices.empty() ? nullptr : g_compact_dispatch_replay.pixel_compact_indices.data(),
                              sizeof(uint32_t),
                              static_cast<uint32_t>(g_compact_dispatch_replay.pixel_compact_indices.size()),
                              14);
    CreateStructuredBufferUav(&g_compact_ray_results_buffer,
                              sizeof(float) * 4,
                              static_cast<uint32_t>(g_compact_dispatch_replay.rays.size()),
                              10);
    CreateStructuredBufferUav(&g_compact_accum_color_buffer,
                              sizeof(float) * 4,
                              static_cast<uint32_t>(RT_W * RT_H),
                              11);
    CreateStructuredBufferUav(&g_compact_accum_count_buffer,
                              sizeof(uint32_t),
                              static_cast<uint32_t>(RT_W * RT_H),
                              12);
    g_dispatch_ray_gpu_dirty = false;
}

D3D12_GPU_DESCRIPTOR_HANDLE GpuDescriptor(uint32_t descriptor_index)
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle(g_srv_uav_cbv_heap->GetGPUDescriptorHandleForHeapStart());
    handle.ptr += descriptor_index * g_srv_uav_cbv_descriptor_size;
    return handle;
}

void RecreateRenderTargetSizedResources()
{
    if (g_device12 == nullptr || g_srv_uav_cbv_heap == nullptr)
    {
        return;
    }

    WaitForPreviousFrame();

    ReleaseResource(&g_rt_output_resource);
    ReleaseResource(&g_cpu_rt_upload);
    ReleaseResource(&g_hitpos_ao);
    ReleaseResource(&g_hitpos_ao_readback);
    ReleaseResource(&g_ray_mapping_upload);
    ReleaseResource(&g_ray_mapping);
    ReleaseResource(&g_aoray_dirs_upload);
    ReleaseResource(&g_aoray_dirs);

    const uint32_t rt_w = static_cast<uint32_t>(std::max(1, RT_W));
    const uint32_t rt_h = static_cast<uint32_t>(std::max(1, RT_H));
    const uint64_t rt_pixels = static_cast<uint64_t>(rt_w) * rt_h;

    D3D12_HEAP_PROPERTIES default_props{};
    default_props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    default_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    default_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    default_props.CreationNodeMask     = 1;
    default_props.VisibleNodeMask      = 1;

    D3D12_HEAP_PROPERTIES upload_props = default_props;
    upload_props.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_HEAP_PROPERTIES readback_props = default_props;
    readback_props.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC rt_desc{};
    rt_desc.DepthOrArraySize = 1;
    rt_desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rt_desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    rt_desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    rt_desc.Width            = rt_w;
    rt_desc.Height           = rt_h;
    rt_desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rt_desc.MipLevels        = 1;
    rt_desc.SampleDesc.Count = 1;
    CE(g_device12->CreateCommittedResource(
        &default_props, D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&g_rt_output_resource)));
    g_rt_output_resource->SetName(L"RT output resource");

    g_device12->GetCopyableFootprints(&rt_desc, 0, 1, 0, &g_cpu_rt_upload_footprint, &g_cpu_rt_upload_num_rows, &g_cpu_rt_upload_row_size, &g_cpu_rt_upload_total_size);

    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Alignment          = 0;
    buffer_desc.Width              = g_cpu_rt_upload_total_size;
    buffer_desc.Height             = 1;
    buffer_desc.DepthOrArraySize   = 1;
    buffer_desc.MipLevels          = 1;
    buffer_desc.Format             = DXGI_FORMAT_UNKNOWN;
    buffer_desc.SampleDesc.Count   = 1;
    buffer_desc.SampleDesc.Quality = 0;
    buffer_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(
        &upload_props, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_cpu_rt_upload)));
    g_cpu_rt_upload->SetName(L"CPU RT upload");

    buffer_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    buffer_desc.Width = rt_pixels * sizeof(float) * 4;
    CE(g_device12->CreateCommittedResource(&default_props, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_hitpos_ao)));
    g_hitpos_ao->SetName(L"Hit position");

    D3D12_RESOURCE_DESC readback_desc = buffer_desc;
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(&readback_props, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_hitpos_ao_readback)));
    g_hitpos_ao_readback->SetName(L"Hit position readback");

    buffer_desc.Width = rt_pixels * sizeof(int);
    readback_desc = buffer_desc;
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(&upload_props, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_ray_mapping_upload)));
    CE(g_device12->CreateCommittedResource(&default_props, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&g_ray_mapping)));
    g_ray_mapping_upload->SetName(L"Ray mapping upload");
    g_ray_mapping->SetName(L"Ray mapping");

    buffer_desc.Width = rt_pixels * sizeof(float) * 3;
    readback_desc = buffer_desc;
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(&upload_props, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_aoray_dirs_upload)));
    CE(g_device12->CreateCommittedResource(&default_props, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&g_aoray_dirs)));
    g_aoray_dirs->SetName(L"AO ray dirs");
    g_aoray_dirs_upload->SetName(L"AO ray dirs upload");

    D3D12_CPU_DESCRIPTOR_HANDLE handle(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device12->CreateUnorderedAccessView(g_rt_output_resource, nullptr, &uav_desc, handle);

    handle.ptr = g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart().ptr + 5 * g_srv_uav_cbv_descriptor_size;
    uav_desc.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.FirstElement         = 0;
    uav_desc.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_NONE;
    uav_desc.Buffer.NumElements          = static_cast<UINT>(rt_pixels);
    uav_desc.Buffer.StructureByteStride  = sizeof(float) * 4;
    g_device12->CreateUnorderedAccessView(g_hitpos_ao, nullptr, &uav_desc, handle);

    handle.ptr += g_srv_uav_cbv_descriptor_size;
    uav_desc.Buffer.StructureByteStride = sizeof(int);
    g_device12->CreateUnorderedAccessView(g_ray_mapping, nullptr, &uav_desc, handle);

    handle.ptr += g_srv_uav_cbv_descriptor_size;
    uav_desc.Buffer.StructureByteStride = sizeof(float) * 3;
    g_device12->CreateUnorderedAccessView(g_aoray_dirs, nullptr, &uav_desc, handle);

    if (g_srv_uav_cbv_heap_fsquad != nullptr)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Shader4ComponentMapping   = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Format                    = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv_desc.Texture2D.MipLevels       = 1;
        srv_desc.Texture2D.MostDetailedMip = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE srv_handle(g_srv_uav_cbv_heap_fsquad->GetCPUDescriptorHandleForHeapStart());
        g_device12->CreateShaderResourceView(g_rt_output_resource, &srv_desc, srv_handle);
    }

    CreateStructuredBufferUav(&g_compact_ray_results_buffer, sizeof(float) * 4, 1, 10);
    CreateStructuredBufferUav(&g_compact_accum_color_buffer, sizeof(float) * 4, 1, 11);
    CreateStructuredBufferUav(&g_compact_accum_count_buffer, sizeof(uint32_t), 1, 12);

    MarkDispatchRayMappingDirty();
    g_ray_mapping_dirty = true;
    g_hitpos_dirty = true;
    g_cpu_refresh_requested = true;
    g_cpu_display_tiles_completed = 0;
    g_cpu_display_tiles_total = 0;
    g_cpu_display_generation = 0;
    g_app_state.SetCpuRenderStats({});
    {
        std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
        g_latest_cpu_result.reset();
    }

    if (g_app_state.as_built.load())
    {
        ApplySceneCamera(g_scene_data);
    }
}

void ApplyRenderTargetSize(int width, int height)
{
    width = std::clamp(width, 1, 16384);
    height = std::clamp(height, 1, 16384);
    if (RT_W == width && RT_H == height)
    {
        g_rt_width_input = RT_W;
        g_rt_height_input = RT_H;
        return;
    }

    RT_W = width;
    RT_H = height;
    g_rt_width_input = RT_W;
    g_rt_height_input = RT_H;
    if (g_device12 != nullptr && g_srv_uav_cbv_heap != nullptr)
    {
        RecreateRenderTargetSizedResources();
    }
    else
    {
        MarkDispatchRayMappingDirty();
    }
    printf("Render target resized to %dx%d\n", RT_W, RT_H);
}

void WaitForPreviousFrame()
{
    int val = g_fence_value++;
    CE(g_command_queue->Signal(g_fence, val));
    if (g_fence->GetCompletedValue() < val)
    {
        CE(g_fence->SetEventOnCompletion(val, g_fence_event));
        CE(WaitForSingleObject(g_fence_event, INFINITE));
    }
    g_frame_index = g_swapchain->GetCurrentBackBufferIndex();
}

void GlmMat4ToDirectXMatrix(DirectX::XMMATRIX* out, const glm::mat4& m)
{
    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 4; c++)
        {
            out->r[c].m128_f32[r] = m[c][r];
        }
    }
}

const char* ToString(SceneLoadStage stage)
{
    switch (stage)
    {
    case SceneLoadStage::kIdle:
        return "Idle";
    case SceneLoadStage::kLoadingTrace:
        return "Loading trace";
    case SceneLoadStage::kExtractingDispatchRays:
        return "Extracting dispatch rays";
    case SceneLoadStage::kExtractingBlas:
        return "Extracting BLAS";
    case SceneLoadStage::kExtractingTlas:
        return "Extracting TLAS";
    case SceneLoadStage::kBuildingGpuBlas:
        return "Building GPU BLAS";
    case SceneLoadStage::kBuildingGpuTlas:
        return "Building GPU TLAS";
    case SceneLoadStage::kBuildingCpuBvh:
        return "Building CPU BVH";
    case SceneLoadStage::kReady:
        return "Ready";
    case SceneLoadStage::kFailed:
        return "Failed";
    default:
        return "Unknown";
    }
}

const char* ToString(RenderBackend backend)
{
    switch (backend)
    {
    case RenderBackend::kDxr:
        return "DXR";
    case RenderBackend::kCpuBruteForce:
        return "CPU brute force";
    case RenderBackend::kCpuBvh:
        return "CPU BVH";
    default:
        return "Unknown";
    }
}

const char* ToString(CpuWorkerStage stage)
{
    switch (stage)
    {
    case CpuWorkerStage::kIdle:
        return "idle";
    case CpuWorkerStage::kBuildingBvh:
        return "building BVH";
    case CpuWorkerStage::kRendering:
        return "rendering";
    default:
        return "unknown";
    }
}

void ApplySceneCamera(const SceneData& scene)
{
    g_invert_y = scene.camera.invert_y;
    g_cam_pos  = scene.camera.eye;

    glm::mat4 view = glm::lookAt(scene.camera.eye, scene.camera.center, scene.camera.up);
    glm::mat4 proj = glm::perspectiveLH_ZO(glm::radians(60.0f), -1.0f * RT_W / RT_H, -0.1f, -499.0f) * (-1.0f);
    if (scene.camera.preset_name == "FallbackCube")
    {
        proj = glm::perspectiveLH_ZO(glm::radians(90.0f), -1.0f * RT_W / RT_H, -0.1f, -499.0f) * (-1.0f);
    }

    g_inv_view = glm::inverse(view);
    g_inv_proj = glm::inverse(proj);
}

std::vector<std::vector<Vertex>> ConvertSceneVertices(const SceneData& scene)
{
    std::vector<std::vector<Vertex>> vertices;
    vertices.reserve(scene.blas_vertices.size());
    for (const auto& mesh_vertices : scene.blas_vertices)
    {
        std::vector<Vertex> converted;
        converted.reserve(mesh_vertices.size());
        for (const glm::vec3& p : mesh_vertices)
        {
            converted.push_back({{p.x, p.y, p.z}});
        }
        vertices.push_back(std::move(converted));
    }
    return vertices;
}

std::vector<InstanceInfo> ConvertSceneInstances(const SceneData& scene)
{
    std::vector<InstanceInfo> instances(scene.instances.size());
    for (size_t i = 0; i < scene.instances.size(); i++)
    {
        instances[i].blas_idx = scene.instances[i].blas_idx;
        memcpy(instances[i].transform, scene.instances[i].transform, sizeof(instances[i].transform));
    }
    return instances;
}

uint32_t InvocationRayBegin(const std::vector<uint32_t>& offsets, uint32_t invocation)
{
    return invocation == 0 ? 0 : offsets[invocation - 1];
}

uint32_t InvocationRayEnd(const std::vector<uint32_t>& offsets, const std::vector<RayInPixDumpFileMinimal>& rays, uint32_t invocation)
{
    return std::min<uint32_t>(offsets[invocation], static_cast<uint32_t>(rays.size()));
}

void ComputeRayOffsetStats(const std::vector<uint32_t>& offsets, uint32_t* active_pixels, uint32_t* max_rays_per_pixel)
{
    *active_pixels = 0;
    *max_rays_per_pixel = 0;
    uint32_t previous = 0;
    for (uint32_t offset : offsets)
    {
        const uint32_t count = offset >= previous ? offset - previous : 0;
        if (count > 0)
        {
            (*active_pixels)++;
            *max_rays_per_pixel = std::max(*max_rays_per_pixel, count);
        }
        previous = offset;
    }
}

const SceneDispatchRays* CurrentRraDispatch()
{
    if (g_scene_data.dispatches.empty())
    {
        return nullptr;
    }
    g_selected_dispatch_index = std::clamp(g_selected_dispatch_index, 0, static_cast<int>(g_scene_data.dispatches.size()) - 1);
    return &g_scene_data.dispatches[g_selected_dispatch_index];
}

void RebuildDisplayDispatchRays()
{
    const SceneDispatchRays* dispatch = CurrentRraDispatch();

    const std::vector<RayInPixDumpFileMinimal>* src_rays = nullptr;
    const std::vector<uint32_t>* src_offsets = nullptr;
    std::vector<uint32_t> pix_dump_offsets;
    glm::uvec3 src_dims(0);
    if (dispatch != nullptr)
    {
        src_rays = &dispatch->rays;
        src_offsets = &dispatch->ray_offsets;
        src_dims = dispatch->dispatch_dims;
    }
    else
    {
        src_rays = &g_rays_in_pix_dumpfile_minimal;
        src_dims = g_ray_in_pix_dispatch_dims;
        pix_dump_offsets.reserve(src_rays->size());
        for (uint32_t i = 0; i < src_rays->size(); i++)
        {
            pix_dump_offsets.push_back(i + 1);
        }
        src_offsets = &pix_dump_offsets;
    }

    g_ray_in_pix_dispatch_dims = src_dims;
    if (src_rays->empty() || src_offsets->empty() || RT_W <= 0 || RT_H <= 0)
    {
        g_display_ray_buffer.clear();
        g_display_ray_offsets.assign(static_cast<size_t>(std::max(0, RT_W)) * static_cast<size_t>(std::max(0, RT_H)), 0);
        g_display_ray_active_pixels = 0;
        g_display_ray_max_rays_per_pixel = 0;
        g_dispatch_ray_mapping_dirty = false;
        g_dispatch_ray_gpu_dirty = true;
        return;
    }

    const uint32_t dst_w = static_cast<uint32_t>(RT_W);
    const uint32_t dst_h = static_cast<uint32_t>(RT_H);
    const uint32_t dst_nr = dst_w * dst_h;
    g_display_ray_buffer.clear();
    g_display_ray_offsets.assign(dst_nr, 0);

    if (static_cast<DispatchRayLayoutMode>(g_dispatch_ray_layout_mode) == DispatchRayLayoutMode::kReflowBlocks)
    {
        const uint32_t block_w = static_cast<uint32_t>(std::max(1, g_dispatch_reflow_block_w));
        const uint32_t block_h = static_cast<uint32_t>(std::max(1, g_dispatch_reflow_block_h));
        const uint32_t src_w = std::max(1u, src_dims.x);
        const uint32_t src_h = std::max(1u, src_dims.y * std::max(1u, src_dims.z));
        const uint32_t src_grid_x = (src_w + block_w - 1) / block_w;
        const uint32_t src_grid_y = (src_h + block_h - 1) / block_h;
        const uint32_t dst_grid_x = (dst_w + block_w - 1) / block_w;
        const uint32_t dst_grid_y = (dst_h + block_h - 1) / block_h;
        std::vector<std::pair<uint32_t, uint32_t>> dst_ranges(dst_nr);

        uint32_t target_gx = 0;
        uint32_t target_gy = 0;
        for (uint32_t gy = 0; gy < src_grid_y; gy++)
        {
            for (uint32_t gx = 0; gx < src_grid_x; gx++)
            {
                bool has_work = false;
                if (!g_dispatch_reflow_skip_empty)
                {
                    has_work = true;
                }
                else
                {
                    for (uint32_t ty = 0; ty < block_h && !has_work; ty++)
                    {
                        for (uint32_t tx = 0; tx < block_w; tx++)
                        {
                            const uint32_t sx = gx * block_w + tx;
                            const uint32_t sy_flat = gy * block_h + ty;
                            if (sx >= src_dims.x || sy_flat >= src_dims.y * src_dims.z)
                            {
                                continue;
                            }
                            const uint32_t sz = src_dims.y == 0 ? 0 : sy_flat / src_dims.y;
                            const uint32_t sy = src_dims.y == 0 ? 0 : sy_flat % src_dims.y;
                            const uint32_t src_invocation = sx + sy * src_dims.x + sz * src_dims.x * src_dims.y;
                            const uint32_t lb = InvocationRayBegin(*src_offsets, src_invocation);
                            const uint32_t ub = InvocationRayEnd(*src_offsets, *src_rays, src_invocation);
                            if (ub > lb)
                            {
                                has_work = true;
                                break;
                            }
                        }
                    }
                }

                if (!has_work || target_gy >= dst_grid_y)
                {
                    continue;
                }

                for (uint32_t ty = 0; ty < block_h; ty++)
                {
                    for (uint32_t tx = 0; tx < block_w; tx++)
                    {
                        const uint32_t sx = gx * block_w + tx;
                        const uint32_t sy_flat = gy * block_h + ty;
                        if (sx >= src_dims.x || sy_flat >= src_dims.y * src_dims.z)
                        {
                            continue;
                        }
                        const uint32_t dx = target_gx * block_w + tx;
                        const uint32_t dy = target_gy * block_h + ty;
                        if (dx >= dst_w || dy >= dst_h)
                        {
                            continue;
                        }
                        const uint32_t sz = src_dims.y == 0 ? 0 : sy_flat / src_dims.y;
                        const uint32_t sy = src_dims.y == 0 ? 0 : sy_flat % src_dims.y;
                        const uint32_t src_invocation = sx + sy * src_dims.x + sz * src_dims.x * src_dims.y;
                        dst_ranges[dx + dy * dst_w] = std::make_pair(InvocationRayBegin(*src_offsets, src_invocation),
                                                                      InvocationRayEnd(*src_offsets, *src_rays, src_invocation));
                    }
                }

                target_gx++;
                if (target_gx >= dst_grid_x)
                {
                    target_gx = 0;
                    target_gy++;
                }
            }
        }

        uint32_t total_ray_count = 0;
        for (uint32_t dst = 0; dst < dst_nr; dst++)
        {
            const auto [lb, ub] = dst_ranges[dst];
            total_ray_count += ub > lb ? ub - lb : 0;
            g_display_ray_offsets[dst] = total_ray_count;
        }

        g_display_ray_buffer.reserve(total_ray_count);
        for (uint32_t dst = 0; dst < dst_nr; dst++)
        {
            const auto [lb, ub] = dst_ranges[dst];
            if (ub > lb)
            {
                g_display_ray_buffer.insert(g_display_ray_buffer.end(), src_rays->begin() + lb, src_rays->begin() + ub);
            }
        }
    }
    else
    {
        std::vector<uint32_t> ray_counts(dst_nr, 0);
        for (uint32_t z = 0; z < std::max(1u, src_dims.z); z++)
        {
            for (uint32_t y = 0; y < src_dims.y; y++)
            {
                for (uint32_t x = 0; x < src_dims.x; x++)
                {
                    const uint32_t src_invocation = x + y * src_dims.x + z * src_dims.x * src_dims.y;
                    const uint32_t dst_x = std::min(x, dst_w - 1);
                    const uint32_t dst_y = std::min(y + z * src_dims.y, dst_h - 1);
                    const uint32_t lb = InvocationRayBegin(*src_offsets, src_invocation);
                    const uint32_t ub = InvocationRayEnd(*src_offsets, *src_rays, src_invocation);
                    if (ub > lb)
                    {
                        ray_counts[dst_x + dst_y * dst_w] += ub - lb;
                    }
                }
            }
        }

        uint32_t total_ray_count = 0;
        for (uint32_t dst = 0; dst < dst_nr; dst++)
        {
            total_ray_count += ray_counts[dst];
            g_display_ray_offsets[dst] = total_ray_count;
        }

        g_display_ray_buffer.resize(total_ray_count);
        std::vector<uint32_t> write_offsets = g_display_ray_offsets;
        for (uint32_t dst = dst_nr; dst > 0; dst--)
        {
            write_offsets[dst - 1] = dst == 1 ? 0 : g_display_ray_offsets[dst - 2];
        }
        for (uint32_t z = 0; z < std::max(1u, src_dims.z); z++)
        {
            for (uint32_t y = 0; y < src_dims.y; y++)
            {
                for (uint32_t x = 0; x < src_dims.x; x++)
                {
                    const uint32_t src_invocation = x + y * src_dims.x + z * src_dims.x * src_dims.y;
                    const uint32_t dst_x = std::min(x, dst_w - 1);
                    const uint32_t dst_y = std::min(y + z * src_dims.y, dst_h - 1);
                    const uint32_t dst = dst_x + dst_y * dst_w;
                    const uint32_t lb = InvocationRayBegin(*src_offsets, src_invocation);
                    const uint32_t ub = InvocationRayEnd(*src_offsets, *src_rays, src_invocation);
                    if (ub > lb)
                    {
                        const uint32_t count = ub - lb;
                        //std::copy(src_rays->begin() + lb, src_rays->begin() + ub, g_display_ray_buffer.begin() + write_offsets[dst]);
                        memcpy(g_display_ray_buffer.data() + write_offsets[dst], src_rays->data() + lb, sizeof(RayInPixDumpFileMinimal) * count);
                        write_offsets[dst] += count;
                    }
                }
            }
        }
    }

    ComputeRayOffsetStats(g_display_ray_offsets, &g_display_ray_active_pixels, &g_display_ray_max_rays_per_pixel);
    g_dispatch_ray_mapping_dirty = false;
    g_dispatch_ray_gpu_dirty = true;
}

void RebuildGpuDispatchRays()
{
    const SceneDispatchRays* dispatch = CurrentRraDispatch();
    if (dispatch == nullptr)
    {
        g_gpu_dispatch_ray_buffer = g_rays_in_pix_dumpfile_minimal;
        g_gpu_dispatch_ray_offsets.clear();
        g_gpu_dispatch_ray_offsets.reserve(g_gpu_dispatch_ray_buffer.size());
        for (uint32_t i = 0; i < g_gpu_dispatch_ray_buffer.size(); i++)
        {
            g_gpu_dispatch_ray_offsets.push_back(i + 1);
        }
        g_gpu_dispatch_ray_dims = g_ray_in_pix_dispatch_dims;
    }
    else if (static_cast<DispatchRayLayoutMode>(g_dispatch_ray_layout_mode) == DispatchRayLayoutMode::kReflowBlocks)
    {
        if (g_dispatch_ray_mapping_dirty)
        {
            RebuildDisplayDispatchRays();
        }
        g_gpu_dispatch_ray_buffer = g_display_ray_buffer;
        g_gpu_dispatch_ray_offsets = g_display_ray_offsets;
        g_gpu_dispatch_ray_dims = glm::uvec3(static_cast<uint32_t>(RT_W), static_cast<uint32_t>(RT_H), 1);
    }
    else
    {
        g_gpu_dispatch_ray_buffer = dispatch->rays;
        g_gpu_dispatch_ray_offsets = dispatch->ray_offsets;
        g_gpu_dispatch_ray_dims = dispatch->dispatch_dims;
    }

    ComputeRayOffsetStats(g_gpu_dispatch_ray_offsets, &g_gpu_dispatch_ray_active_pixels, &g_gpu_dispatch_ray_max_rays_per_pixel);
    g_dispatch_ray_gpu_dirty = true;
}

void BuildCompactDispatchReplay()
{
    if (g_dispatch_ray_mapping_dirty)
    {
        RebuildDisplayDispatchRays();
    }

    g_compact_dispatch_replay = {};
    if (g_display_ray_buffer.empty() || g_display_ray_offsets.empty())
    {
        g_compact_dispatch_replay.batch_offsets.push_back(0);
        g_compact_dispatch_replay_dirty = false;
        return;
    }

    struct Entry
    {
        uint32_t src_ray_index{0};
        uint32_t pixel_index{0};
        uint32_t ray_index{0};
        uint32_t sbt_record_offset{0};
        uint32_t sbt_record_stride{0};
        uint32_t miss_index{0};
    };

    std::vector<Entry> entries;
    entries.reserve(g_display_ray_buffer.size());
    uint32_t previous_offset = 0;
    for (uint32_t pixel = 0; pixel < g_display_ray_offsets.size(); pixel++)
    {
        const uint32_t end = std::min<uint32_t>(g_display_ray_offsets[pixel], static_cast<uint32_t>(g_display_ray_buffer.size()));
        for (uint32_t ray = previous_offset; ray < end; ray++)
        {
            const auto& scene_ray = g_display_ray_buffer[ray];
            entries.push_back({ray,
                               pixel,
                               ray - previous_offset,
                               scene_ray.sbt_record_offset,
                               scene_ray.sbt_record_stride,
                               scene_ray.miss_index});
        }
        previous_offset = end;
    }

    std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return std::tie(a.ray_index, a.sbt_record_offset, a.sbt_record_stride, a.miss_index, a.pixel_index) <
               std::tie(b.ray_index, b.sbt_record_offset, b.sbt_record_stride, b.miss_index, b.pixel_index);
    });

    g_compact_dispatch_replay.rays.reserve(entries.size());
    g_compact_dispatch_replay.pixel_indices.reserve(entries.size());
    g_compact_dispatch_replay.batch_offsets.push_back(0);

    uint32_t current_ray_index = entries.empty() ? 0 : entries.front().ray_index;
    for (const Entry& entry : entries)
    {
        if (entry.ray_index != current_ray_index)
        {
            current_ray_index = entry.ray_index;
            g_compact_dispatch_replay.batch_offsets.push_back(static_cast<uint32_t>(g_compact_dispatch_replay.rays.size()));
        }
        g_compact_dispatch_replay.rays.push_back(g_display_ray_buffer[entry.src_ray_index]);
        g_compact_dispatch_replay.pixel_indices.push_back(entry.pixel_index);
    }
    g_compact_dispatch_replay.batch_offsets.push_back(static_cast<uint32_t>(g_compact_dispatch_replay.rays.size()));

    const uint32_t dst_nr = static_cast<uint32_t>(g_display_ray_offsets.size());
    std::vector<uint32_t> counts(dst_nr, 0);
    std::vector<uint32_t> cursors(dst_nr, 0);
    std::vector<uint32_t> touched_pixels;
    g_compact_dispatch_replay.batch_pixel_range_offsets.reserve(g_compact_dispatch_replay.batch_offsets.size());
    for (uint32_t batch = 0; batch + 1 < g_compact_dispatch_replay.batch_offsets.size(); batch++)
    {
        const uint32_t batch_begin = g_compact_dispatch_replay.batch_offsets[batch];
        const uint32_t batch_end = g_compact_dispatch_replay.batch_offsets[batch + 1];
        touched_pixels.clear();
        for (uint32_t compact_id = batch_begin; compact_id < batch_end; compact_id++)
        {
            const uint32_t pixel = g_compact_dispatch_replay.pixel_indices[compact_id];
            if (counts[pixel] == 0)
            {
                touched_pixels.push_back(pixel);
            }
            counts[pixel]++;
        }
        std::sort(touched_pixels.begin(), touched_pixels.end());

        g_compact_dispatch_replay.batch_pixel_range_offsets.push_back(static_cast<uint32_t>(g_compact_dispatch_replay.batch_pixel_ranges.size()));
        for (uint32_t pixel : touched_pixels)
        {
            const uint32_t begin = static_cast<uint32_t>(g_compact_dispatch_replay.pixel_compact_indices.size());
            const uint32_t end = begin + counts[pixel];
            g_compact_dispatch_replay.batch_pixel_ranges.push_back({pixel, begin, end, 0});
            cursors[pixel] = begin;
            g_compact_dispatch_replay.pixel_compact_indices.resize(end);
        }

        for (uint32_t compact_id = batch_begin; compact_id < batch_end; compact_id++)
        {
            const uint32_t pixel = g_compact_dispatch_replay.pixel_indices[compact_id];
            g_compact_dispatch_replay.pixel_compact_indices[cursors[pixel]++] = compact_id;
        }

        for (uint32_t pixel : touched_pixels)
        {
            counts[pixel] = 0;
            cursors[pixel] = 0;
        }
    }
    g_compact_dispatch_replay.batch_pixel_range_offsets.push_back(static_cast<uint32_t>(g_compact_dispatch_replay.batch_pixel_ranges.size()));

    ComputeRayOffsetStats(g_display_ray_offsets,
                          &g_compact_dispatch_replay.active_pixels,
                          &g_compact_dispatch_replay.max_rays_per_pixel);
    g_compact_dispatch_replay_dirty = false;
}

std::vector<CpuRay> BuildCpuExternalRays()
{
    std::vector<CpuRay> rays;
    rays.reserve(g_display_ray_buffer.size());
    for (const auto& ray : g_display_ray_buffer)
    {
        CpuRay cpu_ray{};
        cpu_ray.origin                  = ray.origin;
        cpu_ray.direction               = ray.direction;
        cpu_ray.tmin                    = ray.tmin;
        cpu_ray.tmax                    = ray.tmax;
        cpu_ray.ray_flags               = ray.ray_flags;
        cpu_ray.instance_inclusion_mask = ray.instance_inclusion_mask;
        rays.push_back(cpu_ray);
    }
    return rays;
}

uint32_t CurrentCpuBvhFanout()
{
    static constexpr uint32_t kFanouts[] = {2, 4, 6, 8, 10, 12, 16};
    constexpr int fanout_count = static_cast<int>(sizeof(kFanouts) / sizeof(kFanouts[0]));
    g_cpu_bvh_fanout_index = std::clamp(g_cpu_bvh_fanout_index, 0, fanout_count - 1);
    return kFanouts[g_cpu_bvh_fanout_index];
}

CpuBvhSettings CurrentCpuBvhSettings()
{
    CpuBvhSettings settings{};
    settings.fanout = CurrentCpuBvhFanout();
    settings.build_mode =
        (g_cpu_use_rra_topology && (settings.fanout == 4 || settings.fanout == 8))
            ? CpuBvhBuildMode::kTranscribedRra
            : CpuBvhBuildMode::kWideMedian;
    settings.split_mode =
        g_cpu_bvh_split_mode_index == 1 ? CpuBvhSplitMode::kBinnedSah : CpuBvhSplitMode::kEqualCounts;
    settings.primitive_node_triangle_capacity =
        static_cast<uint32_t>(std::clamp(g_cpu_primitive_node_triangle_capacity, 1, 3));
    return settings;
}

void WriteCpuCheckerboard(CpuRenderResult* result)
{
    result->rgba.resize(static_cast<size_t>(result->width) * static_cast<size_t>(result->height) * 4);
    for (uint32_t y = 0; y < result->height; y++)
    {
        for (uint32_t x = 0; x < result->width; x++)
        {
            const bool dark = ((x / 8) + (y / 8)) % 2 == 0;
            uint8_t* pixel = result->rgba.data() + (static_cast<size_t>(x) + static_cast<size_t>(y) * result->width) * 4;
            pixel[0] = dark ? 128 : 255;
            pixel[1] = dark ? 128 : 255;
            pixel[2] = dark ? 128 : 0;
            pixel[3] = 255;
        }
    }
}

void PublishCpuCheckerboard(RenderBackend backend, uint64_t generation)
{
    std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
    g_cpu_display_tiles_completed = 0;
    g_cpu_display_tiles_total = RT_W == 0 ? 0 : ((RT_W + 15) / 16) * ((RT_H + 15) / 16);
    g_cpu_display_generation = 0;

    CpuRenderResult checker{};
    checker.width = RT_W;
    checker.height = RT_H;
    checker.tiles_completed = 0;
    checker.tiles_total = g_cpu_display_tiles_total;
    checker.generation = generation;
    checker.complete = false;
    checker.backend = backend;
    WriteCpuCheckerboard(&checker);
    g_latest_cpu_result = std::move(checker);
}

void QueueInitialCpuBvhBuild()
{
    CpuRenderRequest request{};
    request.width = RT_W;
    request.height = RT_H;
    request.thread_count = static_cast<uint32_t>(std::max(1, g_cpu_thread_count));
    request.inverse_view = g_inv_view;
    request.inverse_proj = g_inv_proj;
    request.invert_y = g_invert_y;
    request.backend = RenderBackend::kCpuBvh;
    request.generation = ++g_cpu_request_generation;
    request.rebuild_bvh = true;
    request.render_after_build = false;
    request.bvh_settings = CurrentCpuBvhSettings();

    {
        std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
        if (g_cpu_worker_busy || g_cpu_request_pending)
        {
            return;
        }
        g_pending_cpu_request = std::move(request);
        g_cpu_request_pending = true;
    }
    g_cpu_worker_cv.notify_one();
}

void QueueCpuRender()
{
    if (g_render_backend == RenderBackend::kDxr)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
        if (!g_cpu_refresh_requested || g_cpu_request_pending || g_cpu_worker_busy)
        {
            return;
        }
    }

    {
        std::lock_guard<std::mutex> renderer_lock(g_cpu_renderer_mutex);
        if (!g_cpu_renderer.IsReady())
        {
            PublishCpuCheckerboard(g_render_backend, ++g_cpu_request_generation);
            return;
        }
    }

    if (g_dispatch_ray_mapping_dirty)
    {
        RebuildDisplayDispatchRays();
    }

    CpuRenderRequest request{};
    request.width             = RT_W;
    request.height            = RT_H;
    request.thread_count      = static_cast<uint32_t>(std::max(1, g_cpu_thread_count));
    request.inverse_view      = g_inv_view;
    request.inverse_proj      = g_inv_proj;
    request.invert_y          = g_invert_y;
    request.use_external_rays = g_use_ray_in_pix;
    request.backend           = g_render_backend;
    request.generation        = ++g_cpu_request_generation;
    request.rebuild_bvh       = g_render_backend == RenderBackend::kCpuBvh && g_cpu_bvh_rebuild_requested.load();
    request.bvh_settings      = CurrentCpuBvhSettings();
    request.external_rays     = BuildCpuExternalRays();
    request.external_ray_offsets = g_display_ray_offsets;
    const RenderBackend checker_backend = request.backend;
    const uint64_t checker_generation = request.generation;

    {
        std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
        g_pending_cpu_request  = std::move(request);
        g_cpu_request_pending  = true;
        g_cpu_refresh_requested = false;
        g_cpu_display_tiles_completed = 0;
        g_cpu_display_tiles_total = g_pending_cpu_request->width == 0 ? 0 : ((g_pending_cpu_request->width + 15) / 16) * ((g_pending_cpu_request->height + 15) / 16);
        g_cpu_display_generation = 0;
    }
    PublishCpuCheckerboard(checker_backend, checker_generation);
    g_cpu_worker_cv.notify_one();
}

bool TryConsumeCpuRenderResult()
{
    std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
    if (!g_latest_cpu_result.has_value())
    {
        return false;
    }
    if (g_latest_cpu_result->generation < g_cpu_display_generation)
    {
        return false;
    }
    if (g_latest_cpu_result->backend != g_render_backend)
    {
        return false;
    }
    if (g_latest_cpu_result->generation == g_cpu_display_generation &&
        g_latest_cpu_result->tiles_completed <= g_cpu_display_tiles_completed)
    {
        return false;
    }
    g_cpu_display_generation = g_latest_cpu_result->generation;
    g_cpu_display_tiles_completed = g_latest_cpu_result->tiles_completed;
    g_cpu_display_tiles_total = g_latest_cpu_result->tiles_total;
    g_app_state.SetCpuRenderStats(g_latest_cpu_result->stats);
    return true;
}

void UploadCpuRenderTarget()
{
    std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
    if (!g_latest_cpu_result.has_value())
    {
        return;
    }

    const CpuRenderResult& result = *g_latest_cpu_result;
    if (result.backend != g_render_backend ||
        result.rgba.empty() ||
        result.width != static_cast<uint32_t>(RT_W) ||
        result.height != static_cast<uint32_t>(RT_H))
    {
        return;
    }

    uint8_t* mapped = nullptr;
    g_cpu_rt_upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    for (UINT row = 0; row < static_cast<UINT>(RT_H); row++)
    {
        memcpy(mapped + g_cpu_rt_upload_footprint.Offset + row * g_cpu_rt_upload_footprint.Footprint.RowPitch,
               result.rgba.data() + row * RT_W * 4,
               static_cast<size_t>(RT_W) * 4);
    }
    g_cpu_rt_upload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource        = g_rt_output_resource;
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource       = g_cpu_rt_upload;
    src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = g_cpu_rt_upload_footprint;

    g_command_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

void CpuWorkerMain()
{
    while (true)
    {
        CpuRenderRequest request{};
        {
            std::unique_lock<std::mutex> lock(g_cpu_worker_mutex);
            g_cpu_worker_cv.wait(lock, []() { return g_cpu_worker_exit || g_cpu_request_pending; });
            if (g_cpu_worker_exit)
            {
                return;
            }
            request               = *g_pending_cpu_request;
            g_cpu_request_pending = false;
            g_cpu_worker_busy     = true;
        }

        if (request.rebuild_bvh)
        {
            g_cpu_worker_stage.store(CpuWorkerStage::kBuildingBvh);
            g_app_state.scene_stage.store(SceneLoadStage::kBuildingCpuBvh);
            g_app_state.SetStatus("Building CPU BVH");
            {
                std::lock_guard<std::mutex> renderer_lock(g_cpu_renderer_mutex);
                g_cpu_renderer.BuildFromScene(g_scene_data, request.bvh_settings, &g_app_state);
            }
            g_cpu_bvh_settings = request.bvh_settings;
            g_cpu_bvh_rebuild_requested.store(false);
            g_app_state.scene_stage.store(SceneLoadStage::kReady);
            g_app_state.SetStatus("CPU BVH ready");
        }
        if (!request.render_after_build)
        {
            std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
            g_cpu_worker_busy = false;
            g_cpu_worker_stage.store(CpuWorkerStage::kIdle);
            continue;
        }

        CpuRenderResult result{};
        g_cpu_worker_stage.store(CpuWorkerStage::kRendering);
        {
            std::lock_guard<std::mutex> renderer_lock(g_cpu_renderer_mutex);
            g_cpu_renderer.Render(
                request,
                &result,
                [&](const CpuRenderResult& partial_result) {
                    std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
                    if (!g_latest_cpu_result.has_value() ||
                        partial_result.generation > g_latest_cpu_result->generation ||
                        (partial_result.generation == g_latest_cpu_result->generation &&
                         partial_result.tiles_completed >= g_latest_cpu_result->tiles_completed))
                    {
                        g_latest_cpu_result = partial_result;
                    }
                });
        }

        {
            std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
            if (!g_latest_cpu_result.has_value() ||
                result.generation > g_latest_cpu_result->generation ||
                (result.generation == g_latest_cpu_result->generation &&
                 result.tiles_completed >= g_latest_cpu_result->tiles_completed))
            {
                g_latest_cpu_result = std::move(result);
            }
            g_cpu_worker_busy = false;
            g_cpu_worker_stage.store(CpuWorkerStage::kIdle);
        }
    }
}

void InitImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 1;
    heap_desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CE(g_device12->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&g_imgui_srv_heap)));

    ImGui_ImplGlfw_InitForOther(g_window, true);
    ImGui_ImplDX12_InitInfo init_info{};
    init_info.Device                    = g_device12;
    init_info.CommandQueue              = g_command_queue;
    init_info.NumFramesInFlight         = FRAME_COUNT;
    init_info.RTVFormat                 = DXGI_FORMAT_R8G8B8A8_UNORM;
    init_info.DSVFormat                 = DXGI_FORMAT_UNKNOWN;
    init_info.SrvDescriptorHeap         = g_imgui_srv_heap;
    init_info.LegacySingleSrvCpuDescriptor = g_imgui_srv_heap->GetCPUDescriptorHandleForHeapStart();
    init_info.LegacySingleSrvGpuDescriptor = g_imgui_srv_heap->GetGPUDescriptorHandleForHeapStart();
    ImGui_ImplDX12_Init(&init_info);
}

void ShutdownImGui()
{
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void DrawImGuiPanel()
{
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    SceneStats stats;
    CpuRenderStats cpu_stats;
    uint64_t cpu_bvh_node_count = 0;
    uint64_t cpu_bvh_primitive_count = 0;
    uint32_t cpu_bvh_fanout = 2;
    uint32_t cpu_bvh_primitive_node_triangle_capacity = 3;
    bool cpu_bvh_uses_rra_topology = false;
    bool cpu_bvh_uses_binned_sah = false;
    std::string status_line;
    std::string last_error;
    {
        std::lock_guard<std::mutex> lock(g_app_state.details_mutex);
        stats       = g_app_state.scene_stats;
        cpu_stats   = g_app_state.cpu_render_stats;
        cpu_bvh_node_count = g_app_state.cpu_bvh_node_count;
        cpu_bvh_primitive_count = g_app_state.cpu_bvh_primitive_count;
        cpu_bvh_fanout = g_app_state.cpu_bvh_fanout;
        cpu_bvh_primitive_node_triangle_capacity = g_app_state.cpu_bvh_primitive_node_triangle_capacity;
        cpu_bvh_uses_rra_topology = g_app_state.cpu_bvh_uses_rra_topology;
        cpu_bvh_uses_binned_sah = g_app_state.cpu_bvh_uses_binned_sah;
        status_line = g_app_state.status_line;
        last_error  = g_app_state.last_error;
    }

    ImGui::SetNextWindowBgAlpha(0.82f);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Runtime"))
    {

        auto check_hover_times = [&]() {
            bool is_compact = (g_use_ray_in_pix && g_use_gpu_compact_dispatch_rays);
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                if (is_compact)
                {
                    ImGui::Text("Click to copy frametime, dispatch_rays_time, scatter_time to clipboard");
                }
                else
                {
                    ImGui::Text("Click to copy frametime to clipboard");
                }
                ImGui::EndTooltip();
                if (ImGui::IsItemClicked())
                {
                    char buf[200];
                    if (is_compact)
                    {
                        snprintf(buf, sizeof(buf), "%g", g_app_state.last_gpu_frame_ms);
                    }
                    else
                    {
                        snprintf(buf, sizeof(buf), "%g, %g, %g, %g", 
                          g_app_state.last_gpu_frame_ms, 
                          g_app_state.last_gpu_compact_dispatch_rays_ms, 
                          g_app_state.last_gpu_compact_scatter_ms,
                          g_app_state.last_gpu_compact_reduce_ms
                        );
                    }
                    CopyToClipboard(std::string(buf));
                }
            }
        };

        ImGui::Text("Stage: %s", ToString(g_app_state.scene_stage.load()));
        ImGui::Text("Adapter: %s", g_adapter_name.c_str());
        bool stable_power_state = g_set_steady_power_state;
        if (ImGui::Checkbox("Stable power state", &stable_power_state))
        {
            ApplyStablePowerState(stable_power_state);
        }
        ImGui::Text("GPU frame:                %.3f ms", g_app_state.last_gpu_frame_ms);
        check_hover_times();
        if (g_use_ray_in_pix && g_use_gpu_compact_dispatch_rays)
        {
            ImGui::Text("GPU compact DispatchRays: %.3f ms", g_app_state.last_gpu_compact_dispatch_rays_ms);
            check_hover_times();
            ImGui::Text("GPU compact Scatter     : %.3f ms", g_app_state.last_gpu_compact_scatter_ms);
            check_hover_times();
            ImGui::Text("GPU compact Reduce      : %.3f ms", g_app_state.last_gpu_compact_reduce_ms);
            check_hover_times();
        }
        else
        {
            ImGui::Text("GPU DispatchRays:         %.3f ms", g_app_state.last_gpu_dispatch_rays_ms);
            check_hover_times();
        }
        ImGui::Text("Render: %dx%d -> %dx%d", RT_W, RT_H, WIN_W, WIN_H);
        ImGui::PushItemWidth(90);
        ImGui::InputInt("RT width", &g_rt_width_input, 1, 64);
        ImGui::SameLine();
        ImGui::InputInt("RT height", &g_rt_height_input, 1, 64);
        ImGui::PopItemWidth();
        if (ImGui::Button("Apply RT"))
        {
            ApplyRenderTargetSize(g_rt_width_input, g_rt_height_input);
        }
        ImGui::Text("Backend: %s", ToString(g_render_backend));
        ImGui::Text("CPU worker: %s", ToString(g_cpu_worker_stage.load()));
        ImGui::Text("CPU tiles done: %u / %u", g_cpu_display_tiles_completed, g_cpu_display_tiles_total);
        ImGui::Text("Mode: %s", g_use_ao ? "AO rays" : "Primary rays");
        int backend = static_cast<int>(g_render_backend);
        if (ImGui::Combo("Render path", &backend, "DXR\0CPU brute force\0CPU BVH\0"))
        {
            const RenderBackend new_backend = static_cast<RenderBackend>(backend);
            if (new_backend != g_render_backend)
            {
                const bool cpu_backend_changed =
                    new_backend != RenderBackend::kDxr &&
                    g_render_backend != RenderBackend::kDxr &&
                    new_backend != g_render_backend;
                g_render_backend = new_backend;
                if (cpu_backend_changed)
                {
                    g_cpu_refresh_requested = true;
                    g_cpu_display_tiles_completed = 0;
                    g_cpu_display_tiles_total = 0;
                    g_app_state.SetCpuRenderStats({});
                }
            }
            if (g_render_backend != RenderBackend::kDxr)
            {
                g_use_ao = false;
            }
        }
        if (g_render_backend != RenderBackend::kDxr)
        {
            ImGui::InputInt("CPU threads", &g_cpu_thread_count, 1, 4);
            if (g_cpu_thread_count < 1)
            {
                g_cpu_thread_count = 1;
            }
            if (g_render_backend == RenderBackend::kCpuBvh)
            {
                if (g_cpu_worker_busy)
                {
                    ImGui::BeginDisabled();
                }
                static const char* kFanoutLabels[] = {"2", "4", "6", "8", "10", "12", "16"};
                if (ImGui::Combo("CPU BVH fanout", &g_cpu_bvh_fanout_index, kFanoutLabels, IM_ARRAYSIZE(kFanoutLabels)))
                {
                    g_cpu_bvh_rebuild_requested.store(true);
                }

                static const char* kSplitLabels[] = {"Equal counts", "Binned SAH"};
                if (ImGui::Combo("CPU BVH split", &g_cpu_bvh_split_mode_index, kSplitLabels, IM_ARRAYSIZE(kSplitLabels)))
                {
                    g_cpu_bvh_rebuild_requested.store(true);
                }
                if (ImGui::InputInt("Primitive node tris", &g_cpu_primitive_node_triangle_capacity, 1, 1))
                {
                    g_cpu_primitive_node_triangle_capacity = std::clamp(g_cpu_primitive_node_triangle_capacity, 1, 3);
                    g_cpu_bvh_rebuild_requested.store(true);
                }

                const uint32_t requested_fanout = CurrentCpuBvhFanout();
                const bool can_transcribe_rra = requested_fanout == 4 || requested_fanout == 8;
                if (!can_transcribe_rra)
                {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Checkbox("Use RRA BVH topology", &g_cpu_use_rra_topology))
                {
                    g_cpu_bvh_rebuild_requested.store(true);
                }
                if (!can_transcribe_rra)
                {
                    ImGui::EndDisabled();
                }
                if (g_cpu_worker_busy)
                {
                    ImGui::EndDisabled();
                }
            }

            if (g_cpu_worker_busy)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Refresh CPU"))
            {
                g_cpu_refresh_requested = true;
                if (g_render_backend == RenderBackend::kCpuBvh)
                {
                    g_cpu_bvh_rebuild_requested.store(true);
                }
                g_cpu_display_tiles_completed = 0;
                g_cpu_display_tiles_total = 0;
                g_cpu_display_generation = 0;
                g_app_state.SetCpuRenderStats({});
                std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
                g_latest_cpu_result.reset();
            }
            if (g_cpu_worker_busy)
            {
                ImGui::EndDisabled();
            }
        }
        ImGui::Separator();
        if (ImGui::Checkbox("Use dispatch rays", &g_use_ray_in_pix))
        {
            MarkDispatchRayMappingDirty();
        }
        const bool has_rra_dispatches = !g_scene_data.dispatches.empty();
        if (g_use_ray_in_pix)
        {
            if (ImGui::Checkbox("GPU compact replay", &g_use_gpu_compact_dispatch_rays))
            {
                MarkDispatchRayMappingDirty();
            }
            if (has_rra_dispatches)
            {
                std::vector<const char*> dispatch_names;
                dispatch_names.reserve(g_scene_data.dispatches.size());
                for (const auto& dispatch : g_scene_data.dispatches)
                {
                    dispatch_names.push_back(dispatch.name.c_str());
                }
                if (ImGui::Combo("RRA dispatch", &g_selected_dispatch_index, dispatch_names.data(), static_cast<int>(dispatch_names.size())))
                {
                    const SceneDispatchRays* dispatch = CurrentRraDispatch();
                    if (dispatch != nullptr)
                    {
                        ApplyRenderTargetSize(static_cast<int>(dispatch->dispatch_dims.x),
                                              static_cast<int>(dispatch->dispatch_dims.y * std::max(1u, dispatch->dispatch_dims.z)));
                    }
                    MarkDispatchRayMappingDirty();
                }
                const SceneDispatchRays* dispatch = CurrentRraDispatch();
                if (dispatch != nullptr && ImGui::Button("RT = dispatch"))
                {
                    ApplyRenderTargetSize(static_cast<int>(dispatch->dispatch_dims.x),
                                          static_cast<int>(dispatch->dispatch_dims.y * std::max(1u, dispatch->dispatch_dims.z)));
                }
            }
            else
            {
                ImGui::Text("RRA dispatches: none");
            }

            static const char* kDispatchLayoutLabels[] = {"Clamp to viewport", "Reflow blocks"};
            if (ImGui::Combo("Ray layout", &g_dispatch_ray_layout_mode, kDispatchLayoutLabels, IM_ARRAYSIZE(kDispatchLayoutLabels)))
            {
                MarkDispatchRayMappingDirty();
            }
            static int reflow_block_w{8}, reflow_block_h{8};
            static bool reflow_skip_empty{false};
            if (static_cast<DispatchRayLayoutMode>(g_dispatch_ray_layout_mode) == DispatchRayLayoutMode::kReflowBlocks)
            {
                ImGui::Text("Reflow block ");
                ImGui::SameLine();
                ImGui::PushItemWidth(80);
                if (ImGui::InputInt("##ReflowBlockW", &reflow_block_w, 1, 8))
                {
                    reflow_block_w               = std::max(1, reflow_block_w);
                }
                ImGui::PopItemWidth();
                ImGui::SameLine();
                ImGui::Text("x");
                ImGui::SameLine();
                ImGui::PushItemWidth(80);
                if (ImGui::InputInt("##ReflowBlockH", &reflow_block_h, 1, 8))
                {
                    reflow_block_h               = std::max(1, reflow_block_h);
                }
                ImGui::PopItemWidth();
                if (ImGui::Checkbox("Skip empty blocks", &reflow_skip_empty))
                {
                }
                ImGui::SameLine();
                if (ImGui::Button("Apply"))
                {
                    g_dispatch_reflow_block_w    = reflow_block_w;
                    g_dispatch_reflow_block_h    = reflow_block_h;
                    g_dispatch_reflow_skip_empty = reflow_skip_empty;
                    MarkDispatchRayMappingDirty();
                }
            }
            if (g_dispatch_ray_mapping_dirty)
            {
                RebuildDisplayDispatchRays();
            }
            ImGui::Text("Dispatch dims: %u x %u x %u", g_ray_in_pix_dispatch_dims.x, g_ray_in_pix_dispatch_dims.y, g_ray_in_pix_dispatch_dims.z);
            ImGui::Text("CPU mapped rays: %zu", g_display_ray_buffer.size());
            ImGui::Text("CPU active pixels: %u / %u", g_display_ray_active_pixels, RT_W * RT_H);
            ImGui::Text("CPU max rays/pixel: %u", g_display_ray_max_rays_per_pixel);
            ImGui::Text("GPU uploaded rays: %zu", g_gpu_dispatch_ray_buffer.size());
            ImGui::Text("GPU active pixels: %u", g_gpu_dispatch_ray_active_pixels);
            ImGui::Text("GPU max rays/pixel: %u", g_gpu_dispatch_ray_max_rays_per_pixel);
            if (g_use_gpu_compact_dispatch_rays)
            {
                ImGui::Text("Compact batches: %zu", g_compact_dispatch_replay.batch_offsets.empty() ? 0 : g_compact_dispatch_replay.batch_offsets.size() - 1);
                ImGui::Text("Compact rays: %zu", g_compact_dispatch_replay.rays.size());
                ImGui::Text("Compact pixel ranges: %zu", g_compact_dispatch_replay.batch_pixel_ranges.size());
                ImGui::Text("Compact pixel ray indices: %zu", g_compact_dispatch_replay.pixel_compact_indices.size());
            }
        }
        if (g_use_ao)
        {
            ImGui::Text("AO samples: %d", g_ao_sample_count);
            ImGui::Text("AO radius: %.1f", g_ao_radius);
        }
        ImGui::Text("PIX rays: %s", g_use_ray_in_pix ? "on" : "off");
        ImGui::Separator();

        const int blas_total = static_cast<int>(g_app_state.blas_total.load());
        const int blas_done  = static_cast<int>(g_app_state.blas_completed.load());
        const int tlas_total = static_cast<int>(g_app_state.tlas_total.load());
        const int tlas_done  = static_cast<int>(g_app_state.tlas_completed.load());
        const int dispatch_total = static_cast<int>(g_app_state.dispatch_total.load());
        const int dispatch_done  = static_cast<int>(g_app_state.dispatch_completed.load());
        ImGui::Text("Dispatch rays progress: %d / %d", dispatch_done, dispatch_total);
        if (dispatch_total > 0)
        {
            ImGui::ProgressBar(static_cast<float>(dispatch_done) / dispatch_total, ImVec2(-1.0f, 0.0f));
        }
        ImGui::Text("BLAS progress: %d / %d", blas_done, blas_total);
        if (blas_total > 0)
        {
            ImGui::ProgressBar(static_cast<float>(blas_done) / blas_total, ImVec2(-1.0f, 0.0f));
        }
        ImGui::Text("TLAS progress: %d / %d", tlas_done, tlas_total);
        if (tlas_total > 0)
        {
            ImGui::ProgressBar(static_cast<float>(tlas_done) / tlas_total, ImVec2(-1.0f, 0.0f));
        }
        if (!status_line.empty())
        {
            ImGui::TextWrapped("%s", status_line.c_str());
        }
        if (!last_error.empty())
        {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", last_error.c_str());
        }

        auto check_hover_stats = [&]() {
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::Text("Click to copy stats to clipboard");
                ImGui::EndTooltip();
                if (ImGui::IsItemClicked())
                {
                    // rays, nodes, boxnodes, ray-box, ray-tri, TLAS->BLAS, ShortStackReEntries
                    char buf[200];
                    snprintf(buf,
                             sizeof(buf),
                             "%llu, %llu, %llu, %llu, %llu, %llu, %llu",
                             cpu_stats.rays,
                             cpu_stats.bvh_steps,
                             cpu_stats.box_nodes,
                             cpu_stats.ray_box_tests,
                             cpu_stats.ray_triangle_tests,
                             cpu_stats.tlas_to_blas,
                             0ULL);
                    CopyToClipboard(std::string(buf));
                }
            }
          
        };

        ImGui::Separator();
        ImGui::Text("Scene source: %s", stats.source_name.empty() ? "<none>" : stats.source_name.c_str());
        ImGui::Text("Camera preset: %s", stats.camera_preset.empty() ? "<default>" : stats.camera_preset.c_str());
        ImGui::Text("TLAS: %llu", stats.tlas_count);
        ImGui::Text("BLAS: %llu", stats.blas_count);
        ImGui::Text("Instances: %llu", stats.instance_count);
        ImGui::Text("Triangles: %llu", stats.total_triangle_count);
        ImGui::Text("CPU BVH prims: %llu", cpu_bvh_primitive_count);
        ImGui::Text("CPU BVH nodes: %llu", cpu_bvh_node_count);
        ImGui::Text("CPU BVH fanout: %u", cpu_bvh_fanout);
        ImGui::Text("CPU primitive node tris: %u", cpu_bvh_primitive_node_triangle_capacity);
        ImGui::Text("CPU BVH source: %s",
                    cpu_bvh_uses_rra_topology ? "RRA topology" : (cpu_bvh_uses_binned_sah ? "binned SAH collapsed" : "equal-count wide"));
        ImGui::Separator();
        ImGui::Text("CPU rays: %llu", cpu_stats.rays);
        check_hover_stats();
        ImGui::Text("CPU traversal steps: %llu", cpu_stats.bvh_steps);
        check_hover_stats();
        ImGui::Text("CPU box nodes: %llu", cpu_stats.box_nodes);
        check_hover_stats();
        ImGui::Text("CPU ray-box tests: %llu (%.2f/node)", cpu_stats.ray_box_tests, cpu_stats.ray_box_tests * 1.0f / cpu_stats.box_nodes);
        check_hover_stats();
        ImGui::Text("CPU tri nodes: %llu", cpu_stats.tri_nodes);
        check_hover_stats();
        ImGui::Text("CPU ray-tri tests: %llu (%.2f/node)", cpu_stats.ray_triangle_tests, cpu_stats.ray_triangle_tests * 1.0f / cpu_stats.tri_nodes);
        check_hover_stats();
        ImGui::Text("CPU TLAS->BLAS: %llu", cpu_stats.tlas_to_blas);
        check_hover_stats();
        if (cpu_stats.rays > 0)
        {
            ImGui::Text("avg BVH steps/ray: %.3f", static_cast<double>(cpu_stats.bvh_steps) / static_cast<double>(cpu_stats.rays));
            ImGui::Text("avg box tests/ray: %.3f", static_cast<double>(cpu_stats.ray_box_tests) / static_cast<double>(cpu_stats.rays));
            ImGui::Text("avg tri tests/ray: %.3f", static_cast<double>(cpu_stats.ray_triangle_tests) / static_cast<double>(cpu_stats.rays));
            ImGui::Text("avg TLAS->BLAS/ray: %.3f", static_cast<double>(cpu_stats.tlas_to_blas) / static_cast<double>(cpu_stats.rays));
        }
        ImGui::Text("AABB min: %.3f %.3f %.3f", stats.scene_aabb_min.x, stats.scene_aabb_min.y, stats.scene_aabb_min.z);
        ImGui::Text("AABB max: %.3f %.3f %.3f", stats.scene_aabb_max.x, stats.scene_aabb_max.y, stats.scene_aabb_max.z);
    }
    ImGui::End();

    ImGui::Render();
}

IDxcBlob* CompileShaderLibrary(LPCWSTR fileName)
{
    static IDxcCompiler*       pCompiler = nullptr;
    static IDxcLibrary*        pLibrary  = nullptr;
    static IDxcIncludeHandler* dxcIncludeHandler;

    HRESULT hr;

    // Initialize the DXC compiler and compiler helper
    if (!pCompiler)
    {
        CE(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler), reinterpret_cast<void**>(&pCompiler)));
        CE(DxcCreateInstance(CLSID_DxcLibrary, __uuidof(IDxcLibrary), reinterpret_cast<void**>(&pLibrary)));
        CE(pLibrary->CreateIncludeHandler(&dxcIncludeHandler));
    }
    // Open and read the file
    std::ifstream shaderFile(fileName);
    if (shaderFile.good() == false)
    {
        throw std::logic_error("Cannot find shader file");
    }
    std::stringstream strStream;
    strStream << shaderFile.rdbuf();
    std::string sShader = strStream.str();

    // Create blob from the string
    IDxcBlobEncoding* pTextBlob;
    CE(pLibrary->CreateBlobWithEncodingFromPinned(LPBYTE(sShader.c_str()), static_cast<uint32_t>(sShader.size()), 0, &pTextBlob));

    // Compile
    IDxcOperationResult* pResult;
    const wchar_t*       args[] = { L"-O3" };
    CE(pCompiler->Compile(pTextBlob, fileName, L"", L"lib_6_5", args, 1, nullptr, 0, dxcIncludeHandler, &pResult));

    // Verify the result
    HRESULT resultCode;
    CE(pResult->GetStatus(&resultCode));
    if (FAILED(resultCode))
    {
        IDxcBlobEncoding* pError;
        hr = pResult->GetErrorBuffer(&pError);
        if (FAILED(hr))
        {
            throw std::logic_error("Failed to get shader compiler error");
        }

        // Convert error blob to a string
        std::vector<char> infoLog(pError->GetBufferSize() + 1);
        memcpy(infoLog.data(), pError->GetBufferPointer(), pError->GetBufferSize());
        infoLog[pError->GetBufferSize()] = 0;

        std::string errorMsg = "Shader Compiler Error:\n";
        errorMsg.append(infoLog.data());

        MessageBoxA(nullptr, errorMsg.c_str(), "Error!", MB_OK);
        throw std::logic_error("Failed compile shader");
    }

    IDxcBlob* pBlob;
    CE(pResult->GetResult(&pBlob));
    return pBlob;
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard)
    {
        return;
    }

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
        {
            if (g_set_steady_power_state)
            {
                g_device12->SetStablePowerState(false);
            }
            exit(0);
            break;
        }
        case GLFW_KEY_1:
        {
            g_render_backend = RenderBackend::kDxr;
            g_use_ao = true;
            break;
        }
        case GLFW_KEY_0: {
            g_render_backend = RenderBackend::kDxr;
            g_use_ao = false;
            break;
        }
        case GLFW_KEY_4:
        {
            if (g_render_backend != RenderBackend::kCpuBruteForce)
            {
                g_cpu_refresh_requested = true;
                g_cpu_display_tiles_completed = 0;
                g_cpu_display_tiles_total = 0;
                g_app_state.SetCpuRenderStats({});
            }
            g_render_backend = RenderBackend::kCpuBruteForce;
            g_use_ao         = false;
            break;
        }
        case GLFW_KEY_5:
        {
            if (g_render_backend != RenderBackend::kCpuBvh)
            {
                g_cpu_refresh_requested = true;
                g_cpu_display_tiles_completed = 0;
                g_cpu_display_tiles_total = 0;
                g_app_state.SetCpuRenderStats({});
            }
            g_render_backend = RenderBackend::kCpuBvh;
            g_use_ao         = false;
            break;
        }
        case GLFW_KEY_UP:
        {
            g_ao_sample_count++;
            if (g_ao_sample_count > 32)
                g_ao_sample_count = 32;
            break;
        }
        case GLFW_KEY_DOWN:
        {
            g_ao_sample_count--;
            if (g_ao_sample_count < 0)
                g_ao_sample_count = 0;
            break;
        }
        case GLFW_KEY_B:
        {
            if (g_benchmarkState == BenchmarkState::NOT_STARTED)
            {
                g_render_backend = RenderBackend::kDxr;
                g_benchmarkState = BenchmarkState::BENCHMARKING;
                g_ao_sample_count = 0;
                g_use_ao          = true;
                g_bmk_ft_count    = 0;
                g_bmk_frametimes.clear();
            }
            break;
        }
        case GLFW_KEY_D:
        {
            g_force_hitpos_dirty = !g_force_hitpos_dirty;
            break;
        }
        case GLFW_KEY_LEFT_BRACKET:
        {
            g_ao_radius /= 10;
            if (g_ao_radius <= 1)
            {
                g_ao_radius = 1;
            }
            break;
        }
        case GLFW_KEY_RIGHT_BRACKET:
        {
            g_ao_radius *= 10;
            if (g_ao_radius >= 10000)
            {
                g_ao_radius = 10000;
            }
            break;
        }
        case GLFW_KEY_P:
        {
            g_use_ray_in_pix = !g_use_ray_in_pix;
            g_dispatch_ray_mapping_dirty = true;
            printf("g_use_ray_in_pix = %d\n", g_use_ray_in_pix);
            break;
        }
        default:
            break;
        }
    }
}

void OnSwapchainSizeChanged()
{
    WaitForPreviousFrame();

    for (int i = 0; i < FRAME_COUNT; i++)
    {
        g_rendertargets[i]->Release();
    }

    g_swapchain->ResizeBuffers(FRAME_COUNT, WIN_W, WIN_H, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = g_rtv_heap->GetCPUDescriptorHandleForHeapStart();
    for (int i = 0; i < FRAME_COUNT; i++)
    {
        CE(g_swapchain->GetBuffer(i, IID_PPV_ARGS(&g_rendertargets[i])));
        g_device12->CreateRenderTargetView(g_rendertargets[i], nullptr, rtv_handle);
        rtv_handle.ptr += g_rtv_descriptor_size;
    }
}

void WindowResizeCallback(GLFWwindow* window, int width, int height)
{
    if (width < 1 || height < 1) return;
    WIN_W = width;
    WIN_H = height;
    OnSwapchainSizeChanged();
    printf("Resized to %dx%d\n", width, height);
}

void WindowMaximizeCallback(GLFWwindow* window, int maximized)
{
    if (window != g_window)
        return;
    glfwGetWindowSize(window, &WIN_W, &WIN_H);
    OnSwapchainSizeChanged();
    printf("Maximized to %dx%d\n", WIN_W, WIN_H);
}

void CreateMyRRALoaderWindow()
{
    AllocConsole();
    freopen_s((FILE**)stdin, "CONIN$", "r", stderr);
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
    freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);

    if (!glfwInit())
    {
        printf("GLFW initialization failed\n");
    }
    printf("GLFW inited.\n");
    GLFWmonitor*       primary_monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* video_mode      = glfwGetVideoMode(primary_monitor);
    printf("Video mode of primary monitor is %dx%d\n", video_mode->width, video_mode->height);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    g_window = glfwCreateWindow(WIN_W, WIN_H, "MyRraLoader", nullptr, nullptr);

    glfwSetKeyCallback(g_window, KeyCallback);
    glfwSetWindowSizeCallback(g_window, WindowResizeCallback);
    glfwSetWindowSizeLimits(g_window, 64, 64, GLFW_DONT_CARE, GLFW_DONT_CARE);
}

void InitDeviceAndCommandQ()
{
    unsigned dxgi_factory_flags{0};
    if (g_use_debug_layer)
    {
        ID3D12Debug* debug_controller{};
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller))))
        {
            debug_controller->EnableDebugLayer();
            dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
            printf("Enabling DX12 debugging layer.\n");

            ID3D12Debug1* debug_controller1{};
            debug_controller->QueryInterface(IID_PPV_ARGS(&debug_controller1));
            if (debug_controller1)
            {
                printf("Enabling GPU-based validation.\n");
                debug_controller1->SetEnableGPUBasedValidation(true);
            }
        }
    }

    CE(CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&g_factory)));
    IDXGIAdapter1* adapter;
    for (int i = 0; g_factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        else
        {
            CE(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&g_device12)));
            char adapter_name[256]{};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, adapter_name, static_cast<int>(sizeof(adapter_name)), nullptr, nullptr);
            g_adapter_name = adapter_name;
            printf("Created device = %ls\n", desc.Description);
            break;
        }
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5;
    CE(g_device12->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)));
    if (options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0)
    {
        printf("This device supports DXR 1.0.\n");
    }
    if (options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1)
    {
        printf("This device supports DXR 1.1.\n");
    }

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    CE(g_device12->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&g_command_queue)));
    CE(g_device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
    g_fence_value = 1;
    g_fence_event = CreateEventW(nullptr, false, false, L"Fence");
    g_command_queue->SetName(L"Command Queue");

    if (g_set_steady_power_state)
    {
        CE(g_device12->SetStablePowerState(true));
    }
}

void InitSwapChain()
{
    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.BufferCount      = FRAME_COUNT;
    scd.Width            = WIN_W;
    scd.Height           = WIN_H;
    scd.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage      = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect       = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    CE(g_factory->CreateSwapChainForHwnd(g_command_queue, glfwGetWin32Window(g_window), &scd, nullptr, nullptr, (IDXGISwapChain1**)&g_swapchain));
    printf("Created swapchain.\n");

    // RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC dhd{};
    dhd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    dhd.NumDescriptors = FRAME_COUNT;
    dhd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    CE(g_device12->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(&g_rtv_heap)));

    g_rtv_descriptor_size                  = g_device12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = g_rtv_heap->GetCPUDescriptorHandleForHeapStart();
    for (int i = 0; i < FRAME_COUNT; i++)
    {
        CE(g_swapchain->GetBuffer(i, IID_PPV_ARGS(&g_rendertargets[i])));
        g_device12->CreateRenderTargetView(g_rendertargets[i], nullptr, rtv_handle);
        rtv_handle.ptr += g_rtv_descriptor_size;
    }
    printf("Created backbuffers' RTVs\n");

    CE(g_factory->MakeWindowAssociation(glfwGetWin32Window(g_window), DXGI_MWA_NO_ALT_ENTER));
}

void InitDX12Stuff()
{
    CE(g_device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_command_allocator)));
    CE(g_device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_command_allocator, nullptr, IID_PPV_ARGS(&g_command_list)));
    g_command_list->Close();
    g_command_list->SetName(L"Command List");

    CE(g_device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_command_allocator1)));
    CE(g_device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_command_allocator, nullptr, IID_PPV_ARGS(&g_command_list1)));
    g_command_list1->Close();
    g_command_list1->SetName(L"Command List 1");

    // RT output resource
    D3D12_RESOURCE_DESC desc{};
    desc.DepthOrArraySize = 1;
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.Width            = RT_W;
    desc.Height           = RT_H;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;
    CE(g_device12->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&g_rt_output_resource)));
    g_rt_output_resource->SetName(L"RT output resource");

    g_device12->GetCopyableFootprints(&desc, 0, 1, 0, &g_cpu_rt_upload_footprint, &g_cpu_rt_upload_num_rows, &g_cpu_rt_upload_row_size, &g_cpu_rt_upload_total_size);
    D3D12_RESOURCE_DESC upload_desc{};
    upload_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    upload_desc.Alignment          = 0;
    upload_desc.Width              = g_cpu_rt_upload_total_size;
    upload_desc.Height             = 1;
    upload_desc.DepthOrArraySize   = 1;
    upload_desc.MipLevels          = 1;
    upload_desc.Format             = DXGI_FORMAT_UNKNOWN;
    upload_desc.SampleDesc.Count   = 1;
    upload_desc.SampleDesc.Quality = 0;
    upload_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    upload_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES upload_props{};
    upload_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    upload_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    upload_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    upload_props.CreationNodeMask     = 1;
    upload_props.VisibleNodeMask      = 1;
    CE(g_device12->CreateCommittedResource(
        &upload_props, D3D12_HEAP_FLAG_NONE, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_cpu_rt_upload)));
    g_cpu_rt_upload->SetName(L"CPU RT upload");

    // Hit position in world space
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Format    = DXGI_FORMAT_UNKNOWN;
    desc.Width     = RT_W * RT_H * sizeof(float) * 4;
    desc.Height    = 1;
    desc.Layout    = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CE(g_device12->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_hitpos_ao)));
    g_hitpos_ao->SetName(L"Hit position");

    D3D12_HEAP_PROPERTIES props1 = props;
    props1.Type                  = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc1    = desc;
    desc1.Flags                  = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(
        &props1, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_hitpos_ao_readback)));
    CE(g_device12->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_hitpos_ao)));
    g_hitpos_ao_readback->SetName(L"Hit position readback");

    // Mapping
    desc.Width  = RT_W * RT_H * sizeof(int);
    desc1.Width = desc.Width;
    props1.Type = D3D12_HEAP_TYPE_UPLOAD;
    CE(g_device12->CreateCommittedResource(&props1, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_ray_mapping_upload)));
    CE(g_device12->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&g_ray_mapping)));
    g_ray_mapping_upload->SetName(L"Ray mapping upload");
    g_ray_mapping->SetName(L"Ray mapping");

    // Raydirs
    desc.Width  = RT_W * RT_H * sizeof(float) * 3;
    desc1.Width = desc.Width;
    CE(g_device12->CreateCommittedResource(&props1, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_aoray_dirs_upload)));
    CE(g_device12->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&g_aoray_dirs)));
    g_aoray_dirs->SetName(L"AO ray dirs");
    g_aoray_dirs_upload->SetName(L"AO ray dirs upload");

    // CBV SRV UAV Heap
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.NumDescriptors = 15;  // [0]=output, [1]=BVH, [2]=CBV, [3]=Verts, [4]=Offsets, [5]=HitNormal, [6]=Mapping, [7]=Dirs, [8]=RaysInPix, [9]=RayEntryOffsets, [10]=CompactResults, [11]=CompactAccumColor, [12]=CompactAccumCount, [13]=CompactBatchPixelOffsets, [14]=CompactPixelRayIndices
    heap_desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CE(g_device12->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&g_srv_uav_cbv_heap)));
    g_srv_uav_cbv_heap->SetName(L"SRV UAV CBV heap");
    g_srv_uav_cbv_descriptor_size = g_device12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Query heap
    D3D12_QUERY_HEAP_DESC qhd{};
    qhd.Count    = 8;
    qhd.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.NodeMask = 0;
    CE(g_device12->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g_query_heap)));

    // Query read-back resource
    props.Type     = D3D12_HEAP_TYPE_READBACK;
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Format    = DXGI_FORMAT_UNKNOWN;

    desc.Width     = sizeof(uint64_t) * 8;  // [0]=FrameStart, [1]=FrameEnd  <-- For all cases
                                            // [2]=DispatchRaysStart, [3]=DispatchRaysEnd, [4]=ScatterBegin, [5]=ScatterEnd, [6]=ReduceBegin, [7]=ReduceEnd  <-- Only when compacting
    desc.Flags     = D3D12_RESOURCE_FLAG_NONE;
    CE(g_device12->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_query_readback_buffer)));

    // UAV of RT output resource
    D3D12_CPU_DESCRIPTOR_HANDLE      handle(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device12->CreateUnorderedAccessView(g_rt_output_resource, nullptr, &uav_desc, handle);

    handle.ptr += 5 * g_srv_uav_cbv_descriptor_size;
    uav_desc.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.FirstElement         = 0;
    uav_desc.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_NONE;
    uav_desc.Buffer.NumElements          = RT_W * RT_H;
    uav_desc.Buffer.StructureByteStride  = sizeof(float) * 4;
    g_device12->CreateUnorderedAccessView(g_hitpos_ao, nullptr, &uav_desc, handle);

    uav_desc.Buffer.StructureByteStride = sizeof(float);
    handle.ptr += g_srv_uav_cbv_descriptor_size;
    g_device12->CreateUnorderedAccessView(g_ray_mapping, nullptr, &uav_desc, handle);

    uav_desc.Buffer.StructureByteStride = sizeof(float) * 3;
    handle.ptr += g_srv_uav_cbv_descriptor_size;
    g_device12->CreateUnorderedAccessView(g_aoray_dirs, nullptr, &uav_desc, handle);

    UpdateDispatchRayGpuBuffers();
    CreateStructuredBufferUav(&g_compact_ray_results_buffer, sizeof(float) * 4, 1, 10);
    CreateStructuredBufferUav(&g_compact_accum_color_buffer, sizeof(float) * 4, 1, 11);
    CreateStructuredBufferUav(&g_compact_accum_count_buffer, sizeof(uint32_t), 1, 12);

    // Root params for drawing the FSQUAD
    {
        D3D12_ROOT_PARAMETER root_params[1]{};
        root_params[0].ParameterType    = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_DESCRIPTOR_RANGE desc_ranges[1]{};
        desc_ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  // RT output viewed as SRV
        desc_ranges[0].NumDescriptors                    = 1;
        desc_ranges[0].BaseShaderRegister                = 0;
        desc_ranges[0].RegisterSpace                     = 0;
        desc_ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        root_params[0].DescriptorTable.NumDescriptorRanges = _countof(desc_ranges);
        root_params[0].DescriptorTable.pDescriptorRanges   = desc_ranges;

        D3D12_STATIC_SAMPLER_DESC sampler_desc{};
        sampler_desc.ShaderRegister   = 0;
        sampler_desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        sampler_desc.Filter           = D3D12_FILTER_ANISOTROPIC;
        sampler_desc.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.MipLODBias       = 0;
        sampler_desc.MaxAnisotropy    = 8;
        sampler_desc.ComparisonFunc   = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        sampler_desc.BorderColor      = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        sampler_desc.MinLOD           = 0.0f;
        sampler_desc.MaxLOD           = D3D12_FLOAT32_MAX;
        sampler_desc.RegisterSpace    = 0;

        D3D12_ROOT_SIGNATURE_DESC rootsig_desc{};
        rootsig_desc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        rootsig_desc.NumParameters     = 1;
        rootsig_desc.pParameters       = root_params;
        rootsig_desc.NumStaticSamplers = 1;
        rootsig_desc.pStaticSamplers   = &sampler_desc;

        ID3DBlob *signature{}, *error{};
        D3D12SerializeRootSignature(&rootsig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (error)
        {
            printf("Error: %s\n", (char*)(error->GetBufferPointer()));
        }
        CE(g_device12->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&g_rootsig_fsquad)));
        g_rootsig_fsquad->SetName(L"FSQuad root signature");
        signature->Release();
        if (error)
            error->Release();
    }

    // PSO for fullscreen quad
    {
        ID3DBlob *vs_blob, *ps_blob, *error;
        unsigned  compile_flags = 0;
        D3DCompileFromFile(L"shaders/fsquad.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compile_flags, 0, &vs_blob, &error);
        if (error)
            printf("Error building VS: %s\n", (char*)(error->GetBufferPointer()));

        D3DCompileFromFile(L"shaders/fsquad.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compile_flags, 0, &ps_blob, &error);
        if (error)
            printf("Error building PS: %s\n", (char*)(error->GetBufferPointer()));

        D3D12_INPUT_ELEMENT_DESC input_element_descs[] = {{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA},
                                                          {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA}};

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature                                           = g_rootsig_fsquad;
        pso_desc.VS.pShaderBytecode                                       = vs_blob->GetBufferPointer();
        pso_desc.VS.BytecodeLength                                        = vs_blob->GetBufferSize();
        pso_desc.PS.pShaderBytecode                                       = ps_blob->GetBufferPointer();
        pso_desc.PS.BytecodeLength                                        = ps_blob->GetBufferSize();
        const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlendDesc = {
            FALSE,
            FALSE,
            D3D12_BLEND_ONE,
            D3D12_BLEND_ZERO,
            D3D12_BLEND_OP_ADD,
            D3D12_BLEND_ONE,
            D3D12_BLEND_ZERO,
            D3D12_BLEND_OP_ADD,
            D3D12_LOGIC_OP_NOOP,
            D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        pso_desc.BlendState.AlphaToCoverageEnable  = false;
        pso_desc.BlendState.IndependentBlendEnable = false;
        for (unsigned i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
        {
            pso_desc.BlendState.RenderTarget[i] = defaultRenderTargetBlendDesc;
        }
        pso_desc.SampleMask                            = UINT_MAX;
        pso_desc.RasterizerState.FillMode              = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode              = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.FrontCounterClockwise = FALSE;
        pso_desc.RasterizerState.DepthBias             = D3D12_DEFAULT_DEPTH_BIAS;
        pso_desc.RasterizerState.DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        pso_desc.RasterizerState.SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        pso_desc.RasterizerState.DepthClipEnable       = TRUE;
        pso_desc.RasterizerState.MultisampleEnable     = FALSE;
        pso_desc.RasterizerState.AntialiasedLineEnable = FALSE;
        pso_desc.RasterizerState.ForcedSampleCount     = 0;
        pso_desc.RasterizerState.ConservativeRaster    = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        pso_desc.DepthStencilState.DepthEnable         = FALSE;
        pso_desc.DepthStencilState.StencilEnable       = FALSE;
        pso_desc.InputLayout.pInputElementDescs        = input_element_descs;
        pso_desc.InputLayout.NumElements               = _countof(input_element_descs);
        pso_desc.PrimitiveTopologyType                 = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets                      = 1;
        pso_desc.RTVFormats[0]                         = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.DSVFormat                             = DXGI_FORMAT_UNKNOWN;
        pso_desc.SampleDesc.Count                      = 1;
        CE(g_device12->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&g_pipeline_fsquad)));
        g_pipeline_fsquad->SetName(L"FSQuad pipeline");

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.NumDescriptors = 1;  // [0]=rt_output's SRV
        heap_desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CE(g_device12->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&g_srv_uav_cbv_heap_fsquad)));

        // SRV of RT output resource
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Shader4ComponentMapping   = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Format                    = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv_desc.Texture2D.MipLevels       = 1;
        srv_desc.Texture2D.MostDetailedMip = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE srv_handle(g_srv_uav_cbv_heap_fsquad->GetCPUDescriptorHandleForHeapStart());
        g_device12->CreateShaderResourceView(g_rt_output_resource, &srv_desc, srv_handle);

        // Vertex buffer of FSQUAD
        float verts[][4] = {
            {-1, 1, 0, 0},  // top-left
            {3, 1, 2, 0},   // top-right
            {-1, -3, 0, 2}  // bottom-left
        };
        D3D12_RESOURCE_DESC res_desc{};
        res_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        res_desc.Alignment          = 0;
        res_desc.Height             = 1;
        res_desc.DepthOrArraySize   = 1;
        res_desc.MipLevels          = 1;
        res_desc.Format             = DXGI_FORMAT_UNKNOWN;
        res_desc.SampleDesc.Count   = 1;
        res_desc.SampleDesc.Quality = 0;
        res_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        res_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
        res_desc.Width              = sizeof(verts);

        D3D12_HEAP_PROPERTIES heap_props{};
        heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
        heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_props.CreationNodeMask     = 1;
        heap_props.VisibleNodeMask      = 1;

        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_fsquad_vb)));
        g_fsquad_vb->SetName(L"FSQUAD VB");
        char* mapped;
        g_fsquad_vb->Map(0, nullptr, (void**)(&mapped));
        memcpy(mapped, verts, sizeof(verts));
        g_fsquad_vb->Unmap(0, nullptr);

        g_fsquad_vbv.BufferLocation = g_fsquad_vb->GetGPUVirtualAddress();
        g_fsquad_vbv.SizeInBytes    = sizeof(verts);
        g_fsquad_vbv.StrideInBytes  = sizeof(float) * 4;
    }
}

void CreateRTPipeline()
{
    // 1. Root parameters (global)
    {
        D3D12_ROOT_PARAMETER root_params[4]{};
        root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        root_params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;

        D3D12_DESCRIPTOR_RANGE desc_ranges[5]{};
        desc_ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;  // Output0
        desc_ranges[0].NumDescriptors                    = 1;
        desc_ranges[0].BaseShaderRegister                = 0;
        desc_ranges[0].RegisterSpace                     = 0;
        desc_ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        desc_ranges[1].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  // TLAS
        desc_ranges[1].NumDescriptors                    = 1;
        desc_ranges[1].BaseShaderRegister                = 0;
        desc_ranges[1].RegisterSpace                     = 0;
        desc_ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        desc_ranges[2].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;  // CBV
        desc_ranges[2].NumDescriptors                    = 1;
        desc_ranges[2].BaseShaderRegister                = 0;
        desc_ranges[2].RegisterSpace                     = 0;
        desc_ranges[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        desc_ranges[3].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  // Verts and InstanceOffsets
        desc_ranges[3].NumDescriptors                    = 2;
        desc_ranges[3].BaseShaderRegister                = 1;
        desc_ranges[3].RegisterSpace                     = 0;
        desc_ranges[3].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        root_params[0].DescriptorTable.pDescriptorRanges   = desc_ranges;
        root_params[0].DescriptorTable.NumDescriptorRanges = 4;
        root_params[0].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;

        // For PIX rays specifically
        D3D12_DESCRIPTOR_RANGE desc_ranges1[1]{};
        desc_ranges1[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  // RaysInPix and RayEntryOffsets
        desc_ranges1[0].NumDescriptors                    = 2;
        desc_ranges1[0].BaseShaderRegister                = 3;
        desc_ranges1[0].RegisterSpace                     = 0;
        desc_ranges1[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        root_params[1].DescriptorTable.pDescriptorRanges   = desc_ranges1;
        root_params[1].DescriptorTable.NumDescriptorRanges = 1;
        root_params[1].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        root_params[2].Constants.Num32BitValues            = sizeof(CompactReplayRootConstants) / sizeof(uint32_t);
        root_params[2].Constants.ShaderRegister            = 1;
        root_params[2].Constants.RegisterSpace             = 0;
        root_params[2].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_DESCRIPTOR_RANGE compact_uav_range{};
        compact_uav_range.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        compact_uav_range.NumDescriptors                    = 3;
        compact_uav_range.BaseShaderRegister                = 4;
        compact_uav_range.RegisterSpace                     = 0;
        compact_uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        root_params[3].DescriptorTable.pDescriptorRanges    = &compact_uav_range;
        root_params[3].DescriptorTable.NumDescriptorRanges  = 1;
        root_params[3].ShaderVisibility                     = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rootsig_desc{};
        rootsig_desc.NumStaticSamplers = 0;
        rootsig_desc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        rootsig_desc.NumParameters     = 4;
        rootsig_desc.pParameters       = root_params;

        ID3DBlob *signature, *error;
        D3D12SerializeRootSignature(&rootsig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (error)
        {
            printf("Error: %s\n", (char*)(error->GetBufferPointer()));
        }
        CE(g_device12->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&g_global_rootsig)));
        signature->Release();
        if (error)
            error->Release();

        desc_ranges[4].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;  // Hit position, mapping, Raydirs
        desc_ranges[4].NumDescriptors                    = 3;
        desc_ranges[4].BaseShaderRegister                = 1;
        desc_ranges[4].RegisterSpace                     = 0;
        desc_ranges[4].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        root_params[0].DescriptorTable.NumDescriptorRanges = 5;

        D3D12SerializeRootSignature(&rootsig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (error)
        {
            printf("Error: %s\n", (char*)(error->GetBufferPointer()));
        }
        CE(g_device12->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&g_global_rootsig_ao)));
        signature->Release();
        if (error)
            error->Release();
    }

    // 2. RTPSO for primary ray
    {
        std::vector<D3D12_STATE_SUBOBJECT> subobjects;
        subobjects.reserve(8);

        // 1. DXIL Library
        IDxcBlob*         dxil_library = CompileShaderLibrary(L"shaders/primaryray.hlsl");
        D3D12_EXPORT_DESC dxil_lib_exports[3];
        dxil_lib_exports[0].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[0].ExportToRename = nullptr;
        dxil_lib_exports[0].Name           = L"RayGen";
        dxil_lib_exports[1].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[1].ExportToRename = nullptr;
        dxil_lib_exports[1].Name           = L"ClosestHit";
        dxil_lib_exports[2].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[2].ExportToRename = nullptr;
        dxil_lib_exports[2].Name           = L"Miss";

        D3D12_DXIL_LIBRARY_DESC dxil_lib_desc{};
        dxil_lib_desc.DXILLibrary.pShaderBytecode = dxil_library->GetBufferPointer();
        dxil_lib_desc.DXILLibrary.BytecodeLength  = dxil_library->GetBufferSize();
        dxil_lib_desc.NumExports                  = 3;
        dxil_lib_desc.pExports                    = dxil_lib_exports;

        D3D12_STATE_SUBOBJECT subobj_dxil_lib{};
        subobj_dxil_lib.Type  = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        subobj_dxil_lib.pDesc = &dxil_lib_desc;
        subobjects.push_back(subobj_dxil_lib);

        // 2. Shader Config
        D3D12_RAYTRACING_SHADER_CONFIG shader_config{};
        shader_config.MaxAttributeSizeInBytes = 8;   // float2 bary
        shader_config.MaxPayloadSizeInBytes   = 32;  // float4 color
        D3D12_STATE_SUBOBJECT subobj_shaderconfig{};
        subobj_shaderconfig.Type  = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
        subobj_shaderconfig.pDesc = &shader_config;
        subobjects.push_back(subobj_shaderconfig);

        // 3. Global Root Signature
        D3D12_STATE_SUBOBJECT subobj_global_rootsig{};
        subobj_global_rootsig.Type  = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
        subobj_global_rootsig.pDesc = &g_global_rootsig;
        subobjects.push_back(subobj_global_rootsig);

        // 4. Pipeline config
        D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config{};
        pipeline_config.MaxTraceRecursionDepth = 1;
        D3D12_STATE_SUBOBJECT subobj_pipeline_config{};
        subobj_pipeline_config.Type  = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
        subobj_pipeline_config.pDesc = &pipeline_config;
        subobjects.push_back(subobj_pipeline_config);

        // 5. Hit Group
        D3D12_HIT_GROUP_DESC hitgroup_desc{};
        hitgroup_desc.HitGroupExport           = L"HitGroup";
        hitgroup_desc.ClosestHitShaderImport   = L"ClosestHit";
        hitgroup_desc.AnyHitShaderImport       = nullptr;
        hitgroup_desc.IntersectionShaderImport = nullptr;
        D3D12_STATE_SUBOBJECT subobj_hitgroup  = {};
        subobj_hitgroup.Type                   = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        subobj_hitgroup.pDesc                  = &hitgroup_desc;
        subobjects.push_back(subobj_hitgroup);

        D3D12_STATE_OBJECT_DESC rtpso_desc{};
        rtpso_desc.Type          = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
        rtpso_desc.NumSubobjects = int(subobjects.size());
        rtpso_desc.pSubobjects   = subobjects.data();
        CE(g_device12->CreateStateObject(&rtpso_desc, IID_PPV_ARGS(&g_rt_state_object)));

        g_rt_state_object->QueryInterface(IID_PPV_ARGS(&g_rt_state_object_props));
    }

    // RTPSO for AO rays
    {
        std::vector<D3D12_STATE_SUBOBJECT> subobjects;
        subobjects.reserve(8);

        // 1. DXIL Library
        IDxcBlob*         dxil_library = CompileShaderLibrary(L"shaders/aoray.hlsl");
        D3D12_EXPORT_DESC dxil_lib_exports[6];
        dxil_lib_exports[0].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[0].ExportToRename = nullptr;
        dxil_lib_exports[0].Name           = L"RayGen_primary";
        dxil_lib_exports[1].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[1].ExportToRename = nullptr;
        dxil_lib_exports[1].Name           = L"ClosestHit_primary";
        dxil_lib_exports[2].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[2].ExportToRename = nullptr;
        dxil_lib_exports[2].Name           = L"Miss_primary";
        dxil_lib_exports[3].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[3].ExportToRename = nullptr;
        dxil_lib_exports[3].Name           = L"RayGen_ao";
        dxil_lib_exports[4].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[4].ExportToRename = nullptr;
        dxil_lib_exports[4].Name           = L"ClosestHit_ao";
        dxil_lib_exports[5].Flags          = D3D12_EXPORT_FLAG_NONE;
        dxil_lib_exports[5].ExportToRename = nullptr;
        dxil_lib_exports[5].Name           = L"Miss_ao";

        D3D12_DXIL_LIBRARY_DESC dxil_lib_desc{};
        dxil_lib_desc.DXILLibrary.pShaderBytecode = dxil_library->GetBufferPointer();
        dxil_lib_desc.DXILLibrary.BytecodeLength  = dxil_library->GetBufferSize();
        dxil_lib_desc.NumExports                  = 6;
        dxil_lib_desc.pExports                    = dxil_lib_exports;

        D3D12_STATE_SUBOBJECT subobj_dxil_lib{};
        subobj_dxil_lib.Type  = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        subobj_dxil_lib.pDesc = &dxil_lib_desc;
        subobjects.push_back(subobj_dxil_lib);

        // 2. Shader Config
        D3D12_RAYTRACING_SHADER_CONFIG shader_config{};
        shader_config.MaxAttributeSizeInBytes = 8;   // float2 bary
        shader_config.MaxPayloadSizeInBytes   = 4;  // float4 color, int recursionDepth, float3 normal
        D3D12_STATE_SUBOBJECT subobj_shaderconfig{};
        subobj_shaderconfig.Type  = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
        subobj_shaderconfig.pDesc = &shader_config;
        subobjects.push_back(subobj_shaderconfig);

        // 3. Global Root Signature
        D3D12_STATE_SUBOBJECT subobj_global_rootsig{};
        subobj_global_rootsig.Type  = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
        subobj_global_rootsig.pDesc = &g_global_rootsig_ao;
        subobjects.push_back(subobj_global_rootsig);

        // 4. Pipeline config
        D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config{};
        pipeline_config.MaxTraceRecursionDepth = 1;
        D3D12_STATE_SUBOBJECT subobj_pipeline_config{};
        subobj_pipeline_config.Type  = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
        subobj_pipeline_config.pDesc = &pipeline_config;
        subobjects.push_back(subobj_pipeline_config);

        // 5. Hit Group for primary
        D3D12_HIT_GROUP_DESC hitgroup_desc{};
        hitgroup_desc.HitGroupExport           = L"HitGroup_primary";
        hitgroup_desc.ClosestHitShaderImport   = L"ClosestHit_primary";
        hitgroup_desc.AnyHitShaderImport       = nullptr;
        hitgroup_desc.IntersectionShaderImport = nullptr;
        D3D12_STATE_SUBOBJECT subobj_hitgroup  = {};
        subobj_hitgroup.Type                   = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        subobj_hitgroup.pDesc                  = &hitgroup_desc;
        subobjects.push_back(subobj_hitgroup);

        // 6. Hit Group for ao
        D3D12_HIT_GROUP_DESC hitgroup_desc_ao{};
        hitgroup_desc_ao.HitGroupExport           = L"HitGroup_ao";
        hitgroup_desc_ao.ClosestHitShaderImport   = L"ClosestHit_ao";
        hitgroup_desc_ao.AnyHitShaderImport       = nullptr;
        hitgroup_desc_ao.IntersectionShaderImport = nullptr;
        D3D12_STATE_SUBOBJECT subobj_hitgroup_ao     = {};
        subobj_hitgroup_ao.Type                      = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        subobj_hitgroup_ao.pDesc                     = &hitgroup_desc_ao;
        subobjects.push_back(subobj_hitgroup_ao);

        D3D12_STATE_OBJECT_DESC rtpso_desc{};
        rtpso_desc.Type          = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
        rtpso_desc.NumSubobjects = int(subobjects.size());
        rtpso_desc.pSubobjects   = subobjects.data();
        CE(g_device12->CreateStateObject(&rtpso_desc, IID_PPV_ARGS(&g_rt_state_object_ao)));

        g_rt_state_object_ao->QueryInterface(IID_PPV_ARGS(&g_rt_state_object_props_ao));
    }

    // CB and CBV
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask     = 1;
    heap_props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC cb_desc{};
    cb_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    cb_desc.Alignment          = 0;
    cb_desc.Width              = 256;
    cb_desc.Height             = 1;
    cb_desc.DepthOrArraySize   = 1;
    cb_desc.MipLevels          = 1;
    cb_desc.Format             = DXGI_FORMAT_UNKNOWN;
    cb_desc.SampleDesc.Count   = 1;
    cb_desc.SampleDesc.Quality = 0;
    cb_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    cb_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;

    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &cb_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_raygen_cb)));

    D3D12_CPU_DESCRIPTOR_HANDLE cbv_handle(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    cbv_handle.ptr += 2 * g_srv_uav_cbv_descriptor_size;
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc{};
    cbv_desc.BufferLocation = g_raygen_cb->GetGPUVirtualAddress();
    cbv_desc.SizeInBytes    = 256;
    g_device12->CreateConstantBufferView(&cbv_desc, cbv_handle);
}

int RoundUp(int x, int align)
{
    return align * ((x - 1) / align + 1);
}

void CreateCompactReducePipeline()
{
    D3D12_ROOT_PARAMETER root_params[5]{};
    root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;

    D3D12_DESCRIPTOR_RANGE output_uav_range{};
    output_uav_range.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    output_uav_range.NumDescriptors                    = 1;
    output_uav_range.BaseShaderRegister                = 0;
    output_uav_range.RegisterSpace                     = 0;
    output_uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    root_params[0].DescriptorTable.pDescriptorRanges   = &output_uav_range;
    root_params[0].DescriptorTable.NumDescriptorRanges = 1;
    root_params[0].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_DESCRIPTOR_RANGE compact_uav_range{};
    compact_uav_range.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    compact_uav_range.NumDescriptors                    = 3;
    compact_uav_range.BaseShaderRegister                = 4;
    compact_uav_range.RegisterSpace                     = 0;
    compact_uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    root_params[1].DescriptorTable.pDescriptorRanges    = &compact_uav_range;
    root_params[1].DescriptorTable.NumDescriptorRanges  = 1;
    root_params[1].ShaderVisibility                     = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_DESCRIPTOR_RANGE compact_srv_range{};
    compact_srv_range.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    compact_srv_range.NumDescriptors                    = 2;
    compact_srv_range.BaseShaderRegister                = 5;
    compact_srv_range.RegisterSpace                     = 0;
    compact_srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    root_params[2].DescriptorTable.pDescriptorRanges    = &compact_srv_range;
    root_params[2].DescriptorTable.NumDescriptorRanges  = 1;
    root_params[2].ShaderVisibility                     = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_DESCRIPTOR_RANGE cbv_range{};
    cbv_range.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    cbv_range.NumDescriptors                    = 1;
    cbv_range.BaseShaderRegister                = 0;
    cbv_range.RegisterSpace                     = 0;
    cbv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    root_params[3].DescriptorTable.pDescriptorRanges   = &cbv_range;
    root_params[3].DescriptorTable.NumDescriptorRanges = 1;
    root_params[3].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;

    root_params[4].Constants.Num32BitValues = sizeof(CompactReplayRootConstants) / sizeof(uint32_t);
    root_params[4].Constants.ShaderRegister = 1;
    root_params[4].Constants.RegisterSpace  = 0;
    root_params[4].ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootsig_desc{};
    rootsig_desc.Flags         = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    rootsig_desc.NumParameters = _countof(root_params);
    rootsig_desc.pParameters   = root_params;

    ID3DBlob *signature{}, *error{};
    D3D12SerializeRootSignature(&rootsig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
    if (error)
    {
        printf("Error: %s\n", static_cast<char*>(error->GetBufferPointer()));
    }
    CE(g_device12->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&g_compact_reduce_rootsig)));
    signature->Release();
    if (error)
    {
        error->Release();
    }

    ID3DBlob *cs_blob{}, *cs_error{};
    D3DCompileFromFile(L"shaders/compact_reduce.hlsl", nullptr, nullptr, "CSMain", "cs_5_0", 0, 0, &cs_blob, &cs_error);
    if (cs_error)
    {
        printf("Error building compact reduce CS: %s\n", static_cast<char*>(cs_error->GetBufferPointer()));
        cs_error->Release();
    }
    CE(cs_blob == nullptr ? E_FAIL : S_OK);

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = g_compact_reduce_rootsig;
    pso_desc.CS.pShaderBytecode = cs_blob->GetBufferPointer();
    pso_desc.CS.BytecodeLength = cs_blob->GetBufferSize();
    CE(g_device12->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&g_compact_reduce_pso)));
    cs_blob->Release();
}

void CreateShaderBindingTable()
{
    // Primay ray's SBT
    void* raygen_shader_id = g_rt_state_object_props->GetShaderIdentifier(L"RayGen");
    void* hitgroup_id      = g_rt_state_object_props->GetShaderIdentifier(L"HitGroup");
    void* miss_shader_id   = g_rt_state_object_props->GetShaderIdentifier(L"Miss");

    int shader_record_size = RoundUp(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, 64);

    D3D12_RESOURCE_DESC sbt_desc{};
    sbt_desc.DepthOrArraySize   = 1;
    sbt_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    sbt_desc.Format             = DXGI_FORMAT_UNKNOWN;
    sbt_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
    sbt_desc.Width              = shader_record_size;
    sbt_desc.Height             = 1;
    sbt_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    sbt_desc.SampleDesc.Count   = 1;
    sbt_desc.SampleDesc.Quality = 0;
    sbt_desc.MipLevels          = 1;

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask     = 1;
    heap_props.VisibleNodeMask      = 1;

    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &sbt_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_raygen_sbt_storage)));
    char* mapped;
    g_raygen_sbt_storage->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, raygen_shader_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    g_raygen_sbt_storage->Unmap(0, nullptr);
    g_raygen_sbt_storage->SetName(L"RayGen SBT storage for primary ray");

    sbt_desc.Width = 64;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &sbt_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_hit_sbt_storage)));
    g_hit_sbt_storage->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, hitgroup_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    g_hit_sbt_storage->Unmap(0, nullptr);
    g_hit_sbt_storage->SetName(L"Hit group SBT storage for primary ray");

    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &sbt_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_miss_sbt_storage)));
    g_miss_sbt_storage->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, miss_shader_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    g_miss_sbt_storage->Unmap(0, nullptr);

    // AO ray's SBT
    raygen_shader_id = g_rt_state_object_props_ao->GetShaderIdentifier(L"RayGen_primary");
    hitgroup_id      = g_rt_state_object_props_ao->GetShaderIdentifier(L"HitGroup_primary");
    miss_shader_id   = g_rt_state_object_props_ao->GetShaderIdentifier(L"Miss_primary");
    void* raygen_shader_id_ao = g_rt_state_object_props_ao->GetShaderIdentifier(L"RayGen_ao");
    void* hitgroup_id_ao      = g_rt_state_object_props_ao->GetShaderIdentifier(L"HitGroup_ao");
    void* miss_shader_id_ao   = g_rt_state_object_props_ao->GetShaderIdentifier(L"Miss_ao");

    sbt_desc.Width   = 128;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &sbt_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_raygen_sbt_storage_ao)));
    g_raygen_sbt_storage_ao->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, raygen_shader_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    memcpy(mapped + 64, raygen_shader_id_ao, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    g_raygen_sbt_storage_ao->Unmap(0, nullptr);

    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &sbt_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_hit_sbt_storage_ao)));
    g_hit_sbt_storage_ao->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, hitgroup_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    memcpy(mapped + 64, hitgroup_id_ao, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    g_hit_sbt_storage_ao->Unmap(0, nullptr);

    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &sbt_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_miss_sbt_storage_ao)));
    g_miss_sbt_storage_ao->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, miss_shader_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    memcpy(mapped + 64, miss_shader_id_ao, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    g_miss_sbt_storage_ao->Unmap(0, nullptr);
}

void Render()
{
    if (g_render_backend != RenderBackend::kDxr)
    {
        QueueCpuRender();
        TryConsumeCpuRenderResult();
    }

    DrawImGuiPanel();
    if (g_use_ray_in_pix && g_use_gpu_compact_dispatch_rays && (g_dispatch_ray_mapping_dirty || g_dispatch_ray_gpu_dirty || g_compact_dispatch_replay_dirty))
    {
        UpdateCompactDispatchReplayGpuBuffers();
    }
    else if (g_use_ray_in_pix && (g_dispatch_ray_mapping_dirty || g_dispatch_ray_gpu_dirty))
    {
        UpdateDispatchRayGpuBuffers();
    }

    // Update
    char* mapped;
    g_raygen_cb->Map(0, nullptr, (void**)(&mapped));
    RayGenCB cb{};
    GlmMat4ToDirectXMatrix(&cb.inverse_view, g_inv_view);
    GlmMat4ToDirectXMatrix(&cb.inverse_proj, g_inv_proj);
    cb.invert_y   = g_invert_y;
    cb.ao_samples = g_ao_sample_count;
    cb.use_ray_binning = false;
    cb.ao_radius       = g_ao_radius;
    cb.load_ray_from_buffer = g_use_ray_in_pix ? (g_use_gpu_compact_dispatch_rays ? 5u : 1u) : 0u;
    cb.buffer_w             = g_use_ray_in_pix ? g_gpu_dispatch_ray_dims.x : g_ray_in_pix_dispatch_dims.x;
    cb.buffer_h             = g_use_ray_in_pix ? g_gpu_dispatch_ray_dims.y : g_ray_in_pix_dispatch_dims.y;
    cb.buffer_d             = g_use_ray_in_pix ? g_gpu_dispatch_ray_dims.z : g_ray_in_pix_dispatch_dims.z;
    cb.rt_w                 = static_cast<uint32_t>(RT_W);
    cb.rt_h                 = static_cast<uint32_t>(RT_H);
    memcpy(mapped, &cb, sizeof(RayGenCB));
    g_raygen_cb->Unmap(0, nullptr);

    // Render
    D3D12_CPU_DESCRIPTOR_HANDLE handle_rtv(g_rtv_heap->GetCPUDescriptorHandleForHeapStart());
    handle_rtv.ptr += g_rtv_descriptor_size * g_frame_index;

    float bg_color[] = {0.8f, 1.0f, 0.8f, 1.0f};
    CE(g_command_list->Reset(g_command_allocator, nullptr));

    D3D12_RESOURCE_BARRIER barrier_rtv{};
    barrier_rtv.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier_rtv.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier_rtv.Transition.pResource   = g_rendertargets[g_frame_index];
    barrier_rtv.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier_rtv.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier_rtv.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    g_command_list->ResourceBarrier(1, &barrier_rtv);


    g_command_list->ClearRenderTargetView(handle_rtv, bg_color, 0, nullptr);

    if (g_app_state.as_built.load())
    {
        D3D12_RESOURCE_BARRIER barrier_rt_out = barrier_rtv;
        barrier_rt_out.Transition.pResource   = g_rt_output_resource;

        if (g_render_backend == RenderBackend::kDxr)
        {
            barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);

            g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

            D3D12_GPU_DESCRIPTOR_HANDLE srv_uav_cbv_handle(g_srv_uav_cbv_heap->GetGPUDescriptorHandleForHeapStart());
            D3D12_GPU_DESCRIPTOR_HANDLE pix_rays_dump_handle(g_srv_uav_cbv_heap->GetGPUDescriptorHandleForHeapStart());
            pix_rays_dump_handle.ptr += 8 * g_srv_uav_cbv_descriptor_size;

            D3D12_RESOURCE_BARRIER hitpos_barrier{};
            hitpos_barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            hitpos_barrier.Transition.pResource   = g_hitpos_ao;
            hitpos_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_GENERIC_READ;
            hitpos_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

            D3D12_DISPATCH_RAYS_DESC desc{};
            if (g_use_ao == false)
            {
                g_command_list->SetComputeRootSignature(g_global_rootsig);
                g_command_list->SetDescriptorHeaps(1, &g_srv_uav_cbv_heap);
                g_command_list->SetComputeRootDescriptorTable(0, srv_uav_cbv_handle);
                g_command_list->SetComputeRootDescriptorTable(1, pix_rays_dump_handle);
                CompactReplayRootConstants compact_constants{};
                g_command_list->SetComputeRoot32BitConstants(2, sizeof(CompactReplayRootConstants) / sizeof(uint32_t), &compact_constants, 0);
                g_command_list->SetComputeRootDescriptorTable(3, GpuDescriptor(10));
                g_command_list->SetPipelineState1(g_rt_state_object);
                desc.RayGenerationShaderRecord.StartAddress = g_raygen_sbt_storage->GetGPUVirtualAddress();
                desc.RayGenerationShaderRecord.SizeInBytes  = 64;
                desc.MissShaderTable.StartAddress           = g_miss_sbt_storage->GetGPUVirtualAddress();
                desc.MissShaderTable.SizeInBytes            = 64;
                desc.HitGroupTable.StartAddress             = g_hit_sbt_storage->GetGPUVirtualAddress();
                desc.HitGroupTable.SizeInBytes              = 64;
                if (g_use_ray_in_pix && g_use_gpu_compact_dispatch_rays)
                {
                    g_command_list->SetComputeRootSignature(g_compact_reduce_rootsig);
                    g_command_list->SetComputeRootDescriptorTable(0, GpuDescriptor(0));
                    g_command_list->SetComputeRootDescriptorTable(1, GpuDescriptor(10));
                    g_command_list->SetComputeRootDescriptorTable(2, GpuDescriptor(13));
                    g_command_list->SetComputeRootDescriptorTable(3, GpuDescriptor(2));
                    g_command_list->SetPipelineState(g_compact_reduce_pso);
                    compact_constants.compact_mode = 0;
                    g_command_list->SetComputeRoot32BitConstants(4, sizeof(CompactReplayRootConstants) / sizeof(uint32_t), &compact_constants, 0);
                    g_command_list->Dispatch((RT_W * RT_H + 63) / 64, 1, 1);

                    D3D12_RESOURCE_BARRIER uav_barriers[3]{};
                    for (auto& uav_barrier : uav_barriers)
                    {
                        uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    }
                    uav_barriers[0].UAV.pResource = g_compact_ray_results_buffer;
                    uav_barriers[1].UAV.pResource = g_compact_accum_color_buffer;
                    uav_barriers[2].UAV.pResource = g_compact_accum_count_buffer;
                    g_command_list->ResourceBarrier(3, uav_barriers);

                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 2);
                    for (uint32_t batch = 0; batch + 1 < g_compact_dispatch_replay.batch_offsets.size(); batch++)
                    {
                        const uint32_t batch_begin = g_compact_dispatch_replay.batch_offsets[batch];
                        const uint32_t batch_end = g_compact_dispatch_replay.batch_offsets[batch + 1];
                        const uint32_t batch_count = batch_end - batch_begin;
                        if (batch_count == 0)
                        {
                            continue;
                        }
                        const uint32_t range_begin = g_compact_dispatch_replay.batch_pixel_range_offsets[batch];
                        const uint32_t range_end = g_compact_dispatch_replay.batch_pixel_range_offsets[batch + 1];
                        const uint32_t range_count = range_end - range_begin;
                        if (range_count == 0)
                        {
                            continue;
                        }

                        g_command_list->SetComputeRootSignature(g_global_rootsig);
                        g_command_list->SetComputeRootDescriptorTable(0, srv_uav_cbv_handle);
                        g_command_list->SetComputeRootDescriptorTable(1, pix_rays_dump_handle);
                        compact_constants = {};
                        compact_constants.batch_base = batch_begin;
                        compact_constants.compact_mode = 1;
                        g_command_list->SetComputeRoot32BitConstants(2, sizeof(CompactReplayRootConstants) / sizeof(uint32_t), &compact_constants, 0);
                        g_command_list->SetComputeRootDescriptorTable(3, GpuDescriptor(10));
                        g_command_list->SetPipelineState1(g_rt_state_object);
                        desc.Width = batch_count;
                        desc.Height = 1;
                        desc.Depth = 1;
                        g_command_list->DispatchRays(&desc);
                    }
                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 3);
                    g_command_list->ResourceBarrier(1, &uav_barriers[0]);

                    
                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 4);
                    for (uint32_t batch = 0; batch + 1 < g_compact_dispatch_replay.batch_offsets.size(); batch++)
                    {
                        const uint32_t range_begin = g_compact_dispatch_replay.batch_pixel_range_offsets[batch];
                        const uint32_t range_end = g_compact_dispatch_replay.batch_pixel_range_offsets[batch + 1];
                        const uint32_t range_count = range_end - range_begin;
                        if (range_count == 0)
                        {
                            continue;
                        }
                        g_command_list->SetComputeRootSignature(g_compact_reduce_rootsig);
                        g_command_list->SetComputeRootDescriptorTable(0, GpuDescriptor(0));
                        g_command_list->SetComputeRootDescriptorTable(1, GpuDescriptor(10));
                        g_command_list->SetComputeRootDescriptorTable(2, GpuDescriptor(13));
                        g_command_list->SetComputeRootDescriptorTable(3, GpuDescriptor(2));
                        g_command_list->SetPipelineState(g_compact_reduce_pso);
                        compact_constants = {};
                        compact_constants.batch_base = range_count;
                        compact_constants.compact_mode = 1;
                        compact_constants.pixel_offset_base = range_begin;
                        compact_constants.pixel_index_base = 0;
                        g_command_list->SetComputeRoot32BitConstants(4, sizeof(CompactReplayRootConstants) / sizeof(uint32_t), &compact_constants, 0);
                        g_command_list->Dispatch((range_count + 63) / 64, 1, 1);
                        g_command_list->ResourceBarrier(2, &uav_barriers[1]);
                    }
                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 5);

                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 6);
                    g_command_list->SetComputeRootSignature(g_compact_reduce_rootsig);
                    g_command_list->SetComputeRootDescriptorTable(0, GpuDescriptor(0));
                    g_command_list->SetComputeRootDescriptorTable(1, GpuDescriptor(10));
                    g_command_list->SetComputeRootDescriptorTable(2, GpuDescriptor(13));
                    g_command_list->SetComputeRootDescriptorTable(3, GpuDescriptor(2));
                    g_command_list->SetPipelineState(g_compact_reduce_pso);
                    compact_constants = {};
                    compact_constants.compact_mode = 2;
                    g_command_list->SetComputeRoot32BitConstants(4, sizeof(CompactReplayRootConstants) / sizeof(uint32_t), &compact_constants, 0);
                    g_command_list->Dispatch((RT_W * RT_H + 63) / 64, 1, 1);
                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 7);
                }
                else
                {
                    desc.Width                                  = RT_W;
                    desc.Height                                 = RT_H;
                    desc.Depth                                  = 1;
                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 2);
                    g_command_list->DispatchRays(&desc);
                    g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 3);
                }
            }
            else
            {
                g_command_list->SetComputeRootSignature(g_global_rootsig_ao);
                g_command_list->SetDescriptorHeaps(1, &g_srv_uav_cbv_heap);
                g_command_list->SetComputeRootDescriptorTable(0, srv_uav_cbv_handle);
                g_command_list->SetComputeRootDescriptorTable(1, pix_rays_dump_handle);
                CompactReplayRootConstants compact_constants{};
                g_command_list->SetComputeRoot32BitConstants(2, sizeof(CompactReplayRootConstants) / sizeof(uint32_t), &compact_constants, 0);
                g_command_list->SetComputeRootDescriptorTable(3, GpuDescriptor(10));
                g_command_list->SetPipelineState1(g_rt_state_object_ao);

                g_command_list->ResourceBarrier(1, &hitpos_barrier);

                if (g_force_hitpos_dirty)
                {
                    g_hitpos_dirty = true;
                }
                if (g_hitpos_dirty)
                {
                    desc.RayGenerationShaderRecord.StartAddress = g_raygen_sbt_storage_ao->GetGPUVirtualAddress();
                    desc.RayGenerationShaderRecord.SizeInBytes  = 64;
                    desc.MissShaderTable.StartAddress           = g_miss_sbt_storage_ao->GetGPUVirtualAddress();
                    desc.MissShaderTable.SizeInBytes            = 64;
                    desc.HitGroupTable.StartAddress             = g_hit_sbt_storage_ao->GetGPUVirtualAddress();
                    desc.HitGroupTable.SizeInBytes              = 64;
                    desc.Width                                  = RT_W;
                    desc.Height                                 = RT_H;
                    desc.Depth                                  = 1;
                    g_command_list->DispatchRays(&desc);

                    D3D12_RESOURCE_BARRIER uav_barrier{};
                    uav_barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    uav_barrier.UAV.pResource = g_hitpos_ao;
                    g_command_list->ResourceBarrier(1, &uav_barrier);

                    g_hitpos_dirty = false;
                }

                desc.RayGenerationShaderRecord.StartAddress = g_raygen_sbt_storage_ao->GetGPUVirtualAddress() + 64;
                desc.RayGenerationShaderRecord.SizeInBytes  = 64;
                desc.MissShaderTable.StartAddress           = g_miss_sbt_storage_ao->GetGPUVirtualAddress() + 64;
                desc.MissShaderTable.SizeInBytes            = 64;
                desc.HitGroupTable.StartAddress             = g_hit_sbt_storage_ao->GetGPUVirtualAddress() + 64;
                desc.HitGroupTable.SizeInBytes              = 64;
                desc.Width                                  = RT_W;
                desc.Height                                 = RT_H;
                desc.Depth                                  = 1;
                g_command_list->DispatchRays(&desc);
            }

            g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            const bool measure_compact_dispatch_rays = !g_use_ao && g_use_ray_in_pix && g_use_gpu_compact_dispatch_rays;
            g_command_list->ResolveQueryData(g_query_heap,
                                             D3D12_QUERY_TYPE_TIMESTAMP,
                                             0,
                                             measure_compact_dispatch_rays ? 8 : 4,
                                             g_query_readback_buffer,
                                             0);

            if (g_use_ao)
            {
                hitpos_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                hitpos_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
                g_command_list->ResourceBarrier(1, &hitpos_barrier);

                g_command_list->CopyResource(g_hitpos_ao_readback, g_hitpos_ao);

                hitpos_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                hitpos_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_GENERIC_READ;
                g_command_list->ResourceBarrier(1, &hitpos_barrier);
            }

            barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);
        }
        else
        {
            g_app_state.last_gpu_frame_ms = 0.0f;

            barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);
            UploadCpuRenderTarget();
            barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);
        }

        g_command_list->SetGraphicsRootSignature(g_rootsig_fsquad);
        g_command_list->SetPipelineState(g_pipeline_fsquad);
        g_command_list->SetDescriptorHeaps(1, &g_srv_uav_cbv_heap_fsquad);
        D3D12_GPU_DESCRIPTOR_HANDLE srv_uav_cbv_fsquad_handle(g_srv_uav_cbv_heap_fsquad->GetGPUDescriptorHandleForHeapStart());
        g_command_list->SetGraphicsRootDescriptorTable(0, srv_uav_cbv_fsquad_handle);

        D3D12_VIEWPORT viewport{};
        viewport.TopLeftX = 0;
        viewport.TopLeftY = 0;
        viewport.Width    = static_cast<float>(WIN_W);
        viewport.Height   = static_cast<float>(WIN_H);
        viewport.MinDepth = 0;
        viewport.MaxDepth = 1;

        D3D12_RECT scissor{};
        scissor.left   = 0;
        scissor.top    = 0;
        scissor.right  = WIN_W;
        scissor.bottom = WIN_H;
        g_command_list->RSSetViewports(1, &viewport);
        g_command_list->RSSetScissorRects(1, &scissor);

        g_command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_command_list->IASetVertexBuffers(0, 1, &g_fsquad_vbv);

        g_command_list->OMSetRenderTargets(1, &handle_rtv, false, nullptr);
        g_command_list->DrawInstanced(3, 1, 0, 0);

        barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_command_list->ResourceBarrier(1, &barrier_rt_out);
    }

    g_command_list->OMSetRenderTargets(1, &handle_rtv, false, nullptr);
    g_command_list->SetDescriptorHeaps(1, &g_imgui_srv_heap);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_command_list);

    barrier_rtv.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier_rtv.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    g_command_list->ResourceBarrier(1, &barrier_rtv);

    CE(g_command_list->Close());
    g_command_queue->ExecuteCommandLists(1, (ID3D12CommandList* const*)&g_command_list);
    CE(g_swapchain->Present(1, 0));
    WaitForPreviousFrame();
    CE(g_command_allocator->Reset());

    // Read timer
    float frame_time_ms = 0.0f;
    if (g_render_backend == RenderBackend::kDxr)
    {
        uint64_t freq{0};
        g_command_queue->GetTimestampFrequency(&freq);

        uint64_t timestamps[8]{};
        g_query_readback_buffer->Map(0, nullptr, (void**)(&mapped));
        memcpy(timestamps, mapped, sizeof(timestamps));
        g_query_readback_buffer->Unmap(0, nullptr);
        float sec = (timestamps[1] - timestamps[0]) * 1.0f / freq;

        g_frame_time.AddSample(sec);

        // Update of the following FrameTime's should be synchronized
        frame_time_ms = g_frame_time.GetFrameTime() * 1000.0f;
        const float compact_dispatch_rays_time = g_compact_dispatch_rays_time.GetFrameTime();
        const float scatter_time               = g_compact_scatter_time.GetFrameTime();
        const float dispatch_rays_time         = g_dispatch_rays_time.GetFrameTime();
        const float reduce_time                = g_compact_reduce_time.GetFrameTime();

        g_app_state.last_gpu_frame_ms = frame_time_ms;
        if (!g_use_ao && g_use_ray_in_pix && g_use_gpu_compact_dispatch_rays)
        {
            const float compact_dispatch_sec = (timestamps[3] - timestamps[2]) * 1.0f / freq;
            g_compact_dispatch_rays_time.AddSample(compact_dispatch_sec);
            const float compact_scatter_sec = (timestamps[5] - timestamps[4]) * 1.0f / freq;
            g_compact_scatter_time.AddSample(compact_scatter_sec);
            const float compact_reduce_sec                = (timestamps[7] - timestamps[6]) * 1.0f / freq;
            g_compact_reduce_time.AddSample(compact_reduce_sec);
            g_app_state.last_gpu_compact_dispatch_rays_ms = compact_dispatch_rays_time * 1000.0f;
            g_app_state.last_gpu_compact_scatter_ms       = scatter_time * 1000.0f;
            g_app_state.last_gpu_compact_reduce_ms        = reduce_time * 1000.0f;
            g_app_state.last_gpu_dispatch_rays_ms         = 0.0f;
        }
        else
        {
            const float dispatch_sec                      = (timestamps[3] - timestamps[2]) * 1.0f / freq;
            g_dispatch_rays_time.AddSample(dispatch_sec);
            g_app_state.last_gpu_dispatch_rays_ms         = dispatch_rays_time * 1000.0f;
            g_app_state.last_gpu_compact_dispatch_rays_ms = 0.0f;
            g_app_state.last_gpu_compact_scatter_ms       = 0.0f;
            g_app_state.last_gpu_compact_reduce_ms        = 0.0f;
        }
    }
    if (g_frame_time.ShouldUpdate())
    {
        std::stringstream ss;
        if (!g_app_state.as_built.load())
        {
            std::stringstream ss;
            ss << "MyRRAPlayground " << ToString(g_app_state.scene_stage.load());
            glfwSetWindowTitle(g_window, ss.str().c_str());
        }
        else
        {
            if (g_benchmarkState == BenchmarkState::BENCHMARKING)
            {
                if (g_frame_time.ShouldUpdate())
                {
                    g_bmk_ft_count++;
                    if (g_bmk_ft_count > 5)
                    {
                        
                        g_bmk_ft_count = 0;
                        g_bmk_frametimes.push_back(g_frame_time.GetFrameTime() * 1000);
                        printf("Benchmarked sample count %d/%d = %g ms\n", g_ao_sample_count, BMK_AO_SAMPLE_COUNT_LIMIT, g_bmk_frametimes.back());
                        g_ao_sample_count++;
                        if (g_ao_sample_count > BMK_AO_SAMPLE_COUNT_LIMIT)
                        {
                            g_ao_sample_count = 1;
                            g_benchmarkState  = BenchmarkState::NOT_STARTED;
                            printf("BMK results\n");
                            for (unsigned i = 0; i < g_bmk_frametimes.size(); i++)
                            {
                                printf("%g\n", g_bmk_frametimes[i]);
                            }
                            g_bmk_frametimes.clear();
                        }
                    }
                }
            }

            ss << "MyRRAPlayground [" << ToString(g_render_backend) << "] ";
            ss << "Render res " << std::to_string(RT_W) << "x" << std::to_string(RT_H) << " ";
            ss << std::setprecision(4) << frame_time_ms << "ms/frame";
            if (g_render_backend == RenderBackend::kDxr && g_use_ao)
            {
                ss << " AO rays, " << g_ao_sample_count << " samples";
            }
            else
            {
                ss << " primary_ray";
            }
            if (g_force_hitpos_dirty)
            {
                ss << " force_hitpos_dirty";
            }
            glfwSetWindowTitle(g_window, ss.str().c_str());
        }
    }
}

void CreateAS(const std::vector<std::vector<Vertex>>& vertices, const std::vector<InstanceInfo>& inst_infos)
{
    g_app_state.scene_stage.store(SceneLoadStage::kBuildingGpuBlas);
    g_app_state.blas_total.store(static_cast<uint32_t>(vertices.size()));
    g_app_state.blas_completed.store(0);
    g_app_state.SetStatus("Building GPU BLAS");

    std::vector<ID3D12Resource*> blases;
    std::vector<ID3D12Resource*> transform_buffers;

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instance_descs;
    ID3D12Resource*                             tlas_insts;

    // Overall vertices and offsets
    std::vector<Vertex> all_verts;
    std::vector<int>    blas_offsets, inst_offsets;

    for (uint32_t i_blas = 0; i_blas < vertices.size(); i_blas++)
    {
        ID3D12Resource*            verts_buf;
        size_t                     num_verts = vertices[i_blas].size();
        const std::vector<Vertex>* verts     = &(vertices[i_blas]);

        std::vector<Vertex> dummy = {{{0, 0, 0}}, {{0, 1, 0}}, {{1, 0, 0}}};

        if (num_verts < 1)  // FIXME: Why does BLAS[0] have 0 vertices
        {
            num_verts = 3;
            verts     = &dummy;
        }

        size_t verts_size = sizeof(Vertex) * num_verts;

        blas_offsets.push_back(all_verts.size());
        all_verts.insert(all_verts.end(), verts->begin(), verts->end());

        D3D12_HEAP_PROPERTIES heap_props{};
        heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
        heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_props.CreationNodeMask     = 1;
        heap_props.VisibleNodeMask      = 1;

        D3D12_RESOURCE_DESC res_desc{};
        res_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        res_desc.Alignment          = 0;
        res_desc.Width              = verts_size;
        res_desc.Height             = 1;
        res_desc.DepthOrArraySize   = 1;
        res_desc.MipLevels          = 1;
        res_desc.Format             = DXGI_FORMAT_UNKNOWN;
        res_desc.SampleDesc.Count   = 1;
        res_desc.SampleDesc.Quality = 0;
        res_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        res_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;

        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&verts_buf)));
        char*       mapped{nullptr};
        D3D12_RANGE read_range{};
        read_range.Begin = 0;
        read_range.End   = 0;
        verts_buf->Map(0, &read_range, (void**)(&mapped));
        memcpy(mapped, verts->data(), verts_size);
        verts_buf->Unmap(0, nullptr);
        verts_buf->SetName(L"Verts Buf BLAS");

        D3D12_RAYTRACING_GEOMETRY_DESC geom_desc{};
        geom_desc.Type                                 = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geom_desc.Triangles.VertexBuffer.StartAddress  = verts_buf->GetGPUVirtualAddress();
        geom_desc.Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
        geom_desc.Triangles.VertexCount                = num_verts;
        geom_desc.Triangles.VertexFormat               = DXGI_FORMAT_R32G32B32_FLOAT;
        geom_desc.Triangles.IndexBuffer                = 0;
        geom_desc.Triangles.IndexFormat                = DXGI_FORMAT_UNKNOWN;
        geom_desc.Triangles.IndexCount                 = 0;
        geom_desc.Triangles.Transform3x4               = 0;
        //transform_buf->GetGPUVirtualAddress();
        geom_desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
        inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs       = 1;
        inputs.pGeometryDescs = &geom_desc;
        inputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pb_info{};
        g_device12->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &pb_info);
        printf("BLAS[%u] prebuild info:", i_blas);
        printf(" Scratch: %d", int(pb_info.ScratchDataSizeInBytes));
        printf(", Result : %d\n", int(pb_info.ResultDataMaxSizeInBytes));
        g_app_state.SetStatus(std::string("Building BLAS ") + std::to_string(i_blas + 1) + "/" + std::to_string(vertices.size()));

        D3D12_RESOURCE_DESC scratch_desc{};
        scratch_desc.Alignment          = 0;
        scratch_desc.DepthOrArraySize   = 1;
        scratch_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        scratch_desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        scratch_desc.Format             = DXGI_FORMAT_UNKNOWN;
        scratch_desc.Height             = 1;
        scratch_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        scratch_desc.MipLevels          = 1;
        scratch_desc.SampleDesc.Count   = 1;
        scratch_desc.SampleDesc.Quality = 0;
        scratch_desc.Width              = pb_info.ScratchDataSizeInBytes;

        heap_props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
        heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_props.CreationNodeMask     = 1;
        heap_props.VisibleNodeMask      = 1;

        ID3D12Resource* blas_scratch;
        ID3D12Resource* blas_result;

        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &scratch_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&blas_scratch)));
        blas_scratch->SetName(L"BLAS Scratch");

        D3D12_RESOURCE_DESC result_desc = scratch_desc;
        result_desc.Width               = pb_info.ResultDataMaxSizeInBytes;

        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &result_desc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&blas_result)));
        blas_result->SetName(L"BLAS Result");

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc{};
        build_desc.Inputs.Type                      = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        build_desc.Inputs.DescsLayout               = D3D12_ELEMENTS_LAYOUT_ARRAY;
        build_desc.Inputs.NumDescs                  = 1;
        build_desc.Inputs.pGeometryDescs            = &geom_desc;
        build_desc.DestAccelerationStructureData    = blas_result->GetGPUVirtualAddress();
        build_desc.ScratchAccelerationStructureData = blas_scratch->GetGPUVirtualAddress();
        build_desc.SourceAccelerationStructureData  = 0;
        build_desc.Inputs.Flags                     = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        // Build BLAS
        g_command_list1->Reset(g_command_allocator1, nullptr);
        g_command_list1->BuildRaytracingAccelerationStructure(&build_desc, 0, nullptr);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = blas_result;
        g_command_list1->ResourceBarrier(1, &barrier);

        g_command_list1->Close();
        g_command_queue->ExecuteCommandLists(1, (ID3D12CommandList* const*)(&g_command_list1));
        //WaitForPreviousFrame();

        //blas_scratch->Release();
        blases.push_back(blas_result);
        g_app_state.blas_completed.store(i_blas + 1);
    }

    for (uint32_t i_inst = 0; i_inst < inst_infos.size(); i_inst++)
    {
        const InstanceInfo& info = inst_infos[i_inst];
        ID3D12Resource*     transform_buf;

        inst_offsets.push_back(blas_offsets.at(info.blas_idx));

        D3D12_RESOURCE_DESC res_desc{};
        res_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        res_desc.Alignment          = 0;
        res_desc.Height             = 1;
        res_desc.DepthOrArraySize   = 1;
        res_desc.MipLevels          = 1;
        res_desc.Format             = DXGI_FORMAT_UNKNOWN;
        res_desc.SampleDesc.Count   = 1;
        res_desc.SampleDesc.Quality = 0;
        res_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        res_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
        res_desc.Width              = sizeof(float) * 12;

        D3D12_HEAP_PROPERTIES heap_props{};
        heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
        heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_props.CreationNodeMask     = 1;
        heap_props.VisibleNodeMask      = 1;

        char* mapped{};
        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&transform_buf)));
        D3D12_RANGE read_range{};
        read_range.Begin = read_range.End = 0;
        transform_buf->Map(0, &read_range, (void**)(&mapped));
        memcpy(mapped, info.transform, sizeof(float) * 12);
        transform_buf->Unmap(0, nullptr);

        transform_buffers.push_back(transform_buf);

        D3D12_RAYTRACING_INSTANCE_DESC inst_desc{};
        inst_desc.InstanceID                          = i_inst;
        inst_desc.InstanceContributionToHitGroupIndex = 0;
        inst_desc.Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        memcpy(inst_desc.Transform, info.transform, sizeof(float) * 12);
        inst_desc.AccelerationStructure = blases[info.blas_idx]->GetGPUVirtualAddress();
        inst_desc.InstanceMask          = 0xFF;
        instance_descs.push_back(inst_desc);
    }

    D3D12_RESOURCE_DESC tlas_insts_desc_desc{};
    tlas_insts_desc_desc.Alignment          = 0;
    tlas_insts_desc_desc.DepthOrArraySize   = 1;
    tlas_insts_desc_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    tlas_insts_desc_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
    tlas_insts_desc_desc.Format             = DXGI_FORMAT_UNKNOWN;
    tlas_insts_desc_desc.Height             = 1;
    tlas_insts_desc_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    tlas_insts_desc_desc.MipLevels          = 1;
    tlas_insts_desc_desc.SampleDesc.Count   = 1;
    tlas_insts_desc_desc.SampleDesc.Quality = 0;
    tlas_insts_desc_desc.Width              = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instance_descs.size();

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask     = 1;
    heap_props.VisibleNodeMask      = 1;

    ID3D12Resource* tlas_insts_desc;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &tlas_insts_desc_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&tlas_insts_desc)));
    char* mapped;
    tlas_insts_desc->Map(0, nullptr, (void**)&mapped);
    memcpy(mapped, instance_descs.data(), sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instance_descs.size());
    tlas_insts_desc->Unmap(0, nullptr);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs       = instance_descs.size();
    tlas_inputs.pGeometryDescs = nullptr;
    tlas_inputs.Flags          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pb_info{};
    g_device12->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &pb_info);
    printf("TLAS prebuild info:");
    printf(" Scratch: %d", int(pb_info.ScratchDataSizeInBytes));
    printf(", Result : %d\n", int(pb_info.ResultDataMaxSizeInBytes));
    g_app_state.scene_stage.store(SceneLoadStage::kBuildingGpuTlas);
    g_app_state.tlas_total.store(1);
    g_app_state.tlas_completed.store(0);
    g_app_state.SetStatus("Building GPU TLAS");

    // TLAS
    D3D12_RESOURCE_DESC scratch_desc{};
    scratch_desc.Alignment          = 0;
    scratch_desc.DepthOrArraySize   = 1;
    scratch_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    scratch_desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    scratch_desc.Format             = DXGI_FORMAT_UNKNOWN;
    scratch_desc.Height             = 1;
    scratch_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    scratch_desc.MipLevels          = 1;
    scratch_desc.SampleDesc.Count   = 1;
    scratch_desc.SampleDesc.Quality = 0;
    scratch_desc.Width              = pb_info.ScratchDataSizeInBytes;

    ID3D12Resource* tlas_scratch{};
    ID3D12Resource* tlas_result{};

    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &scratch_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tlas_scratch)));
    tlas_scratch->SetName(L"TLAS Scratch");

    D3D12_RESOURCE_DESC result_desc = scratch_desc;
    result_desc.Width               = pb_info.ResultDataMaxSizeInBytes;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &result_desc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&tlas_result)));
    tlas_result->SetName(L"TLAS Result");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlas_build_desc{};
    tlas_build_desc.Inputs.Type                      = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_build_desc.Inputs.DescsLayout               = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_build_desc.Inputs.InstanceDescs             = tlas_insts_desc->GetGPUVirtualAddress();
    tlas_build_desc.Inputs.NumDescs                  = instance_descs.size();
    tlas_build_desc.DestAccelerationStructureData    = tlas_result->GetGPUVirtualAddress();
    tlas_build_desc.ScratchAccelerationStructureData = tlas_scratch->GetGPUVirtualAddress();
    tlas_build_desc.SourceAccelerationStructureData  = 0;
    tlas_build_desc.Inputs.Flags                     = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    // Build BLAS
    g_command_list1->Reset(g_command_allocator1, nullptr);
    g_command_list1->BuildRaytracingAccelerationStructure(&tlas_build_desc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = tlas_result;
    g_command_list1->ResourceBarrier(1, &barrier);

    g_command_list1->Close();
    g_command_queue->ExecuteCommandLists(1, (ID3D12CommandList* const*)(&g_command_list1));
    //WaitForPreviousFrame();
    g_app_state.tlas_completed.store(1);
    
    // Cannot release until command is done
    // tlas_scratch->Release();

    // SRV of TLAS
    D3D12_CPU_DESCRIPTOR_HANDLE srv_handle(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    srv_handle.ptr += g_srv_uav_cbv_descriptor_size;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format                                   = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srv_desc.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.RaytracingAccelerationStructure.Location = tlas_result->GetGPUVirtualAddress();
    g_device12->CreateShaderResourceView(nullptr, &srv_desc, srv_handle);

    D3D12_RESOURCE_DESC res_desc{};
    res_desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    res_desc.Alignment          = 0;
    res_desc.Height             = 1;
    res_desc.DepthOrArraySize   = 1;
    res_desc.MipLevels          = 1;
    res_desc.Format             = DXGI_FORMAT_UNKNOWN;
    res_desc.SampleDesc.Count   = 1;
    res_desc.SampleDesc.Quality = 0;
    res_desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
    res_desc.Width              = all_verts.size() * sizeof(Vertex);

    heap_props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask     = 1;
    heap_props.VisibleNodeMask      = 1;

    ID3D12Resource* d_all_verts;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&d_all_verts)));
    d_all_verts->Map(0, nullptr, (void**)(&mapped));
    memcpy(mapped, all_verts.data(), res_desc.Width);
    d_all_verts->Unmap(0, nullptr);

    res_desc.Width = inst_offsets.size() * sizeof(int);
    ID3D12Resource* d_inst_offsets;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&d_inst_offsets)));
    d_inst_offsets->Map(0, nullptr, (void**)(&mapped));
    memcpy(mapped, inst_offsets.data(), res_desc.Width);
    d_inst_offsets->Unmap(0, nullptr);

    srv_handle = D3D12_CPU_DESCRIPTOR_HANDLE(g_srv_uav_cbv_heap->GetCPUDescriptorHandleForHeapStart());
    srv_handle.ptr += 3 * g_srv_uav_cbv_descriptor_size;

    srv_desc                            = {};
    srv_desc.Buffer.FirstElement        = 0;
    srv_desc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Buffer.NumElements         = all_verts.size();
    srv_desc.Buffer.StructureByteStride = sizeof(Vertex);
    srv_desc.Format                     = DXGI_FORMAT_UNKNOWN;
    srv_desc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
    g_device12->CreateShaderResourceView(d_all_verts, &srv_desc, srv_handle);

    srv_handle.ptr += g_srv_uav_cbv_descriptor_size;
    srv_desc.Buffer.NumElements         = inst_offsets.size();
    srv_desc.Buffer.StructureByteStride = sizeof(int);
    g_device12->CreateShaderResourceView(d_inst_offsets, &srv_desc, srv_handle);
}

void LoadSceneAndCreateAS(bool rra_file_exists)
{
    g_app_state.scene_stage.store(SceneLoadStage::kIdle);
    g_app_state.dispatch_completed.store(0);
    g_app_state.dispatch_total.store(0);
    g_app_state.SetStatus("Preparing scene");

    SceneData scene;
    if (rra_file_exists)
    {
        if (!LoadSceneFromRra(g_rra_file_name, &g_app_state, &scene))
        {
            return;
        }
    }
    else
    {
        scene = BuildFallbackCubeScene();
        g_app_state.SetSceneStats(scene.stats);
        g_app_state.scene_loaded.store(true);
        g_app_state.blas_total.store(static_cast<uint32_t>(scene.blas_vertices.size()));
        g_app_state.tlas_total.store(1);
        g_app_state.dispatch_completed.store(0);
        g_app_state.dispatch_total.store(0);
        g_app_state.SetStatus("Using fallback cube scene");
    }

    g_scene_aabb_min = scene.stats.scene_aabb_min;
    g_scene_aabb_max = scene.stats.scene_aabb_max;
    g_selected_dispatch_index = 0;
    g_dispatch_ray_mapping_dirty = true;
    g_dispatch_ray_gpu_dirty = true;
    ApplySceneCamera(scene);

    CreateAS(ConvertSceneVertices(scene), ConvertSceneInstances(scene));
    g_cpu_bvh_settings = CurrentCpuBvhSettings();
    g_cpu_bvh_rebuild_requested.store(true);
    g_app_state.SetCpuRenderStats({});
    {
        std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
        g_latest_cpu_result.reset();
        g_pending_cpu_request.reset();
        g_cpu_request_pending = false;
        g_cpu_display_tiles_completed = 0;
        g_cpu_display_tiles_total = 0;
        g_cpu_display_generation = 0;
    }
    g_cpu_refresh_requested = true;
    g_scene_data = std::move(scene);

    g_app_state.scene_stage.store(SceneLoadStage::kReady);
    g_app_state.as_built.store(true);
    g_app_state.SetStatus("DXR scene ready");
    QueueInitialCpuBvhBuild();
}

void ReadPixBufferDump(const char* filename)
{
    g_rays_in_pix_dumpfile_minimal.clear();
    g_ray_in_pix_dispatch_dims = glm::uvec3(0);
    printf("Will print a pix buffer dump, named %s\n", filename);
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs.good())
    {
        printf("Oh! file %s is not good.\n", filename);
        exit(0);
    }
    uint32_t fsize = static_cast<uint32_t>(ifs.tellg());
    printf("File size: %u\n", fsize);
    ifs.close();

    FILE* f = nullptr;
    fopen_s(&f, filename, "rb");

    struct RayInPixBufferDump
    {
        uint32_t   type;
        glm::uvec3 dispatch_rays_idx;
        glm::vec3 origin;
        glm::vec3 direction;
        float     tmin;
        float     tcurrent;
        uint32_t   ray_flags;
    };

    std::vector<RayInPixBufferDump> rays_in_pix;
    const uint32_t                  num_rays = fsize / sizeof(RayInPixBufferDump);

    for (uint32_t i = 0; i < num_rays; i++)
    {
        RayInPixBufferDump r{};
        fread_s(&r, sizeof(r), sizeof(r), 1, f);
        rays_in_pix.push_back(r);
    }

    auto cmp = [&](const RayInPixBufferDump& a, const RayInPixBufferDump& b) {
        auto key_a = std::tie(a.dispatch_rays_idx.z, a.dispatch_rays_idx.y, a.dispatch_rays_idx.x);
        auto key_b = std::tie(b.dispatch_rays_idx.z, b.dispatch_rays_idx.y, b.dispatch_rays_idx.x);
        return key_a < key_b;
    };

    std::sort(rays_in_pix.begin(), rays_in_pix.end(), cmp);

    for (uint32_t i = 0; i < num_rays; i++)
    {
        const auto&             r = rays_in_pix[i];
        RayInPixDumpFileMinimal r1{};
        r1.origin                    = r.origin;
        r1.direction                 = r.direction;
        r1.tmax                      = r.tcurrent;
        r1.tmin                      = r.tmin;
        r1.ray_flags                 = r.ray_flags;
        r1.instance_inclusion_mask   = 0xFF;
        g_ray_in_pix_dispatch_dims.x = std::max(g_ray_in_pix_dispatch_dims.x, r.dispatch_rays_idx.x + 1);
        g_ray_in_pix_dispatch_dims.y = std::max(g_ray_in_pix_dispatch_dims.y, r.dispatch_rays_idx.y + 1);
        g_ray_in_pix_dispatch_dims.z = std::max(g_ray_in_pix_dispatch_dims.z, r.dispatch_rays_idx.z + 1);
        g_rays_in_pix_dumpfile_minimal.push_back(r1);
    }

    fclose(f);
    g_dispatch_ray_mapping_dirty = true;
    g_dispatch_ray_gpu_dirty = true;
    printf("Read %zu rays\n", g_rays_in_pix_dumpfile_minimal.size());
}

int main(int argc, char** argv)
{
    if (argc == 3 && !strcmp(argv[1], "-pixbufferdump"))
    {
        ReadPixBufferDump(argv[2]);
        exit(0);
    }

    g_rra_file_name      = "3DMarkSolarBay-20241020-003039.rra";
    bool rra_file_exists = true;

    for (int i = 0; i < argc; i++)
    {
        if (!strcmp(argv[i], "-w") && i + 1 < argc)
        {
            RT_W = std::atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc)
        {
            RT_H = std::atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)
        {
            g_rra_file_name = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-pixbufferdump") || !strcmp(argv[i], "-p"))
        {
            ReadPixBufferDump(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-setsteadypowerstate") ||
                 !strcmp(argv[i], "-setstablepowerstate"))
        {
            g_set_steady_power_state = true;
        }
    }
    g_rt_width_input = RT_W;
    g_rt_height_input = RT_H;

    if (!std::filesystem::exists(g_rra_file_name))
    {
        printf("Oh! file %s does not exist. Will show a cube instead.\n", g_rra_file_name);
        rra_file_exists = false;
    }

    CreateMyRRALoaderWindow();
    InitDeviceAndCommandQ();
    InitSwapChain();
    InitDX12Stuff();
    InitImGui();
    g_cpu_worker_thread = std::thread(CpuWorkerMain);

    CreateRTPipeline();
    CreateCompactReducePipeline();
    CreateShaderBindingTable();

    std::thread thd([&]() {
        LoadSceneAndCreateAS(rra_file_exists);
    });

    while (!glfwWindowShouldClose(g_window))
    {
        Render();
        glfwPollEvents();
    }

    thd.join();
    {
        std::lock_guard<std::mutex> lock(g_cpu_worker_mutex);
        g_cpu_worker_exit = true;
    }
    g_cpu_worker_cv.notify_one();
    if (g_cpu_worker_thread.joinable())
    {
        g_cpu_worker_thread.join();
    }
    ShutdownImGui();

    return 0;
}
