#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
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

#include "public/rra_bvh.h"
#include "public/rra_blas.h"
#include "public/rra_tlas.h"
#include "public/rra_trace_loader.h"
#include "public/rra_ray_history.h"

#undef min
#undef max

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

struct CamParams
{
    glm::vec3 eye, center, up;
    bool      invert_y;
};

std::map<std::string, CamParams> CAM_PARAMS = {
    {"SolarBay", {glm::vec3(5.964f, 1.691f, 5.374f), glm::vec3(2.921f, 1.691f, 2.120f), glm::vec3(0, 1, 0), false}},
    {"PortRoyal", {glm::vec3(-7.2252469f, 0.8361527f, 25.2023430f), glm::vec3(-6.8860960f, 0.8613553f, 24.2284565f), glm::vec3(0, 1, 0), true}},
    {"DXRFeatureTest", {glm::vec3(-6.1447086f, 2.7448003f, -11.9588842f), glm::vec3(-6.1102533f, 2.7394657f, -11.9192486f), glm::vec3(0, 1, 0), true}},
    {"Cyberpunk2077", {glm::vec3(667.6618652f, -804.2122192f, 128.7313995f), glm::vec3(666.0505371f, -802.7095947f, 128.0240326f), glm::vec3(0, 0, 1), true}},
    {"RealTimeDenoisedAmbientOcclusion", {glm::vec3(-43.5119209f, 24.3670177f, -29.0387344f), glm::vec3(-43.2385712f, 24.1981163f, -28.8011036f), glm::vec3(0, 1, 0), true}}
};

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
    int               ao_samples;
};

int WIN_W = 1280, WIN_H = 720;
constexpr const int FRAME_COUNT = 2;

GLFWwindow*      g_window;
bool             g_use_debug_layer{true};
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

ID3D12RootSignature* g_rootsig_fsquad{};
ID3D12PipelineState* g_pipeline_fsquad{};
ID3D12DescriptorHeap* g_srv_uav_cbv_heap_fsquad{};
ID3D12Resource*       g_fsquad_vb;
D3D12_VERTEX_BUFFER_VIEW g_fsquad_vbv;

glm::mat4 g_inv_view;
glm::mat4 g_inv_proj;
bool      g_invert_y = false;

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
ID3D12Resource* g_raygen_sbt_storage_ao;
ID3D12Resource* g_hit_sbt_storage_ao;
ID3D12Resource* g_miss_sbt_storage_ao;
int             g_ao_sample_count{1};

ID3D12QueryHeap* g_query_heap;
ID3D12Resource*  g_query_readback_buffer;

bool g_use_ao{false};

ID3D12Fence* g_fence;
int          g_fence_value;
HANDLE       g_fence_event;
int          g_frame_index;

const char* g_rra_file_name;

std::atomic<bool> g_as_built{false};

void CE(HRESULT x)
{
    if (FAILED(x))
    {
        printf("ERROR: %X\n", x);
        throw std::exception();
    }
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
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_0:
        {
            break;
        }
        case GLFW_KEY_ESCAPE:
        {
            exit(0);
            break;
        }
        case GLFW_KEY_SPACE:
        {
            g_use_ao = !g_use_ao;
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
        }
        default:
            break;
        }
    }
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
    g_fence_event = CreateEvent(nullptr, false, false, L"Fence");
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

    CE(g_device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_command_allocator1)));
    CE(g_device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_command_allocator, nullptr, IID_PPV_ARGS(&g_command_list1)));
    g_command_list1->Close();

    // RT output resource
    D3D12_RESOURCE_DESC desc{};
    desc.DepthOrArraySize = 1;
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.Width            = WIN_W;
    desc.Height           = WIN_H;
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

    // Hit position in world space
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Format    = DXGI_FORMAT_UNKNOWN;
    desc.Width     = WIN_W * WIN_H * sizeof(float) * 4;
    desc.Height    = 1;
    desc.Layout    = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CE(g_device12->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_hitpos_ao)));
    g_hitpos_ao->SetName(L"Hit position");

    // CBV SRV UAV Heap
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.NumDescriptors = 6;  // [0]=output, [1]=BVH, [2]=CBV, [3]=Verts, [4]=Offsets, [5]=HitNormal
    heap_desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CE(g_device12->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&g_srv_uav_cbv_heap)));
    g_srv_uav_cbv_heap->SetName(L"SRV UAV CBV heap");
    g_srv_uav_cbv_descriptor_size = g_device12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Query heap
    D3D12_QUERY_HEAP_DESC qhd{};
    qhd.Count    = 2;
    qhd.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.NodeMask = 0;
    CE(g_device12->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g_query_heap)));

    // Query read-back resource
    props.Type     = D3D12_HEAP_TYPE_READBACK;
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Format    = DXGI_FORMAT_UNKNOWN;
    desc.Width     = sizeof(uint64_t) * 2;
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
    uav_desc.Buffer.NumElements          = WIN_W * WIN_H;
    uav_desc.Buffer.StructureByteStride  = sizeof(float) * 4;
    g_device12->CreateUnorderedAccessView(g_hitpos_ao, nullptr, &uav_desc, handle);

    // Root params for drawing the FSQUAD
    {
        D3D12_ROOT_PARAMETER root_params[1];
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
        sampler_desc.ShaderRegister = 0;
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
        pso_desc.pRootSignature     = g_rootsig_fsquad;
        pso_desc.VS.pShaderBytecode = vs_blob->GetBufferPointer();
        pso_desc.VS.BytecodeLength  = vs_blob->GetBufferSize();
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
        pso_desc.SampleMask                               = UINT_MAX;
        pso_desc.RasterizerState.FillMode                 = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode                 = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.FrontCounterClockwise    = FALSE;
        pso_desc.RasterizerState.DepthBias                = D3D12_DEFAULT_DEPTH_BIAS;
        pso_desc.RasterizerState.DepthBiasClamp           = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        pso_desc.RasterizerState.SlopeScaledDepthBias     = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        pso_desc.RasterizerState.DepthClipEnable          = TRUE;
        pso_desc.RasterizerState.MultisampleEnable        = FALSE;
        pso_desc.RasterizerState.AntialiasedLineEnable    = FALSE;
        pso_desc.RasterizerState.ForcedSampleCount        = 0;
        pso_desc.RasterizerState.ConservativeRaster       = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        pso_desc.DepthStencilState.DepthEnable            = TRUE;
        pso_desc.DepthStencilState.DepthWriteMask         = D3D12_DEPTH_WRITE_MASK_ALL;
        pso_desc.DepthStencilState.DepthFunc              = D3D12_COMPARISON_FUNC_LESS;
        pso_desc.DepthStencilState.StencilEnable          = FALSE;
        pso_desc.DepthStencilState.StencilReadMask        = D3D12_DEFAULT_STENCIL_READ_MASK;
        pso_desc.DepthStencilState.StencilWriteMask       = D3D12_DEFAULT_STENCIL_WRITE_MASK;
        const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
        pso_desc.DepthStencilState.FrontFace              = defaultStencilOp;
        pso_desc.DepthStencilState.BackFace               = defaultStencilOp;
        pso_desc.InputLayout.pInputElementDescs           = input_element_descs;
        pso_desc.InputLayout.NumElements                  = _countof(input_element_descs);
        pso_desc.PrimitiveTopologyType                    = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets                         = 1;
        pso_desc.RTVFormats[0]                            = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.DSVFormat                                = DXGI_FORMAT_D32_FLOAT;
        pso_desc.SampleDesc.Count                         = 1;
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

        ID3D12Resource* d_all_verts;
        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_fsquad_vb)));
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
        D3D12_ROOT_PARAMETER root_params[1];
        root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;

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

        D3D12_ROOT_SIGNATURE_DESC rootsig_desc{};
        rootsig_desc.NumStaticSamplers = 0;
        rootsig_desc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        rootsig_desc.NumParameters     = 1;
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

        desc_ranges[4].RangeType      = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;  // Hit position
        desc_ranges[4].NumDescriptors = 1;
        desc_ranges[4].BaseShaderRegister = 1;
        desc_ranges[4].RegisterSpace      = 0;
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
    // Update
    char* mapped;
    g_raygen_cb->Map(0, nullptr, (void**)(&mapped));
    RayGenCB cb{};
    GlmMat4ToDirectXMatrix(&cb.inverse_view, g_inv_view);
    GlmMat4ToDirectXMatrix(&cb.inverse_proj, g_inv_proj);
    cb.invert_y   = g_invert_y;
    cb.ao_samples = g_ao_sample_count;
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

    if (g_as_built)
    {
        D3D12_RESOURCE_BARRIER barrier_rt_out = barrier_rtv;
        barrier_rt_out.Transition.pResource   = g_rt_output_resource;
        barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        g_command_list->ResourceBarrier(1, &barrier_rt_out);

        g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

        // Dispatch ray
        D3D12_GPU_DESCRIPTOR_HANDLE srv_uav_cbv_handle(g_srv_uav_cbv_heap->GetGPUDescriptorHandleForHeapStart());

        D3D12_DISPATCH_RAYS_DESC desc{};
        if (g_use_ao == false)
        {
            g_command_list->SetComputeRootSignature(g_global_rootsig);
            g_command_list->SetDescriptorHeaps(1, &g_srv_uav_cbv_heap);
            g_command_list->SetComputeRootDescriptorTable(0, srv_uav_cbv_handle);
            g_command_list->SetPipelineState1(g_rt_state_object);
            desc.RayGenerationShaderRecord.StartAddress = g_raygen_sbt_storage->GetGPUVirtualAddress();
            desc.RayGenerationShaderRecord.SizeInBytes  = 64;
            desc.MissShaderTable.StartAddress           = g_miss_sbt_storage->GetGPUVirtualAddress();
            desc.MissShaderTable.SizeInBytes            = 64;
            desc.HitGroupTable.StartAddress             = g_hit_sbt_storage->GetGPUVirtualAddress();
            desc.HitGroupTable.SizeInBytes              = 64;
            desc.Width                                  = WIN_W;
            desc.Height                                 = WIN_H;
            desc.Depth                                  = 1;
            g_command_list->DispatchRays(&desc);
        }
        else
        {
            g_command_list->SetComputeRootSignature(g_global_rootsig_ao);
            g_command_list->SetDescriptorHeaps(1, &g_srv_uav_cbv_heap);
            g_command_list->SetComputeRootDescriptorTable(0, srv_uav_cbv_handle);
            g_command_list->SetPipelineState1(g_rt_state_object_ao);

            D3D12_RESOURCE_BARRIER hitpos_barrier{};
            hitpos_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            hitpos_barrier.Transition.pResource = g_hitpos_ao;
            hitpos_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_GENERIC_READ;
            hitpos_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            g_command_list->ResourceBarrier(1, &hitpos_barrier);

            // Primary Rays
            desc.RayGenerationShaderRecord.StartAddress = g_raygen_sbt_storage_ao->GetGPUVirtualAddress();
            desc.RayGenerationShaderRecord.SizeInBytes  = 64;
            desc.MissShaderTable.StartAddress           = g_miss_sbt_storage_ao->GetGPUVirtualAddress();
            desc.MissShaderTable.SizeInBytes            = 64;
            desc.HitGroupTable.StartAddress             = g_hit_sbt_storage_ao->GetGPUVirtualAddress();
            desc.HitGroupTable.SizeInBytes              = 64;
            desc.Width                                  = WIN_W;
            desc.Height                                 = WIN_H;
            desc.Depth                                  = 1;
            g_command_list->DispatchRays(&desc);

            D3D12_RESOURCE_BARRIER uav_barrier{};
            uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav_barrier.UAV.pResource = g_hitpos_ao;
            g_command_list->ResourceBarrier(1, &uav_barrier);

            // AO Rays
            desc.RayGenerationShaderRecord.StartAddress = g_raygen_sbt_storage_ao->GetGPUVirtualAddress() + 64;
            desc.RayGenerationShaderRecord.SizeInBytes  = 64;
            desc.MissShaderTable.StartAddress           = g_miss_sbt_storage_ao->GetGPUVirtualAddress() + 64;
            desc.MissShaderTable.SizeInBytes            = 64;
            desc.HitGroupTable.StartAddress             = g_hit_sbt_storage_ao->GetGPUVirtualAddress() + 64;
            desc.HitGroupTable.SizeInBytes              = 64;
            desc.Width                                  = WIN_W;
            desc.Height                                 = WIN_H;
            desc.Depth                                  = 1;
            g_command_list->DispatchRays(&desc);

            hitpos_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            hitpos_barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_GENERIC_READ;
            g_command_list->ResourceBarrier(1, &hitpos_barrier);
        }

        g_command_list->EndQuery(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        g_command_list->ResolveQueryData(g_query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, g_query_readback_buffer, 0);

        if (0)  // COPY
        {
            barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);

            barrier_rtv.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier_rtv.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
            g_command_list->ResourceBarrier(1, &barrier_rtv);

            g_command_list->CopyResource(g_rendertargets[g_frame_index], g_rt_output_resource);

            barrier_rtv.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier_rtv.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
            g_command_list->ResourceBarrier(1, &barrier_rtv);
        }
        else  // FSQUAD
        {
            barrier_rt_out.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barrier_rt_out.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);

            g_command_list->SetGraphicsRootSignature(g_rootsig_fsquad);
            g_command_list->SetPipelineState(g_pipeline_fsquad);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_uav_cbv_fsquad_handle(g_srv_uav_cbv_heap_fsquad->GetGPUDescriptorHandleForHeapStart());
            g_command_list->SetGraphicsRootDescriptorTable(0, srv_uav_cbv_fsquad_handle);
            g_command_list->SetDescriptorHeaps(1, &g_srv_uav_cbv_heap_fsquad);

            D3D12_VIEWPORT viewport{};
            viewport.TopLeftX = 0;
            viewport.TopLeftY = 0;
            viewport.Width    = WIN_W;
            viewport.Height   = WIN_H;
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

            barrier_rt_out.Transition.StateBefore  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            barrier_rt_out.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            g_command_list->ResourceBarrier(1, &barrier_rt_out);

            barrier_rtv.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier_rtv.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
            g_command_list->ResourceBarrier(1, &barrier_rtv);
        }
    }
    else
    {
        barrier_rtv.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier_rtv.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        g_command_list->ResourceBarrier(1, &barrier_rtv);
    }

    CE(g_command_list->Close());
    g_command_queue->ExecuteCommandLists(1, (ID3D12CommandList* const*)&g_command_list);
    CE(g_swapchain->Present(1, 0));
    WaitForPreviousFrame();
    CE(g_command_allocator->Reset());

    // Read timer
    uint64_t freq{0};
    g_command_queue->GetTimestampFrequency(&freq);

    uint64_t timestamps[2];
    g_query_readback_buffer->Map(0, nullptr, (void**)(&mapped));
    memcpy(timestamps, mapped, 2 * sizeof(uint64_t));
    g_query_readback_buffer->Unmap(0, nullptr);
    float sec = (timestamps[1] - timestamps[0]) * 1.0f / freq;

    g_frame_time.AddSample(sec);
    if (g_frame_time.ShouldUpdate())
    {
        std::stringstream ss;
        if (!g_as_built)
        {
            // Update in building procedure
        }
        else
        {
            ss << "MyRRAPlayground ";
            ss << std::setprecision(4) << (g_frame_time.GetFrameTime() * 1000) << "ms/frame";
            if (g_use_ao)
            {
                ss << " AO rays, " << g_ao_sample_count << " samples";
            }
            else
            {
                ss << " primary ray";
            }
            glfwSetWindowTitle(g_window, ss.str().c_str());
        }
    }
}

void CreateAS(const std::vector<std::vector<Vertex>>& vertices, const std::vector<InstanceInfo>& inst_infos)
{
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
        glfwSetWindowTitle(g_window, (std::string("Building BLAS ") + std::to_string(i_blas+1) + "/" + std::to_string(vertices.size())).c_str());

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

        D3D12_RESOURCE_DESC result_desc = scratch_desc;
        result_desc.Width               = pb_info.ResultDataMaxSizeInBytes;

        CE(g_device12->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &result_desc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&blas_result)));

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
        WaitForPreviousFrame();

        //blas_scratch->Release();
        blases.push_back(blas_result);
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

    D3D12_RESOURCE_DESC result_desc = scratch_desc;
    result_desc.Width               = pb_info.ResultDataMaxSizeInBytes;
    CE(g_device12->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &result_desc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&tlas_result)));

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
    WaitForPreviousFrame();

    tlas_scratch->Release();

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

void LoadCubeAndCreateAS()
{
    std::vector<std::vector<Vertex>> verts = {{// Front face
                                               {{-1.0, -1.0, 1.0}},
                                               {{1.0, -1.0, 1.0}},
                                               {{1.0, 1.0, 1.0}},

                                               {{-1.0, -1.0, 1.0}},
                                               {{1.0, 1.0, 1.0}},
                                               {{-1.0, 1.0, 1.0}},

                                               // Back face
                                               {{-1.0, -1.0, -1.0}},
                                               {{-1.0, 1.0, -1.0}},
                                               {{1.0, 1.0, -1.0}},

                                               {{-1.0, -1.0, -1.0}},
                                               {{1.0, 1.0, -1.0}},
                                               {{1.0, -1.0, -1.0}},

                                               // Top face
                                               {{-1.0, 1.0, -1.0}},
                                               {{-1.0, 1.0, 1.0}},
                                               {{1.0, 1.0, 1.0}},

                                               {{-1.0, 1.0, -1.0}},
                                               {{1.0, 1.0, 1.0}},
                                               {{1.0, 1.0, -1.0}},

                                               // Bottom face
                                               {{-1.0, -1.0, -1.0}},
                                               {{1.0, -1.0, -1.0}},
                                               {{1.0, -1.0, 1.0}},

                                               {{-1.0, -1.0, -1.0}},
                                               {{1.0, -1.0, 1.0}},
                                               {{-1.0, -1.0, 1.0}},

                                               // Right face
                                               {{1.0, -1.0, -1.0}},
                                               {{1.0, 1.0, -1.0}},
                                               {{1.0, 1.0, 1.0}},

                                               {{1.0, -1.0, -1.0}},
                                               {{1.0, 1.0, 1.0}},
                                               {{1.0, -1.0, 1.0}},

                                               // Left face
                                               {{-1.0, -1.0, -1.0}},
                                               {{-1.0, -1.0, 1.0}},
                                               {{-1.0, 1.0, 1.0}},

                                               {{-1.0, -1.0, -1.0}},
                                               {{-1.0, 1.0, 1.0}},
                                               {{-1.0, 1.0, -1.0}}}};

    InstanceInfo info{};
    info.blas_idx      = 0;
    info.transform[0]  = 1;
    info.transform[5]  = 1;
    info.transform[10] = 1;

    std::vector<InstanceInfo> infos = {info};

    CreateAS(verts, infos);

    // Set Camera
    glm::vec3 eye(2, 2, 5);
    glm::vec3 center(0, 0, 0);
    glm::vec3 up(0, 1, 0);

    glm::mat4 view = glm::lookAt(eye, center, up);
    glm::mat4 proj = glm::perspectiveLH_ZO(glm::radians(90.0f), -1.0f * WIN_W / WIN_H, -0.1f, -499.0f) * (-1.0f);

    g_inv_view = glm::inverse(view);
    g_inv_proj = glm::inverse(proj);

    char* mapped{};
    g_raygen_cb->Map(0, nullptr, (void**)(&mapped));
    RayGenCB cb{};
    GlmMat4ToDirectXMatrix(&cb.inverse_view, g_inv_view);
    GlmMat4ToDirectXMatrix(&cb.inverse_proj, g_inv_proj);
    cb.invert_y = g_invert_y;
    cb.ao_samples = g_ao_sample_count;
    memcpy(mapped, &cb, sizeof(RayGenCB));
    g_raygen_cb->Unmap(0, nullptr);
}

void LoadRRAFileAndCreateAS(const char* rra_file_name)
{
    if (!std::filesystem::exists(rra_file_name))
    {
        printf("%s does not exist.\n", rra_file_name);
        return;
    }

    RraErrorCode ec = RraTraceLoaderLoad(rra_file_name);
    printf("Error: %d\n", static_cast<int>(ec));
    if (ec)
    {
        printf("Error encountered, quitting.\n");
        return;
    }

    {
        time_t  ct = RraTraceLoaderGetCreateTime();
        std::tm tm;
        localtime_s(&tm, &ct);
        char buffer[100];
        std::strftime(buffer, 32, "%a, %Y-%m-%d %H:%M:%S", &tm);
        printf("Trace create time: %s\n", buffer);

        uint64_t tlas_count{}, blas_count{};
        RraBvhGetTlasCount(&tlas_count);
        RraBvhGetBlasCount(&blas_count);
        printf("Trace has %llu TLASs and %llu BLASs\n", tlas_count, blas_count);

        for (unsigned i = 1; i <= blas_count; i++)
        {
            uint32_t cnt{}, cnt1{}, cnt2{}, cnt3{};
            uint64_t addr{};
            RraBlasGetGeometryCount(i, &cnt);
            RraBlasGetProceduralNodeCount(i, &cnt1);
            RraBlasGetTriangleNodeCount(i, &cnt2);
            RraBlasGetUniqueTriangleCount(i, &cnt3);
            RraBlasGetBaseAddress(i, &addr);
            printf("  BLAS[%u] (%llx) has %u geometries, %u proc nodes, %u tri nodes, %u uniq tris\n", i, addr, cnt, cnt1, cnt2, cnt3);
        }
    }

    // TLAS[0]'s instances
    uint32_t tlas0_inst_count{0};
    uint64_t tlas_count{0}, blas_count{0};

    std::vector<InstanceInfo> tlas0_inst_infos;

    // BLAS's vertices
    std::vector<std::vector<Vertex>> vertices;
    uint32_t                         tot_tri_count{0};

    // Rays
    {
        uint32_t dispatch_count{};
        RraRayGetDispatchCount(&dispatch_count);
        printf("dispatch_count=%u\n", dispatch_count);

        for (uint32_t d = 0; d < dispatch_count; d++)
        {
            uint32_t x, y, z;
            if (RraRayGetDispatchDimensions(d, &x, &y, &z) != kRraOk)
                continue;
            printf("  dispatch[%u], dim=(%u,%u,%u)\n", d, x, y, z);
        }
    }

    // Triangles
    {
        RraBvhGetTlasCount(&tlas_count);
        RraBvhGetBlasCount(&blas_count);
        printf("Trace has %llu TLASs and %llu BLASs\n", tlas_count, blas_count);

        uint32_t ptr{};
        RraBvhGetRootNodePtr(&ptr) == kRraOk;

        for (unsigned i = 0; i <= blas_count; i++)
        {
            std::vector<Vertex> geom_verts;

            uint32_t cnt{}, cnt1{}, cnt2{}, cnt3{};
            uint64_t addr;
            RraBlasGetGeometryCount(i, &cnt);
            RraBlasGetProceduralNodeCount(i, &cnt1);
            RraBlasGetTriangleNodeCount(i, &cnt2);
            RraBlasGetUniqueTriangleCount(i, &cnt3);
            RraBlasGetBaseAddress(i, &addr);

            uint32_t root_node{};
            RraBvhGetRootNodePtr(&root_node);
            std::deque<uint32_t> n2v = {root_node};

            if (i > 0)
            {
                float sa{};
                RraBlasGetSurfaceArea(i, root_node, &sa);
                if (sa <= 0)
                {
                    throw std::exception();
                }
            }

            uint32_t num_tris{0};
            while (!n2v.empty())
            {
                uint32_t node = n2v.front();
                n2v.pop_front();

                uint32_t nc{};
                RraBlasGetChildNodeCount(i, node, &nc);
                std::vector<uint32_t> children(nc);
                RraBlasGetChildNodes(i, node, children.data());

                for (uint32_t j = 0; j < children.size(); j++)
                {
                    uint32_t ch = children[j];
                    if (RraBvhIsBoxNode(ch))
                    {
                        n2v.push_back(ch);
                    }
                    else if (RraBvhIsTriangleNode(ch))
                    {
                        float sa{};
                        RraBlasGetSurfaceArea(i, ch, &sa);
                        if (sa <= 0)
                        {
                            printf("BLAS[%u]'s node %08X's surface area is zero\n", i, ch);
                        }

                        uint32_t tc{};
                        if (RraBlasGetNodeTriangleCount(i, ch, &tc) != kRraOk)
                        {
                            continue;
                        }
                        assert(tc < 3);

                        std::vector<VertexPosition> v;
                        v.resize(tc == 1 ? 3 : 4);
                        if (RraBlasGetNodeVertices(i, ch, v.data()) != kRraOk)
                        {
                            continue;
                        }

                        if (sa > 0)
                        {
                            num_tris += tc;
                            if (tc >= 1)
                            {
                                geom_verts.push_back({{v[0].x, v[0].y, v[0].z}});
                                geom_verts.push_back({{v[1].x, v[1].y, v[1].z}});
                                geom_verts.push_back({{v[2].x, v[2].y, v[2].z}});
                                // printf("Tri1:(%g,%g,%g)-(%g,%g,%g)-(%g,%g,%g)\n",
                                //     v[0].x, v[0].y, v[0].z,
                                //     v[1].x, v[1].y, v[1].z,
                                //     v[2].x, v[2].y, v[2].z);
                            }

                            // Note the winding direction of this one.
                            if (tc >= 2)
                            {
                                geom_verts.push_back({{v[1].x, v[1].y, v[1].z}});
                                geom_verts.push_back({{v[3].x, v[3].y, v[3].z}});
                                geom_verts.push_back({{v[2].x, v[2].y, v[2].z}});
                            }
                        }

                        if (sa <= 0)
                        {
                            printf("Tri1:(%g,%g,%g)-(%g,%g,%g)-(%g,%g,%g)\n", v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z, v[2].x, v[2].y, v[2].z);
                            if (tc >= 2)
                            {
                                printf("Tri2:(%g,%g,%g)-(%g,%g,%g)-(%g,%g,%g)\n", v[1].x, v[1].y, v[1].z, v[3].x, v[3].y, v[3].z, v[2].x, v[2].y, v[2].z);
                            }
                        }
                    }
                }
            }
            // printf("  BLAS[%u] (%lx): %u geoms, %u proc & %u tri nodes, %u uniq tris, %u visited\n",
            //     i, addr, cnt, cnt1, cnt2, cnt3, num_tris);
            tot_tri_count += num_tris;
            vertices.push_back(geom_verts);
        }

        // Tlas
        if (tlas_count > 1)
        {
            printf("%zu TLAS detected. Will only make use of the first TLAS.\n", tlas_count);
        }

        for (unsigned i = 0; i < std::min(1, static_cast<int>(tlas_count)); i++)
        {
            uint64_t node_count{};
            uint32_t inst_count{};
            RraTlasGetBoxNodeCount(i, &node_count);

            uint32_t root_node{};
            RraBvhGetRootNodePtr(&root_node);
            std::deque<uint32_t> n2v = {root_node};

            // for (unsigned j=0; j<=blas_count; j++) {
            //     uint64_t x{};
            //     RraTlasGetInstanceCount(i, j, &x) == kRraOk);
            //     inst_count += x;
            // }

            std::vector<InstanceInfo> instance_infos;

            while (!n2v.empty())
            {
                uint32_t node = n2v.front();
                n2v.pop_front();

                uint32_t nc{};
                RraTlasGetChildNodeCount(i, node, &nc);
                std::vector<uint32_t> children(nc);
                RraTlasGetChildNodes(i, node, children.data());

                for (uint32_t j = 0; j < children.size(); j++)
                {
                    uint32_t ch = children[j];
                    if (RraBvhIsBoxNode(ch))
                    {
                        n2v.push_back(ch);
                    }
                    else if (RraBvhIsInstanceNode(ch))
                    {
                        InstanceInfo ii{};
                        RraTlasGetOriginalInstanceNodeTransform(i, ch, ii.transform);
                        RraTlasGetBlasIndexFromInstanceNode(i, ch, &(ii.blas_idx));
                        uint32_t iidx{};
                        RraTlasGetInstanceIndexFromInstanceNode(i, ch, &iidx);
                        if (instance_infos.size() < iidx + 1)
                        {
                            instance_infos.resize(iidx + 1);
                        }
                        instance_infos[iidx] = ii;
                    }
                }
            }
            inst_count = instance_infos.size();

            printf("TLAS %u: %lu nodes, %u insts\n", i, node_count, inst_count);

            tlas0_inst_count = inst_count;
            tlas0_inst_infos = instance_infos;
        }
    }

    CreateAS(vertices, tlas0_inst_infos);

    // Set Camera
    glm::vec3 eye(0, 0, 0);
    glm::vec3 center(0, 1, 0);
    glm::vec3 up(0, 1, 0);
    g_invert_y = false;

    std::string fn(g_rra_file_name);
    for (const auto& entry : CAM_PARAMS)
    {
        if (fn.find(entry.first) != std::string::npos)
        {
            printf("Using camera params for %s\n", entry.first.c_str());
            eye      = entry.second.eye;
            center   = entry.second.center;
            up       = entry.second.up;
            g_invert_y = entry.second.invert_y;
        }
    }

    glm::mat4 view = glm::lookAt(eye, center, up);
    glm::mat4 proj = glm::perspectiveLH_ZO(glm::radians(60.0f), -1.0f * WIN_W / WIN_H, -0.1f, -499.0f) * (-1.0f);

    g_inv_view = glm::inverse(view);
    g_inv_proj = glm::inverse(proj);

    char* mapped{};
    g_raygen_cb->Map(0, nullptr, (void**)(&mapped));
    RayGenCB cb{};
    GlmMat4ToDirectXMatrix(&cb.inverse_view, g_inv_view);
    GlmMat4ToDirectXMatrix(&cb.inverse_proj, g_inv_proj);
    cb.invert_y = g_invert_y;
    cb.ao_samples = g_ao_sample_count;
    memcpy(mapped, &cb, sizeof(RayGenCB));
    g_raygen_cb->Unmap(0, nullptr);
}

int main(int argc, char** argv)
{
    g_rra_file_name      = "3DMarkSolarBay-20241020-003039.rra";
    bool rra_file_exists = true;

    for (int i = 0; i < argc; i++)
    {
        if (!strcmp(argv[i], "-w") && i + 1 < argc)
        {
            WIN_W = std::atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-h") && i + 1 < argc)
        {
            WIN_H = std::atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc)
        {
            g_rra_file_name = argv[i + 1];
            i++;
        }
    }

    if (!std::filesystem::exists(g_rra_file_name))
    {
        printf("Oh! file %s does not exist. Will show a cube instead.\n", g_rra_file_name);
        rra_file_exists = false;
    }

    CreateMyRRALoaderWindow();
    InitDeviceAndCommandQ();
    InitSwapChain();
    InitDX12Stuff();

    CreateRTPipeline();
    CreateShaderBindingTable();

    std::thread thd([&]() {
        if (rra_file_exists)
        {
            LoadRRAFileAndCreateAS(g_rra_file_name);
        }
        else
        {
            LoadCubeAndCreateAS();
        }
        g_as_built = true;
    });

    while (!glfwWindowShouldClose(g_window))
    {
        Render();
        glfwPollEvents();
    }

    thd.join();

    return 0;
}