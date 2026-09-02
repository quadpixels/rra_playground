RWTexture2D<float4> RenderTarget : register(u0);
RWStructuredBuffer<float4> CompactRayResults : register(u4);
RWStructuredBuffer<float4> CompactAccumColor : register(u5);
RWStructuredBuffer<uint> CompactAccumCount : register(u6);
StructuredBuffer<uint> CompactPixelRayIndices : register(t6);

struct CompactBatchPixelRange
{
    uint pixel;
    uint begin;
    uint end;
    uint pad;
};

StructuredBuffer<CompactBatchPixelRange> CompactBatchPixelRanges : register(t5);

cbuffer RayGenCB : register(b0)
{
    float4x4 inverse_view;
    float4x4 inverse_proj;
    bool invert_y;
    int use_ray_binning;
    int ao_samples;
    float ao_radius;
    uint load_ray_from_buffer;
    uint buffer_w;
    uint buffer_h;
    uint buffer_d;
    uint rt_w;
    uint rt_h;
};

cbuffer CompactReplayCB : register(b1)
{
    uint compact_batch_base;
    uint compact_mode;
    uint compact_pixel_offset_base;
    uint compact_pixel_index_base;
};

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatch_id : SV_DispatchThreadID)
{
    if (compact_mode == 0)
    {
        const uint pixel = dispatch_id.x;
        if (pixel >= rt_w * rt_h)
        {
            return;
        }
        CompactAccumColor[pixel] = float4(0, 0, 0, 0);
        CompactAccumCount[pixel] = 0;
        return;
    }

    if (compact_mode == 1)
    {
        if (dispatch_id.x >= compact_batch_base)
        {
            return;
        }
        const CompactBatchPixelRange range = CompactBatchPixelRanges[compact_pixel_offset_base + dispatch_id.x];
        const uint pixel = range.pixel;
        const uint begin = range.begin;
        const uint end = range.end;
        float4 sum = CompactAccumColor[pixel];
        uint count = CompactAccumCount[pixel];
        for (uint i = begin; i < end; i++)
        {
            const uint compact_ray_index = CompactPixelRayIndices[i];
            sum += CompactRayResults[compact_ray_index];
            count++;
        }
        CompactAccumColor[pixel] = sum;
        CompactAccumCount[pixel] = count;
        return;
    }

    const uint pixel = dispatch_id.x;
    if (pixel >= rt_w * rt_h)
    {
        return;
    }
    const uint x = pixel % rt_w;
    const uint y = pixel / rt_w;
    const uint xx = x % 16;
    const uint yy = y % 16;
    float4 c = float4(1, 1, 0, 1);
    if ((xx < 8 && yy < 8) || (xx >= 8 && yy >= 8))
    {
        c = float4(0.5, 0.5, 0.5, 1);
    }

    const uint count = CompactAccumCount[pixel];
    RenderTarget[uint2(x, y)] = count > 0 ? CompactAccumColor[pixel] / float(count) : c;
}
