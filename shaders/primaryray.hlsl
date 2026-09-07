#include "includes.hlsli"

struct Attributes
{
    float2 bary;
};

struct HitInfo
{
    float4 colorAndDistance;
    int idx;
    uint pixel_index;
};

[shader("raygeneration")]
void RayGen()
{
    const float2 uv = DispatchRaysIndex().xy * 1.0 / DispatchRaysDimensions().xy;

    int xx = round(DispatchRaysIndex().x % 16);
    int yy = round(DispatchRaysIndex().y % 16);
    float4 c = { 1, 1, 0, 1 };
    if ((xx < 8 && yy < 8) || (xx >= 8 && yy >= 8))
    {
        c = float4(0.5, 0.5, 0.5, 1);
    }

    float4 ret = c;
    bool should_skip = false;
    RayDesc ray;

    if ((load_ray_from_buffer & 8) != 0)
    {
        const uint work_count = compact_batch_base;
        while (true)
        {
            uint work_index = 0;
            InterlockedAdd(CompactAccumCount[rt_w * rt_h], 1, work_index);
            if (work_index >= work_count)
            {
                return;
            }

            const uint pixel = CompactPixelRayIndices[work_index];
            const uint rayidx_lb = pixel == 0 ? 0 : RayEntryOffsets[pixel - 1];
            const uint rayidx_ub = RayEntryOffsets[pixel];
            const uint nr = rayidx_ub - rayidx_lb;
            if (nr == 0)
            {
                continue;
            }

            float4 sum = float4(0, 0, 0, 0);
            for (uint rayidx = rayidx_lb; rayidx < rayidx_ub; rayidx++)
            {
                RayInPixBufferMinimal rpbm = RaysInPixBufferMinimal[rayidx];
                ray.Origin = rpbm.origin;
                ray.Direction = rpbm.direction;
                ray.TMin = rpbm.tmin;
                ray.TMax = rpbm.tmax;

                HitInfo payload = { float4(0, 0, 0, 1), 0, pixel };
                TraceRay(Scene,
                    rpbm.ray_flags,
                    rpbm.instance_inclusion_mask & 0xFF, 0, 0, 0, ray, payload);
                sum += payload.colorAndDistance;
            }

            const uint x = pixel % rt_w;
            const uint y = pixel / rt_w;
            RenderTarget[uint2(x, y)] = sum / float(nr);
        }
    }

    if ((load_ray_from_buffer & 16) != 0)
    {
        const uint work_count = compact_batch_base;
        while (true)
        {
            uint compact_id = 0;
            InterlockedAdd(CompactAccumCount[rt_w * rt_h], 1, compact_id);
            if (compact_id >= work_count)
            {
                return;
            }

            RayInPixBufferMinimal rpbm = RaysInPixBufferMinimal[compact_id];
            ray.Origin = rpbm.origin;
            ray.Direction = rpbm.direction;
            ray.TMin = rpbm.tmin;
            ray.TMax = rpbm.tmax;

            HitInfo payload = { float4(0, 0, 0, 1), 0, rpbm.original_pixel_index };
            TraceRay(Scene,
                rpbm.ray_flags,
                rpbm.instance_inclusion_mask & 0xFF, 0, 0, 0, ray, payload);
            CompactRayResults[compact_id] = payload.colorAndDistance;
        }
    }

    if ((load_ray_from_buffer & 4) != 0)
    {
        const uint compact_id = compact_batch_base + DispatchRaysIndex().x;
        RayInPixBufferMinimal rpbm = RaysInPixBufferMinimal[compact_id];
        ray.Origin = rpbm.origin;
        ray.Direction = rpbm.direction;
        ray.TMin = rpbm.tmin;
        ray.TMax = rpbm.tmax;

        HitInfo payload = { float4(0, 0, 0, 1), 0, rpbm.original_pixel_index };
        TraceRay(Scene,
            rpbm.ray_flags,
            rpbm.instance_inclusion_mask & 0xFF, 0, 0, 0, ray, payload);
        CompactRayResults[compact_id] = payload.colorAndDistance;
        return;
    }

    if (load_ray_from_buffer > 0)
    {
        const uint2 dixy = DispatchRaysIndex().xy;
        if (dixy.x < buffer_w && dixy.y < buffer_h * buffer_d)
        {
            const uint tidx = dixy.x + dixy.y * buffer_w;
            const uint rayidx_lb = tidx == 0 ? 0 : RayEntryOffsets[tidx - 1];
            const uint rayidx_ub = RayEntryOffsets[tidx];
            const uint nr = rayidx_ub - rayidx_lb;
            if (nr > 0)
            {
                float4 sum = float4(0, 0, 0, 0);
                for (uint rayidx = rayidx_lb; rayidx < rayidx_ub; rayidx++)
                {
                    RayInPixBufferMinimal rpbm = RaysInPixBufferMinimal[rayidx];
                    ray.Origin = rpbm.origin;
                    ray.Direction = rpbm.direction;
                    ray.TMin = rpbm.tmin;
                    ray.TMax = rpbm.tmax;

                    HitInfo payload = { float4(0, 0, 0, 1), 0, tidx };
                    TraceRay(Scene,
                        rpbm.ray_flags,
                        rpbm.instance_inclusion_mask & 0xFF, 0, 0, 0, ray, payload);
                    sum += payload.colorAndDistance;
                }
                ret = sum / float(nr);
            }
            else
            {
                should_skip = true;
            }
        }
        else
        {
            should_skip = true;
        }
    }
    else
    {
        ray.Origin = TransformPosition(inverse_view, float3(0, 0, 0));
        float2 d = (((DispatchRaysIndex().xy + 0.5f) / DispatchRaysDimensions().xy) * 2.f - 1.f);
        if (invert_y)
            d.y *= -1;
        float3 target = TransformPosition(inverse_proj, float3(d.x, -d.y, 1));
        ray.Direction = TransformDirection(inverse_view, normalize(target));
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
    }
    
    if (should_skip == false && load_ray_from_buffer == 0)
    {
        const uint pixel_index = DispatchRaysIndex().x + DispatchRaysIndex().y * rt_w;
        HitInfo payload = { float4(0, 0, 0, 1), 0, pixel_index };
        TraceRay(Scene,
            RAY_FLAG_NONE,
            0xFF, 0, 0, 0, ray, payload);
        ret = payload.colorAndDistance;
    }
    RenderTarget[DispatchRaysIndex().xy] = ret;
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload)
{
    float2 uv = DispatchRaysIndex().xy * 1.0 / DispatchRaysDimensions().xy;
    if ((load_ray_from_buffer & 4) != 0)
    {
        const uint pixel_x = payload.pixel_index % rt_w;
        const uint pixel_y = payload.pixel_index / rt_w;
        uv = float2(pixel_x, pixel_y) / float2(rt_w, rt_h);
    }
    payload.colorAndDistance.x = lerp(0.9, 0.3, uv.y);
    payload.colorAndDistance.y = lerp(0.9, 0.3, uv.y);
    payload.colorAndDistance.z = 0.9;
}

[shader("closesthit")]
void ClosestHit(inout HitInfo payload, Attributes attrib)
{
    uint vert_ofst = InstanceVertOffsets[InstanceIndex()];
    uint vert_idx = PrimitiveIndex() * 3 + vert_ofst;
    float3 v0 = Vertices[vert_idx + 0];
    float3 v1 = Vertices[vert_idx + 1];
    float3 v2 = Vertices[vert_idx + 2];
    float3 v0v1 = v1 - v0, v0v2 = v2 - v0;
    float3 n = normalize(cross(v0v1, v0v2));

    float3x4 o2w = ObjectToWorld3x4();
    n = mul(o2w, float4(n, 0));  // Transform local-space normal to world-space
    
    n = (n + 1.0) / 2.0;
    payload.colorAndDistance.xyz = n;
}
