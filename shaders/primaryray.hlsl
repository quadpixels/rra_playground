#include "includes.hlsli"

// Hey

struct Attributes
{
    float2 bary;
};

struct HitInfo
{
    float4 colorAndDistance;
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

    if (load_ray_from_buffer & 1)
    {
        uint tidx, iidx;  // Thread idx and invocation idx
        bool reflow = (load_ray_from_buffer & 0x2);
        const uint2 dixy = DispatchRaysIndex().xy;
        
        const uint l = buffer_w * buffer_h * buffer_d;
        
        if (reflow)
        {
            const uint2 ddxy = DispatchRaysDimensions().xy;
            iidx = dixy.x + dixy.y * ddxy.x;
        }
        else
        {
            if (dixy.x >= buffer_w || dixy.y >= buffer_h)
            {
                iidx = l;
            }
            else
            {
                iidx = dixy.x + dixy.y * buffer_w;
            }
        }
        
        if (iidx < l)
        {
            uint rayidx_lb, rayidx_ub;
            if (iidx == 0)
            {
                rayidx_lb = 0;
                rayidx_ub = RayEntryOffsets[0];
            }
            else
            {
                rayidx_lb = RayEntryOffsets[iidx - 1];
                rayidx_ub = RayEntryOffsets[iidx];
            }
            
            uint nr = rayidx_ub - rayidx_lb;
            
            if (nr > 0)
            {
                ret = float4(0, 0, 0, 0);   
                for (uint rayidx = rayidx_lb; rayidx < rayidx_ub; rayidx++)
                {
                    RayInPixBufferMinimal rpbm = RaysInPixBufferMinimal[rayidx];
                    ray.Origin = rpbm.origin;
                    ray.Direction = rpbm.direction;
                    ray.TMin = rpbm.tmin;
                    ray.TMax = 100000.0;
                
                    HitInfo payload = { float4(0, 0, 0, 1) };
                    TraceRay(Scene,
                    ray_flag,
                    0xFF, 0, 0, 0, ray, payload);
                    ret += payload.colorAndDistance;
                }
                ret /= (nr) * 1.0;
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
        
        HitInfo payload = { float4(0, 0, 0, 1) };
        TraceRay(Scene,
        ray_flag,
        0xFF, 0, 0, 0, ray, payload);
        ret = payload.colorAndDistance;
    }
    
    RenderTarget[DispatchRaysIndex().xy] = ret;
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload)
{
    const float2 uv = DispatchRaysIndex().xy * 1.0 / DispatchRaysDimensions().xy;
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