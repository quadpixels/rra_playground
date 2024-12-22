#include "includes.hlsli"

struct Attributes
{
    float2 bary;
};

struct HitInfo
{
    float4 colorAndDistance;
    int recursionDepth;  // 0: primary, 1+: AO
    float3 normal;
};

cbuffer RayGenCB : register(b0)
{
    float4x4 inverse_view;
    float4x4 inverse_proj;
    bool invert_y;
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
    RayDesc ray;
    float3 origin = TransformPosition(inverse_view, float3(0, 0, 0));
    ray.Origin = origin;
    float2 d = (((DispatchRaysIndex().xy + 0.5f) / DispatchRaysDimensions().xy) * 2.f - 1.f);
    if (invert_y)
        d.y *= -1;
    float3 target = TransformPosition(inverse_proj, float3(d.x, -d.y, 1));
    float3 dir = TransformDirection(inverse_view, normalize(target));
    ray.Direction = dir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    HitInfo payload = { 
        float4(0, 0, 0, 0),
        0,
        float3(0, 0, 0)
    };
    TraceRay(Scene,
        RAY_FLAG_NONE,
        0xFF, 0, 0, 0, ray, payload);
    
    ret.xyz = payload.colorAndDistance.xyz;
    int ao = 0;
    const int AO_SAMPLES = 16;

    if (payload.colorAndDistance.w > 0)
    {
        payload.recursionDepth = 1;
        
        float3 world_p = origin + dir * (payload.colorAndDistance.w - 0.001);
        float3 world_n = payload.normal;
        if (dot(dir, world_n) > 0)
        {
            world_n *= -1;
        }

        for (int i = 0; i < AO_SAMPLES; i++)
        {
            ray.Origin = world_p;
            ray.TMin = 0;
            ray.TMax = 200.0;
            int seed = tea(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().r, i, 16).x;
            ray.Direction = SampleHemisphereCosine(world_n, seed);

            payload.colorAndDistance.w = 0;
            TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, 0, 0, ray, payload);
            if (payload.colorAndDistance.w > 0)
            {
                ao++;
            }
        }
    }
    
    float ao_occ = (1.0f - (ao * 1.0 / AO_SAMPLES)) * 0.8 + 0.2;
    ret.xyz = /*ret.xyz*/float3(1, 1, 1) * ao_occ;
    ret.w = 1;
    RenderTarget[DispatchRaysIndex().xy] = ret;
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload)
{
    const float2 uv = DispatchRaysIndex().xy * 1.0 / DispatchRaysDimensions().xy;
    payload.colorAndDistance.x = lerp(0.9, 0.3, uv.y);
    payload.colorAndDistance.y = lerp(0.9, 0.3, uv.y);
    payload.colorAndDistance.z = 0.9;
    payload.colorAndDistance.w = 0;
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
    
    payload.colorAndDistance.xyz = n * 0.5 + 0.5;
    payload.colorAndDistance.w = RayTCurrent();
    payload.normal = n;
}