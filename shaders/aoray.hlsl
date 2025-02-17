#include "includes.hlsli"

RWStructuredBuffer<float4> HitNormalAndT : register(u1);
RWStructuredBuffer<int> RayMapping : register(u2);
RWStructuredBuffer<float3> RayDirs : register(u3);

struct Attributes
{
    float2 bary;
};

struct HitInfo_primary
{
    float tHit;
};

struct HitInfo
{
    float4 colorAndDistance;
    int recursionDepth; // 0: primary, 1+: AO
    float3 normal;
};

[shader("raygeneration")]
void RayGen_primary()
{
    const float2 uv = DispatchRaysIndex().xy * 1.0 / DispatchRaysDimensions().xy;
    const int idx = DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x;

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

    HitInfo_primary payload = { -1 };
    TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
    
    RenderTarget[DispatchRaysIndex().xy] = float4(HitNormalAndT[idx].xyz / 100.0, 1);
}

[shader("miss")]
void Miss_primary(inout HitInfo_primary payload : SV_RayPayload)
{
}

[shader("closesthit")]
void ClosestHit_primary(inout HitInfo_primary payload, Attributes attrib)
{
    uint vert_ofst = InstanceVertOffsets[InstanceIndex()];
    uint vert_idx = PrimitiveIndex() * 3 + vert_ofst;
    float3 v0 = Vertices[vert_idx + 0];
    float3 v1 = Vertices[vert_idx + 1];
    float3 v2 = Vertices[vert_idx + 2];
    float3 v0v1 = v1 - v0, v0v2 = v2 - v0;
    float3 n = normalize(cross(v0v1, v0v2));

    float3x4 o2w = ObjectToWorld3x4();
    n = mul(o2w, float4(n, 0)); // Transform local-space normal to world-space

    const int idx = DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x;
    if (dot(n, WorldRayDirection()) > 0)
    {
        n *= -1;
    }
    HitNormalAndT[idx].xyz = n;
    HitNormalAndT[idx].w = RayTCurrent() - 0.001;
}

[shader("raygeneration")]
void RayGen_ao()
{
    const int idx = DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x;
    RayDesc ray;
    
    float3 origin = TransformPosition(inverse_view, float3(0, 0, 0));
    float2 d;
    float3 n;
    float t;
    if (use_ray_binning)
    {
        int outidx = RayMapping[idx];
        int2 rtidx = { outidx % DispatchRaysDimensions().r, outidx / DispatchRaysDimensions().r };
        d = (((rtidx + 0.5f) / DispatchRaysDimensions().xy) * 2.f - 1.f);
        n = HitNormalAndT[outidx].xyz;
        t = HitNormalAndT[outidx].w;
    }
    else
    {
        d = (((DispatchRaysIndex().xy + 0.5f) / DispatchRaysDimensions().xy) * 2.f - 1.f);
        n = HitNormalAndT[idx].xyz;
        t = HitNormalAndT[idx].w;
    }
    if (invert_y)
        d.y *= -1;
    float3 target = TransformPosition(inverse_proj, float3(d.x, -d.y, 1));
    float3 dir = TransformDirection(inverse_view, normalize(target));
    
    ray.Origin = origin + dir * t;
    ray.TMin = 0.001;
    ray.TMax = ao_radius;

    int ao = 0;
    for (int i = 0; i < ao_samples; i++)
    {
        if (use_ray_binning == 0)
        {
            int seed = tea(DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().r, i, 16).x;
            ray.Direction = SampleHemisphereCosine(n, seed);
        }
        else
        {
            ray.Direction = RayDirs[idx];
        }
        HitInfo_primary payload = { -1 };
        TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, 0, 0, ray, payload);
        if (payload.tHit > 0)
        {
            ao++;
        }
    }

    float ao_occ = (1.0f - (ao * 1.0 / ao_samples)) * 0.8 + 0.2;
    float4 ret;
    ret.xyz = /*ret.xyz*/float3(1, 1, 1) * ao_occ;
    ret.w = 1;
    
    if (use_ray_binning > 0)
    {
        int outidx = RayMapping[idx];
        int2 rtidx = { outidx % DispatchRaysDimensions().r, outidx / DispatchRaysDimensions().r };
        RenderTarget[rtidx] = ret;
    }
    else
    {
        RenderTarget[DispatchRaysIndex().xy] = ret;
    }
}

[shader("miss")]
void Miss_ao(inout HitInfo_primary payload : SV_RayPayload)
{
}

[shader("closesthit")]
void ClosestHit_ao(inout HitInfo_primary payload, Attributes attrib)
{
    payload.tHit = RayTCurrent();
}