RaytracingAccelerationStructure Scene : register(t0, space0);
StructuredBuffer<float3> Vertices : register(t1);
StructuredBuffer<int> InstanceVertOffsets : register(t2);

RWTexture2D<float4> RenderTarget : register(u0);

struct Attributes
{
    float2 bary;
};

struct HitInfo
{
    float4 colorAndDistance;
};

float3 TransformPosition(float4x4 m, float3 x)
{
    float4 x4 = float4(x, 0.0f);
    x4 = mul(m, x4);
    x4.x += m[0][3]; // [Col] [Row]
    x4.y += m[1][3];
    x4.z += m[2][3];
    return x4.xyz;
}

float3 TransformDirection(float4x4 m, float3 x)
{
    return (mul(m, float4(x, 0.0f))).xyz;
}

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
        RAY_FLAG_NONE,
        0xFF, 0, 0, 0, ray, payload);
    
    ret = payload.colorAndDistance;
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