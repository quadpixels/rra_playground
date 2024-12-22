RaytracingAccelerationStructure Scene : register(t0, space0);
StructuredBuffer<float3> Vertices : register(t1);
StructuredBuffer<int> InstanceVertOffsets : register(t2);

RWTexture2D<float4> RenderTarget : register(u0);


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

uint2 tea(unsigned int val0, unsigned int val1, unsigned int N)
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

    return uint2(v0, v1);
}

uint lcg(inout int seed)
{
    const unsigned int LCG_A = 1103515245u;
    const unsigned int LCG_C = 12345u;
    const unsigned int LCG_M = 0x00FFFFFFu;
    seed = (LCG_A * seed + LCG_C);
    return seed & LCG_M;
}

float randf(inout int seed)
{
    return float(lcg(seed)) / float(0x01000000);
}

float3 SampleHemisphereCosine(float3 n, inout int seed)
{
    float phi = 2.0f * 3.14159 * randf(seed);
    float sinThetaSqr = randf(seed);
    float sinTheta = sqrt(sinThetaSqr);

    float3 axis = abs(n.x) > 0.001f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 t = cross(axis, n);
    t = normalize(t);
    float3 s = cross(n, t);

    return normalize(s * cos(phi) * sinTheta + t * sin(phi) * sinTheta + n * sqrt(1.0f - sinThetaSqr));
}