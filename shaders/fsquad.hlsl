struct VSInput
{
    float2 position : POSITION;
    float2 texcoord : TEXCOORD0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD0;
};

struct PSOutput
{
    float4 color : SV_Target;
};

Texture2D tex : register(t0);
SamplerState the_sampler;

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    output.position = float4(input.position/* + xyoffset_alpha.xy*/, 0.0f, 1.0f);
    output.texcoord = input.texcoord;
    return output;
}

PSOutput PSMain(VSOutput input)
{
    PSOutput output;
    output.color = tex.Sample(the_sampler, input.texcoord.xy);
    output.color.a = 1.0;
    return output;
}