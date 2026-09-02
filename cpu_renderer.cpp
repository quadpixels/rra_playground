#include "cpu_renderer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>

namespace
{
using Aabb = CpuPrimaryRayRenderer::Aabb;

void ExpandAabb(Aabb* box, const glm::vec3& p)
{
    box->min = glm::min(box->min, p);
    box->max = glm::max(box->max, p);
}

void ExpandAabb(Aabb* box, const Aabb& other)
{
    box->min = glm::min(box->min, other.min);
    box->max = glm::max(box->max, other.max);
}

struct HitInfo
{
    float    t{std::numeric_limits<float>::max()};
    uint32_t instance_index{0};
    uint32_t first_vertex{0};
    bool     hit{false};
};

glm::vec4 ShadeMiss(uint32_t y, uint32_t height)
{
    const float v = static_cast<float>(y) / static_cast<float>(height);
    const float c = glm::mix(0.9f, 0.3f, v);
    return glm::vec4(c, c, 0.9f, 1.0f);
}

glm::vec4 PackColor(const glm::vec3& rgb)
{
    return glm::vec4(rgb, 1.0f);
}

glm::vec4 SkipColor(uint32_t x, uint32_t y)
{
    const uint32_t xx = x % 16;
    const uint32_t yy = y % 16;
    if ((xx < 8 && yy < 8) || (xx >= 8 && yy >= 8))
    {
        return glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
    }
    return glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);
}

void WriteColor(const glm::vec4& color, uint8_t* out_rgba)
{
    out_rgba[0] = static_cast<uint8_t>(glm::clamp(color.x, 0.0f, 1.0f) * 255.0f);
    out_rgba[1] = static_cast<uint8_t>(glm::clamp(color.y, 0.0f, 1.0f) * 255.0f);
    out_rgba[2] = static_cast<uint8_t>(glm::clamp(color.z, 0.0f, 1.0f) * 255.0f);
    out_rgba[3] = static_cast<uint8_t>(glm::clamp(color.w, 0.0f, 1.0f) * 255.0f);
}

glm::vec3 MakeColorFromWorldNormal(const glm::vec3& local_v0,
                                   const glm::vec3& local_v1,
                                   const glm::vec3& local_v2,
                                   const glm::mat4& object_to_world)
{
    const glm::vec3 local_normal = glm::normalize(glm::cross(local_v1 - local_v0, local_v2 - local_v0));
    const glm::vec3 world_normal = glm::normalize(glm::vec3(object_to_world * glm::vec4(local_normal, 0.0f)));
    return (world_normal + 1.0f) * 0.5f;
}

bool IntersectAabb(const Aabb& box, const glm::vec3& origin, const glm::vec3& inv_dir, float tmin, float tmax)
{
    for (int axis = 0; axis < 3; axis++)
    {
        const float t0 = (box.min[axis] - origin[axis]) * inv_dir[axis];
        const float t1 = (box.max[axis] - origin[axis]) * inv_dir[axis];
        tmin           = std::max(tmin, std::min(t0, t1));
        tmax           = std::min(tmax, std::max(t0, t1));
        if (tmax < tmin)
        {
            return false;
        }
    }
    return true;
}

bool IntersectTriangle(const glm::vec3& origin,
                       const glm::vec3& direction,
                       float tmin,
                       float tmax,
                       const glm::vec3& v0,
                       const glm::vec3& v1,
                       const glm::vec3& v2,
                       CpuRenderStats* stats,
                       float* out_t)
{
    stats->ray_triangle_tests++;
    const glm::vec3 e1 = v1 - v0;
    const glm::vec3 e2 = v2 - v0;
    const glm::vec3 p  = glm::cross(direction, e2);
    const float det    = glm::dot(e1, p);
    if (fabs(det) < 1e-8f)
    {
        return false;
    }

    const float inv_det = 1.0f / det;
    const glm::vec3 s   = origin - v0;
    const float u       = glm::dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    const glm::vec3 q = glm::cross(s, e1);
    const float v     = glm::dot(direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }

    const float t = glm::dot(e2, q) * inv_det;
    if (t <= tmin || t > tmax)
    {
        return false;
    }

    *out_t = t;
    return true;
}
}  // namespace

namespace
{
glm::mat4 Transform3x4ToMat4(const float transform[12])
{
    glm::mat4 out(1.0f);
    out[0][0] = transform[0];
    out[1][0] = transform[1];
    out[2][0] = transform[2];
    out[3][0] = transform[3];
    out[0][1] = transform[4];
    out[1][1] = transform[5];
    out[2][1] = transform[6];
    out[3][1] = transform[7];
    out[0][2] = transform[8];
    out[1][2] = transform[9];
    out[2][2] = transform[10];
    out[3][2] = transform[11];
    return out;
}
}  // namespace

glm::vec3 CpuPrimaryRayRenderer::TransformPosition(const glm::mat4& m, const glm::vec3& x)
{
    glm::vec4 x4(x, 0.0f);
    x4 = m * x4;
    x4.x += m[3][0];
    x4.y += m[3][1];
    x4.z += m[3][2];
    return glm::vec3(x4);
}

glm::vec3 CpuPrimaryRayRenderer::TransformDirection(const glm::mat4& m, const glm::vec3& x)
{
    return glm::vec3(m * glm::vec4(x, 0.0f));
}

namespace
{
Aabb EmptyAabb()
{
    Aabb bounds;
    bounds.min = glm::vec3(std::numeric_limits<float>::max());
    bounds.max = glm::vec3(-std::numeric_limits<float>::max());
    return bounds;
}

Aabb ComputeRefRangeBounds(const std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                                 uint32_t first,
                                 uint32_t count)
{
    Aabb bounds = EmptyAabb();
    for (uint32_t i = 0; i < count; i++)
    {
        ExpandAabb(&bounds, refs[first + i].bounds);
    }
    return bounds;
}

float SurfaceArea(const Aabb& bounds)
{
    const glm::vec3 e = glm::max(bounds.max - bounds.min, glm::vec3(0.0f));
    return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
}

struct BinaryBvhNode
{
    Aabb     bounds{};
    uint32_t left{0};
    uint32_t right{0};
    uint32_t first_primitive{0};
    uint32_t primitive_count{0};
    uint32_t subtree_primitive_count{0};

    bool IsLeaf() const
    {
        return primitive_count > 0;
    }
};

CpuPrimaryRayRenderer::BvhNode::ChildNode MakeChildFromNode(
    const std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes,
    uint32_t node_index)
{
    CpuPrimaryRayRenderer::BvhNode::ChildNode child{};
    child.index = node_index;
    child.aabb  = nodes[node_index].bounds;
    child.type  = nodes[node_index].kind == CpuPrimaryRayRenderer::BvhNode::Kind::kTriangle
                    ? CpuPrimaryRayRenderer::BvhNode::SubnodeType::kTriangle
                    : CpuPrimaryRayRenderer::BvhNode::SubnodeType::kBox;
    return child;
}

uint32_t CreateTriangleNode(const std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                            const std::vector<glm::vec3>& verts,
                            std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes,
                            uint32_t first,
                            uint32_t count)
{
    const uint32_t node_index = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    CpuPrimaryRayRenderer::BvhNode& node = nodes[node_index];
    node.kind           = CpuPrimaryRayRenderer::BvhNode::Kind::kTriangle;
    node.bounds         = ComputeRefRangeBounds(refs, first, count);
    node.triangle_count = count;
    for (uint32_t i = 0; i < count; i++)
    {
        const auto& ref = refs[first + i];
        node.triangles[i].first_vertex = ref.payload;
        node.triangles[i].v0           = verts[ref.payload + 0];
        node.triangles[i].v1           = verts[ref.payload + 1];
        node.triangles[i].v2           = verts[ref.payload + 2];
    }
    return node_index;
}

uint32_t CreateTriangleLeafRange(const std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                                 const std::vector<glm::vec3>& verts,
                                 std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes,
                                 uint32_t first,
                                 uint32_t count,
                                 uint32_t fanout,
                                 uint32_t primitive_node_triangle_capacity)
{
    primitive_node_triangle_capacity = std::clamp(primitive_node_triangle_capacity, 1u, 3u);
    if (count <= primitive_node_triangle_capacity)
    {
        return CreateTriangleNode(refs, verts, nodes, first, count);
    }

    const uint32_t node_index = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    CpuPrimaryRayRenderer::BvhNode& node = nodes[node_index];
    node.kind = CpuPrimaryRayRenderer::BvhNode::Kind::kInternal;
    node.bounds = ComputeRefRangeBounds(refs, first, count);
    const uint32_t child_count = std::min<uint32_t>(fanout, (count + primitive_node_triangle_capacity - 1) / primitive_node_triangle_capacity);
    node.child_count = child_count;
    for (uint32_t i = 0; i < child_count; i++)
    {
        const uint32_t child_first = first + (count * i) / child_count;
        const uint32_t child_end   = first + (count * (i + 1)) / child_count;
        const uint32_t child_node_index = CreateTriangleLeafRange(
            refs, verts, nodes, child_first, child_end - child_first, fanout, primitive_node_triangle_capacity);
        nodes[node_index].children[i] = MakeChildFromNode(nodes, child_node_index);
    }
    return node_index;
}

uint32_t CreateInstanceLeafRange(const std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                                 std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes,
                                 uint32_t first,
                                 uint32_t count,
                                 uint32_t fanout)
{
    const uint32_t node_index = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    CpuPrimaryRayRenderer::BvhNode& node = nodes[node_index];
    node.kind = CpuPrimaryRayRenderer::BvhNode::Kind::kInternal;
    node.bounds = ComputeRefRangeBounds(refs, first, count);
    if (count <= fanout)
    {
        node.child_count = count;
        for (uint32_t i = 0; i < count; i++)
        {
            CpuPrimaryRayRenderer::BvhNode::ChildNode child{};
            child.type  = CpuPrimaryRayRenderer::BvhNode::SubnodeType::kInstance;
            child.aabb   = refs[first + i].bounds;
            child.index  = refs[first + i].payload;
            nodes[node_index].children[i] = child;
        }
        return node_index;
    }

    const uint32_t child_count = std::min<uint32_t>(fanout, count);
    node.child_count = child_count;
    for (uint32_t i = 0; i < child_count; i++)
    {
        const uint32_t child_first = first + (count * i) / child_count;
        const uint32_t child_end   = first + (count * (i + 1)) / child_count;
        const uint32_t child_node_index = CreateInstanceLeafRange(refs, nodes, child_first, child_end - child_first, fanout);
        nodes[node_index].children[i] = MakeChildFromNode(nodes, child_node_index);
    }
    return node_index;
}

uint32_t BuildBinnedSahBinaryRecursive(std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                                       std::vector<BinaryBvhNode>& nodes,
                                       uint32_t first,
                                       uint32_t count,
                                       uint32_t leaf_threshold)
{
    constexpr uint32_t kBinCount      = 16;
    leaf_threshold = std::max(1u, leaf_threshold);

    const uint32_t node_index = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    nodes[node_index].first_primitive = first;
    nodes[node_index].primitive_count = count;
    nodes[node_index].subtree_primitive_count = count;

    Aabb bounds          = EmptyAabb();
    Aabb centroid_bounds = EmptyAabb();
    for (uint32_t i = 0; i < count; i++)
    {
        const auto& ref = refs[first + i];
        ExpandAabb(&bounds, ref.bounds);
        ExpandAabb(&centroid_bounds, ref.centroid);
    }
    nodes[node_index].bounds = bounds;

    if (count <= leaf_threshold)
    {
        return node_index;
    }

    struct SplitCandidate
    {
        float    cost{std::numeric_limits<float>::max()};
        uint32_t axis{0};
        uint32_t split_bin{0};
        bool     valid{false};
    } best;

    for (uint32_t axis = 0; axis < 3; axis++)
    {
        const float extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
        if (extent <= 1e-6f)
        {
            continue;
        }

        struct Bin
        {
            Aabb     bounds{};
            uint32_t count{0};
        };
        std::array<Bin, kBinCount> bins{};
        for (Bin& bin : bins)
        {
            bin.bounds = EmptyAabb();
        }

        const float scale = static_cast<float>(kBinCount) / extent;
        for (uint32_t i = 0; i < count; i++)
        {
            const auto& ref = refs[first + i];
            uint32_t bin_index = static_cast<uint32_t>((ref.centroid[axis] - centroid_bounds.min[axis]) * scale);
            bin_index = std::min(bin_index, kBinCount - 1);
            bins[bin_index].count++;
            ExpandAabb(&bins[bin_index].bounds, ref.bounds);
        }

        std::array<Aabb, kBinCount - 1> left_bounds{};
        std::array<Aabb, kBinCount - 1> right_bounds{};
        std::array<uint32_t, kBinCount - 1> left_counts{};
        std::array<uint32_t, kBinCount - 1> right_counts{};

        Aabb running_bounds = EmptyAabb();
        uint32_t running_count = 0;
        for (uint32_t i = 0; i < kBinCount - 1; i++)
        {
            if (bins[i].count > 0)
            {
                ExpandAabb(&running_bounds, bins[i].bounds);
                running_count += bins[i].count;
            }
            left_bounds[i] = running_bounds;
            left_counts[i] = running_count;
        }

        running_bounds = EmptyAabb();
        running_count = 0;
        for (uint32_t i = kBinCount - 1; i > 0; i--)
        {
            if (bins[i].count > 0)
            {
                ExpandAabb(&running_bounds, bins[i].bounds);
                running_count += bins[i].count;
            }
            right_bounds[i - 1] = running_bounds;
            right_counts[i - 1] = running_count;
        }

        for (uint32_t split = 0; split < kBinCount - 1; split++)
        {
            if (left_counts[split] == 0 || right_counts[split] == 0)
            {
                continue;
            }
            const float cost =
                SurfaceArea(left_bounds[split]) * static_cast<float>(left_counts[split]) +
                SurfaceArea(right_bounds[split]) * static_cast<float>(right_counts[split]);
            if (cost < best.cost)
            {
                best.cost      = cost;
                best.axis      = axis;
                best.split_bin = split;
                best.valid     = true;
            }
        }
    }

    if (!best.valid)
    {
        uint32_t axis = 0;
        const glm::vec3 extent = centroid_bounds.max - centroid_bounds.min;
        if (extent.y > extent.x)
        {
            axis = 1;
        }
        if (extent.z > extent[axis])
        {
            axis = 2;
        }

        const uint32_t mid = first + count / 2;
        std::nth_element(refs.begin() + first,
                         refs.begin() + mid,
                         refs.begin() + first + count,
                         [axis](const auto& a, const auto& b) {
                             return a.centroid[axis] < b.centroid[axis];
                         });

        nodes[node_index].primitive_count = 0;
        const uint32_t left_index  = BuildBinnedSahBinaryRecursive(refs, nodes, first, mid - first, leaf_threshold);
        const uint32_t right_index = BuildBinnedSahBinaryRecursive(refs, nodes, mid, count - (mid - first), leaf_threshold);
        nodes[node_index].left  = left_index;
        nodes[node_index].right = right_index;
        return node_index;
    }

    const float best_extent = centroid_bounds.max[best.axis] - centroid_bounds.min[best.axis];
    const float scale = static_cast<float>(kBinCount) / best_extent;
    const auto split_it = std::partition(
        refs.begin() + first,
        refs.begin() + first + count,
        [&](const auto& ref) {
            uint32_t bin_index = static_cast<uint32_t>((ref.centroid[best.axis] - centroid_bounds.min[best.axis]) * scale);
            bin_index = std::min(bin_index, kBinCount - 1);
            return bin_index <= best.split_bin;
        });

    const uint32_t left_count = static_cast<uint32_t>(split_it - (refs.begin() + first));
    if (left_count == 0 || left_count == count)
    {
        const uint32_t mid = first + count / 2;
        nodes[node_index].primitive_count = 0;
        const uint32_t left_index  = BuildBinnedSahBinaryRecursive(refs, nodes, first, mid - first, leaf_threshold);
        const uint32_t right_index = BuildBinnedSahBinaryRecursive(refs, nodes, mid, count - (mid - first), leaf_threshold);
        nodes[node_index].left  = left_index;
        nodes[node_index].right = right_index;
        return node_index;
    }

    nodes[node_index].primitive_count = 0;
    const uint32_t left_index  = BuildBinnedSahBinaryRecursive(refs, nodes, first, left_count, leaf_threshold);
    const uint32_t right_index = BuildBinnedSahBinaryRecursive(refs, nodes, first + left_count, count - left_count, leaf_threshold);
    nodes[node_index].left  = left_index;
    nodes[node_index].right = right_index;
    return node_index;
}

uint32_t CollapseBinaryBvhRecursive(const std::vector<BinaryBvhNode>& binary_nodes,
                                    uint32_t binary_index,
                                    const std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                                    const std::vector<glm::vec3>* triangle_vertices,
                                    std::vector<CpuPrimaryRayRenderer::BvhNode>& wide_nodes,
                                    uint32_t fanout,
                                    uint32_t primitive_node_triangle_capacity)
{
    const BinaryBvhNode& binary = binary_nodes[binary_index];
    if (binary.IsLeaf())
    {
        if (triangle_vertices != nullptr)
        {
            return CreateTriangleLeafRange(refs,
                                           *triangle_vertices,
                                           wide_nodes,
                                           binary.first_primitive,
                                           binary.primitive_count,
                                           fanout,
                                           primitive_node_triangle_capacity);
        }
        return CreateInstanceLeafRange(refs, wide_nodes, binary.first_primitive, binary.primitive_count, fanout);
    }

    const uint32_t wide_index = static_cast<uint32_t>(wide_nodes.size());
    wide_nodes.push_back({});
    wide_nodes[wide_index].kind = CpuPrimaryRayRenderer::BvhNode::Kind::kInternal;
    wide_nodes[wide_index].bounds = binary.bounds;

    std::vector<uint32_t> frontier{binary.left, binary.right};
    while (frontier.size() < fanout)
    {
        auto expand_it = frontier.end();

        float best_area = -1.0f;
        for (auto it = frontier.begin(); it != frontier.end(); ++it)
        {
            const BinaryBvhNode& candidate = binary_nodes[*it];
            if (candidate.IsLeaf())
            {
                continue;
            }

            const float area = SurfaceArea(candidate.bounds);
            if (area > best_area)
            {
                expand_it = it;
                best_area = area;
            }
        }

        if (expand_it == frontier.end())
        {
            break;
        }

        const BinaryBvhNode& expand_node = binary_nodes[*expand_it];
        *expand_it = expand_node.left;
        frontier.insert(expand_it + 1, expand_node.right);
    }

    wide_nodes[wide_index].child_count = static_cast<uint32_t>(frontier.size());
    for (uint32_t i = 0; i < frontier.size(); i++)
    {
        const uint32_t child_node_index =
            CollapseBinaryBvhRecursive(
                binary_nodes, frontier[i], refs, triangle_vertices, wide_nodes, fanout, primitive_node_triangle_capacity);
        wide_nodes[wide_index].children[i] = MakeChildFromNode(wide_nodes, child_node_index);
    }
    return wide_index;
}

uint32_t BuildWideBvhRecursive(std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                               const std::vector<glm::vec3>* triangle_vertices,
                               std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes,
                               uint32_t first,
                               uint32_t count,
                               uint32_t fanout,
                               uint32_t primitive_node_triangle_capacity)
{
    Aabb bounds          = EmptyAabb();
    Aabb centroid_bounds = EmptyAabb();
    for (uint32_t i = 0; i < count; i++)
    {
        const auto& ref = refs[first + i];
        ExpandAabb(&bounds, ref.bounds);
        ExpandAabb(&centroid_bounds, ref.centroid);
    }

    if (count <= 4)
    {
        if (triangle_vertices != nullptr)
        {
            return CreateTriangleLeafRange(refs, *triangle_vertices, nodes, first, count, fanout, primitive_node_triangle_capacity);
        }
        return CreateInstanceLeafRange(refs, nodes, first, count, fanout);
    }

    const glm::vec3 extent = centroid_bounds.max - centroid_bounds.min;
    int axis               = 0;
    if (extent.y > extent.x)
    {
        axis = 1;
    }
    if (extent.z > extent[axis])
    {
        axis = 2;
    }

    if (extent[axis] <= 1e-6f)
    {
        axis = 0;
    }

    std::sort(refs.begin() + first,
              refs.begin() + first + count,
              [axis](const auto& a, const auto& b) {
                  return a.centroid[axis] < b.centroid[axis];
              });

    const uint32_t child_count = std::min<uint32_t>(fanout, count);
    const uint32_t node_index = static_cast<uint32_t>(nodes.size());
    nodes.push_back({});
    nodes[node_index].kind = CpuPrimaryRayRenderer::BvhNode::Kind::kInternal;
    nodes[node_index].bounds = bounds;
    nodes[node_index].child_count     = child_count;
    for (uint32_t child = 0; child < child_count; child++)
    {
        const uint32_t child_first = first + (count * child) / child_count;
        const uint32_t child_end   = first + (count * (child + 1)) / child_count;
        const uint32_t child_node_index =
            BuildWideBvhRecursive(
                refs, triangle_vertices, nodes, child_first, child_end - child_first, fanout, primitive_node_triangle_capacity);
        nodes[node_index].children[child] = MakeChildFromNode(nodes, child_node_index);
    }
    return node_index;
}

uint32_t BuildConfiguredBvh(std::vector<CpuPrimaryRayRenderer::BuildRef>& refs,
                            const std::vector<glm::vec3>* triangle_vertices,
                            std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes,
                            uint32_t fanout,
                            uint32_t primitive_node_triangle_capacity,
                            CpuBvhSplitMode split_mode,
                            bool* used_binned_sah)
{
    nodes.clear();
    if (refs.empty())
    {
        return 0;
    }

    if (split_mode == CpuBvhSplitMode::kBinnedSah)
    {
        std::vector<BinaryBvhNode> binary_nodes;
        binary_nodes.reserve(refs.size() * 2);
        const uint32_t binary_root =
            BuildBinnedSahBinaryRecursive(refs, binary_nodes, 0, static_cast<uint32_t>(refs.size()),
                                          triangle_vertices != nullptr ? primitive_node_triangle_capacity : 1u);
        nodes.reserve(binary_nodes.size());
        if (used_binned_sah != nullptr)
        {
            *used_binned_sah = true;
        }
        return CollapseBinaryBvhRecursive(
            binary_nodes, binary_root, refs, triangle_vertices, nodes, fanout, primitive_node_triangle_capacity);
    }

    nodes.reserve(refs.size() * 2);
    return BuildWideBvhRecursive(
        refs, triangle_vertices, nodes, 0, static_cast<uint32_t>(refs.size()), fanout, primitive_node_triangle_capacity);
}

Aabb TransformAabb(const Aabb& local_bounds, const glm::mat4& object_to_world)
{
    Aabb world_bounds = EmptyAabb();
    for (uint32_t x = 0; x < 2; x++)
    {
        for (uint32_t y = 0; y < 2; y++)
        {
            for (uint32_t z = 0; z < 2; z++)
            {
                const glm::vec3 p(x == 0 ? local_bounds.min.x : local_bounds.max.x,
                                  y == 0 ? local_bounds.min.y : local_bounds.max.y,
                                  z == 0 ? local_bounds.min.z : local_bounds.max.z);
                ExpandAabb(&world_bounds, CpuPrimaryRayRenderer::TransformPosition(object_to_world, p));
            }
        }
    }
    return world_bounds;
}

struct BlasTranscribeContext
{
    const SceneData& scene;
    const std::vector<CpuPrimaryRayRenderer::BuildRef>& primitive_refs;
    const std::vector<glm::vec3>& verts;
    std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes;
    uint32_t fanout{2};
    uint32_t primitive_node_triangle_capacity{3};
};

bool TranscribeBlasNode(const BlasTranscribeContext& context,
                        uint32_t blas_index,
                        uint32_t source_node_index,
                        uint32_t* out_node_index)
{
    if (blas_index >= context.scene.blas_topologies.size())
    {
        return false;
    }
    const SceneBlasTopology& topology = context.scene.blas_topologies[blas_index];
    if (source_node_index >= topology.nodes.size())
    {
        return false;
    }
    const SceneBlasBvhNode& source_node = topology.nodes[source_node_index];
    if (source_node.children.size() > context.fanout || source_node.children.size() > 16)
    {
        return false;
    }

    const uint32_t node_index = static_cast<uint32_t>(context.nodes.size());
    context.nodes.push_back({});

    if (source_node.is_leaf)
    {
        const uint32_t first = source_node.primitive_start;
        const uint32_t count = source_node.primitive_count;
        if (first + count > context.primitive_refs.size())
        {
            return false;
        }
        context.nodes.pop_back();
        *out_node_index = CreateTriangleLeafRange(context.primitive_refs,
                                                  context.verts,
                                                  context.nodes,
                                                  first,
                                                  count,
                                                  context.fanout,
                                                  context.primitive_node_triangle_capacity);
        return true;
    }

    context.nodes[node_index].kind = CpuPrimaryRayRenderer::BvhNode::Kind::kInternal;
    context.nodes[node_index].bounds = EmptyAabb();
    for (uint32_t child_source_index : source_node.children)
    {
        uint32_t child_node_index = 0;
        if (!TranscribeBlasNode(context, blas_index, child_source_index, &child_node_index))
        {
            return false;
        }
        CpuPrimaryRayRenderer::BvhNode& node = context.nodes[node_index];
        node.children[node.child_count++] = MakeChildFromNode(context.nodes, child_node_index);
        ExpandAabb(&node.bounds, context.nodes[child_node_index].bounds);
    }

    *out_node_index = node_index;
    return context.nodes[node_index].child_count > 0;
}

struct TlasTranscribeContext
{
    const SceneData& scene;
    const std::vector<CpuPrimaryRayRenderer::BuildRef>& instance_refs;
    const std::vector<uint32_t>& instance_to_ref_index;
    std::vector<CpuPrimaryRayRenderer::BvhNode>& nodes;
    uint32_t fanout{2};
};

bool TranscribeTlasNode(const TlasTranscribeContext& context, uint32_t source_node_index, uint32_t* out_node_index)
{
    if (!context.scene.tlas_topology.valid || source_node_index >= context.scene.tlas_topology.nodes.size())
    {
        return false;
    }
    const SceneTlasNode& source_node = context.scene.tlas_topology.nodes[source_node_index];
    if (source_node.children.size() > context.fanout || source_node.children.size() > 16)
    {
        return false;
    }

    const uint32_t node_index = static_cast<uint32_t>(context.nodes.size());
    context.nodes.push_back({});
    context.nodes[node_index].kind = CpuPrimaryRayRenderer::BvhNode::Kind::kInternal;
    context.nodes[node_index].bounds = EmptyAabb();

    if (source_node.is_leaf)
    {
        const uint32_t instance_index = source_node.instance_index;
        if (instance_index >= context.instance_to_ref_index.size())
        {
            return false;
        }
        const uint32_t ref_index = context.instance_to_ref_index[instance_index];
        if (ref_index >= context.instance_refs.size())
        {
            return false;
        }
        CpuPrimaryRayRenderer::BvhNode::ChildNode child{};
        child.type = CpuPrimaryRayRenderer::BvhNode::SubnodeType::kInstance;
        child.aabb = context.instance_refs[ref_index].bounds;
        child.index = context.instance_refs[ref_index].payload;
        context.nodes[node_index].children[0] = child;
        context.nodes[node_index].child_count = 1;
        context.nodes[node_index].bounds = child.aabb;
        *out_node_index = node_index;
        return true;
    }

    for (uint32_t child_source_index : source_node.children)
    {
        if (child_source_index >= context.scene.tlas_topology.nodes.size())
        {
            return false;
        }
        const SceneTlasNode& child_source_node = context.scene.tlas_topology.nodes[child_source_index];
        CpuPrimaryRayRenderer::BvhNode& node = context.nodes[node_index];
        if (child_source_node.is_leaf)
        {
            const uint32_t instance_index = child_source_node.instance_index;
            if (instance_index >= context.instance_to_ref_index.size())
            {
                return false;
            }
            const uint32_t ref_index = context.instance_to_ref_index[instance_index];
            if (ref_index >= context.instance_refs.size())
            {
                return false;
            }

            CpuPrimaryRayRenderer::BvhNode::ChildNode child{};
            child.type = CpuPrimaryRayRenderer::BvhNode::SubnodeType::kInstance;
            child.aabb = context.instance_refs[ref_index].bounds;
            child.index = context.instance_refs[ref_index].payload;
            node.children[node.child_count++] = child;
            ExpandAabb(&node.bounds, child.aabb);
        }
        else
        {
            uint32_t child_node_index = 0;
            if (!TranscribeTlasNode(context, child_source_index, &child_node_index))
            {
                return false;
            }
            CpuPrimaryRayRenderer::BvhNode& refreshed_node = context.nodes[node_index];
            refreshed_node.children[refreshed_node.child_count++] = MakeChildFromNode(context.nodes, child_node_index);
            ExpandAabb(&refreshed_node.bounds, context.nodes[child_node_index].bounds);
        }
    }

    *out_node_index = node_index;
    return context.nodes[node_index].child_count > 0;
}
}  // namespace

void CpuPrimaryRayRenderer::BuildFromScene(const SceneData& scene, const CpuBvhSettings& settings, AppState* app_state)
{
    mesh_vertices_ = scene.blas_vertices;
    instances_.clear();
    blases_.clear();
    tlas_refs_.clear();
    tlas_nodes_.clear();
    tlas_root_index_ = 0;
    bvh_fanout_ = std::clamp(settings.fanout, 2u, 16u);
    primitive_node_triangle_capacity_ = std::clamp(settings.primitive_node_triangle_capacity, 1u, 3u);
    bvh_uses_rra_topology_ = false;
    bvh_uses_binned_sah_ = false;

    if (app_state != nullptr)
    {
        app_state->scene_stage.store(SceneLoadStage::kBuildingCpuBvh);
        app_state->SetStatus("Building CPU BVH");
    }

    const bool can_transcribe =
        settings.build_mode == CpuBvhBuildMode::kTranscribedRra &&
        (bvh_fanout_ == 4 || bvh_fanout_ == 8) &&
        scene.tlas_topology.valid &&
        !scene.blas_topologies.empty();

    blases_.resize(mesh_vertices_.size());
    uint64_t total_primitives = 0;
    uint64_t total_nodes = 0;
    bool all_rra_blas_transcribed = can_transcribe;

    for (uint32_t blas_index = 0; blas_index < mesh_vertices_.size(); blas_index++)
    {
        CpuBlas& blas = blases_[blas_index];
        const auto& verts = mesh_vertices_[blas_index];
        blas.primitive_refs.reserve(verts.size() / 3);

        for (uint32_t first_vertex = 0; first_vertex + 2 < verts.size(); first_vertex += 3)
        {
            BuildRef ref{};
            ref.payload = first_vertex;
            ref.bounds = EmptyAabb();
            ExpandAabb(&ref.bounds, verts[first_vertex + 0]);
            ExpandAabb(&ref.bounds, verts[first_vertex + 1]);
            ExpandAabb(&ref.bounds, verts[first_vertex + 2]);
            ref.centroid = (verts[first_vertex + 0] + verts[first_vertex + 1] + verts[first_vertex + 2]) / 3.0f;
            blas.primitive_refs.push_back(ref);
        }

        if (!blas.primitive_refs.empty())
        {
            if (can_transcribe && blas_index < scene.blas_topologies.size())
            {
                BlasTranscribeContext context{scene, blas.primitive_refs, verts, blas.nodes, bvh_fanout_, primitive_node_triangle_capacity_};
                if (!TranscribeBlasNode(context, blas_index, scene.blas_topologies[blas_index].root_index, &blas.root_index))
                {
                    blas.nodes.clear();
                    all_rra_blas_transcribed = false;
                }
            }

            if (blas.nodes.empty())
            {
                blas.root_index = BuildConfiguredBvh(
                    blas.primitive_refs,
                    &verts,
                    blas.nodes,
                    bvh_fanout_,
                    primitive_node_triangle_capacity_,
                    settings.split_mode,
                    &bvh_uses_binned_sah_);
            }
        }

        total_primitives += blas.primitive_refs.size();
        total_nodes += blas.nodes.size();
    }

    for (const SceneInstance& instance : scene.instances)
    {
        InstanceData instance_data{};
        instance_data.mesh_index       = static_cast<uint32_t>(instance.blas_idx);
        instance_data.object_to_world  = Transform3x4ToMat4(instance.transform);
        instance_data.world_to_object  = glm::inverse(instance_data.object_to_world);
        if (instance_data.mesh_index < blases_.size() && !blases_[instance_data.mesh_index].nodes.empty())
        {
            const CpuBlas& blas = blases_[instance_data.mesh_index];
            instance_data.world_bounds = TransformAabb(blas.nodes[blas.root_index].bounds, instance_data.object_to_world);
        }
        instances_.push_back(instance_data);
    }

    tlas_refs_.reserve(instances_.size());
    std::vector<uint32_t> instance_to_tlas_ref(instances_.size(), std::numeric_limits<uint32_t>::max());
    for (uint32_t instance_index = 0; instance_index < instances_.size(); instance_index++)
    {
        const InstanceData& instance = instances_[instance_index];
        if (instance.mesh_index >= blases_.size() || blases_[instance.mesh_index].nodes.empty())
        {
            continue;
        }
        BuildRef ref{};
        ref.payload = instance_index;
        ref.bounds = instance.world_bounds;
        ref.centroid = (ref.bounds.min + ref.bounds.max) * 0.5f;
        instance_to_tlas_ref[instance_index] = static_cast<uint32_t>(tlas_refs_.size());
        tlas_refs_.push_back(ref);
    }

    if (!tlas_refs_.empty())
    {
        if (can_transcribe && all_rra_blas_transcribed)
        {
            TlasTranscribeContext context{scene, tlas_refs_, instance_to_tlas_ref, tlas_nodes_, bvh_fanout_};
            if (TranscribeTlasNode(context, scene.tlas_topology.root_index, &tlas_root_index_))
            {
                bvh_uses_rra_topology_ = true;
            }
            else
            {
                tlas_nodes_.clear();
            }
        }

        if (tlas_nodes_.empty())
        {
            tlas_root_index_ = BuildConfiguredBvh(
                tlas_refs_, nullptr, tlas_nodes_, bvh_fanout_, primitive_node_triangle_capacity_, settings.split_mode, &bvh_uses_binned_sah_);
        }
        total_nodes += tlas_nodes_.size();
    }

    if (app_state != nullptr)
    {
        std::lock_guard<std::mutex> lock(app_state->details_mutex);
        app_state->cpu_bvh_node_count      = total_nodes;
        app_state->cpu_bvh_primitive_count = total_primitives;
        app_state->cpu_bvh_fanout          = bvh_fanout_;
        app_state->cpu_bvh_primitive_node_triangle_capacity = primitive_node_triangle_capacity_;
        app_state->cpu_bvh_uses_rra_topology = bvh_uses_rra_topology_;
        app_state->cpu_bvh_uses_binned_sah = bvh_uses_binned_sah_;
    }
}

namespace
{
bool TraceBruteForce(const std::vector<std::vector<glm::vec3>>& mesh_vertices,
                     const std::vector<CpuPrimaryRayRenderer::InstanceData>& instances,
                     const glm::vec3& origin,
                     const glm::vec3& direction,
                     float tmin,
                     float tmax,
                     CpuRenderStats* stats,
                     HitInfo* out_hit)
{
    HitInfo hit;
    for (uint32_t instance_index = 0; instance_index < instances.size(); instance_index++)
    {
        const auto& instance = instances[instance_index];
        const auto& verts    = mesh_vertices[instance.mesh_index];
        const glm::vec3 local_origin = CpuPrimaryRayRenderer::TransformPosition(instance.world_to_object, origin);
        const glm::vec3 local_dir    = CpuPrimaryRayRenderer::TransformDirection(instance.world_to_object, direction);

        for (uint32_t first_vertex = 0; first_vertex + 2 < verts.size(); first_vertex += 3)
        {
            float t = 0.0f;
            if (!IntersectTriangle(local_origin,
                                   local_dir,
                                   tmin,
                                   std::min(tmax, hit.t),
                                   verts[first_vertex + 0],
                                   verts[first_vertex + 1],
                                   verts[first_vertex + 2],
                                   stats,
                                   &t))
            {
                continue;
            }

            if (t < hit.t)
            {
                hit.t            = t;
                hit.instance_index = instance_index;
                hit.first_vertex = first_vertex;
                hit.hit          = true;
            }
        }
    }

    *out_hit = hit;
    return hit.hit;
}

bool TraceBlasBvh(const std::vector<glm::vec3>& verts,
                  const CpuPrimaryRayRenderer::CpuBlas& blas,
                  uint32_t instance_index,
                  const glm::vec3& local_origin,
                  const glm::vec3& local_direction,
                  float tmin,
                  float tmax,
                  CpuRenderStats* stats,
                  HitInfo* hit)
{
    if (blas.nodes.empty() || blas.root_index >= blas.nodes.size())
    {
        return false;
    }

    const glm::vec3 inv_dir(1.0f / local_direction.x, 1.0f / local_direction.y, 1.0f / local_direction.z);
    //stats->ray_box_tests++; //Self
    if (!IntersectAabb(blas.nodes[blas.root_index].bounds, local_origin, inv_dir, tmin, tmax))
    {
        return false;
    }

    std::vector<uint32_t> stack;
    stack.reserve(128);
    stack.push_back(blas.root_index);

    bool any_hit = false;
    while (!stack.empty())
    {
        const auto node_index = stack.back();
        stack.pop_back();
        const auto& node = blas.nodes[node_index];
        stats->bvh_steps++;

        if (node.kind == CpuPrimaryRayRenderer::BvhNode::Kind::kTriangle)
        {
            stats->tri_nodes++;
            for (uint32_t i = 0; i < node.triangle_count; i++)
            {
                const auto& triangle = node.triangles[i];
                float t = 0.0f;
                if (IntersectTriangle(local_origin, local_direction, tmin, std::min(tmax, hit->t), triangle.v0, triangle.v1, triangle.v2, stats, &t) &&
                    t < hit->t)
                {
                    hit->t              = t;
                    hit->instance_index = instance_index;
                    hit->first_vertex   = triangle.first_vertex;
                    hit->hit            = true;
                    any_hit             = true;
                }
            }
        }
        else
        {
            stats->box_nodes++;
            for (uint32_t child_index = 0; child_index < node.child_count; child_index++)
            {
                const auto& child = node.children[child_index];
                if (child.type == CpuPrimaryRayRenderer::BvhNode::SubnodeType::kEmpty ||
                    child.type == CpuPrimaryRayRenderer::BvhNode::SubnodeType::kInstance)
                {
                    continue;
                }
                const uint32_t child_node_index = child.index;
                if (child_node_index >= blas.nodes.size())
                {
                    continue;
                }
                stats->ray_box_tests++;
                if (!IntersectAabb(child.aabb, local_origin, inv_dir, tmin, std::min(tmax, hit->t)))
                {
                    continue;
                }
                stack.push_back(child_node_index);
            }
        }
    }

    return any_hit;
}

bool TraceTlasBlasBvh(const std::vector<std::vector<glm::vec3>>& mesh_vertices,
                      const std::vector<CpuPrimaryRayRenderer::InstanceData>& instances,
                      const std::vector<CpuPrimaryRayRenderer::CpuBlas>& blases,
                      const std::vector<CpuPrimaryRayRenderer::BuildRef>& tlas_refs,
                      const std::vector<CpuPrimaryRayRenderer::BvhNode>& tlas_nodes,
                      uint32_t tlas_root_index,
                      const glm::vec3& origin,
                      const glm::vec3& direction,
                      float tmin,
                      float tmax,
                      CpuRenderStats* stats,
                      HitInfo* out_hit)
{
    if (tlas_nodes.empty() || tlas_root_index >= tlas_nodes.size())
    {
        return false;
    }

    const glm::vec3 inv_dir(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    //stats->ray_box_tests++; // Self
    if (!IntersectAabb(tlas_nodes[tlas_root_index].bounds, origin, inv_dir, tmin, tmax))
    {
        return false;
    }

    std::vector<uint32_t> stack;
    stack.reserve(128);
    stack.push_back(tlas_root_index);

    HitInfo hit;
    while (!stack.empty())
    {
        const auto node_index = stack.back();
        stack.pop_back();
        const auto& node = tlas_nodes[node_index];
        stats->bvh_steps++;

        if (node.kind == CpuPrimaryRayRenderer::BvhNode::Kind::kInternal)
        {
            stats->box_nodes++;
            for (uint32_t child_index = 0; child_index < node.child_count; child_index++)
            {
                const auto& child = node.children[child_index];
                if (child.type == CpuPrimaryRayRenderer::BvhNode::SubnodeType::kEmpty ||
                    child.type == CpuPrimaryRayRenderer::BvhNode::SubnodeType::kTriangle)
                {
                    continue;
                }
                stats->ray_box_tests++;
                if (!IntersectAabb(child.aabb, origin, inv_dir, tmin, std::min(tmax, hit.t)))
                {
                    continue;
                }

                if (child.type == CpuPrimaryRayRenderer::BvhNode::SubnodeType::kInstance)
                {
                    const uint32_t instance_index = child.index;
                    if (instance_index >= instances.size())
                    {
                        continue;
                    }
                    const auto& instance = instances[instance_index];
                    if (instance.mesh_index >= blases.size())
                    {
                        continue;
                    }

                    const CpuPrimaryRayRenderer::CpuBlas& blas = blases[instance.mesh_index];
                    const glm::vec3 local_origin = CpuPrimaryRayRenderer::TransformPosition(instance.world_to_object, origin);
                    const glm::vec3 local_dir    = CpuPrimaryRayRenderer::TransformDirection(instance.world_to_object, direction);
                    stats->tlas_to_blas++;
                    TraceBlasBvh(mesh_vertices[instance.mesh_index],
                                 blas,
                                 instance_index,
                                 local_origin,
                                 local_dir,
                                 tmin,
                                 std::min(tmax, hit.t),
                                 stats,
                                 &hit);
                }
                else if (child.type == CpuPrimaryRayRenderer::BvhNode::SubnodeType::kBox)
                {
                    if (child.index < tlas_nodes.size())
                    {
                        stack.push_back(child.index);
                    }
                }
            }
        }
    }

    *out_hit = hit;
    return hit.hit;
}
}  // namespace

void CpuPrimaryRayRenderer::Render(const CpuRenderRequest& request,
                                   CpuRenderResult* out_result,
                                   const std::function<void(const CpuRenderResult&)>& publish_partial)
{
    constexpr uint32_t kTileSize = 16;

    out_result->width      = request.width;
    out_result->height     = request.height;
    out_result->tiles_completed = 0;
    out_result->tiles_total = ((request.width + kTileSize - 1) / kTileSize) * ((request.height + kTileSize - 1) / kTileSize);
    out_result->generation = request.generation;
    out_result->complete   = false;
    out_result->backend    = request.backend;
    out_result->stats      = {};
    out_result->rgba.resize(static_cast<size_t>(request.width) * static_cast<size_t>(request.height) * 4);

    for (uint32_t y = 0; y < request.height; y++)
    {
        for (uint32_t x = 0; x < request.width; x++)
        {
            WriteColor(SkipColor(x, y), out_result->rgba.data() + (x + y * request.width) * 4);
        }
    }
    if (publish_partial)
    {
        publish_partial(*out_result);
    }

    struct AtomicStats
    {
        std::atomic<uint64_t> rays{0};
        std::atomic<uint64_t> bvh_steps{0};
        std::atomic<uint64_t> box_nodes{0};
        std::atomic<uint64_t> ray_box_tests{0};
        std::atomic<uint64_t> tri_nodes{0};
        std::atomic<uint64_t> ray_triangle_tests{0};
        std::atomic<uint64_t> tlas_to_blas{0};
    } atomic_stats;

    std::atomic<uint32_t> next_tile{0};
    std::atomic<uint32_t> completed_tiles{0};
    std::mutex publish_mutex;

    const uint32_t tiles_x = (request.width + kTileSize - 1) / kTileSize;
    const uint32_t tiles_y = (request.height + kTileSize - 1) / kTileSize;
    const uint32_t worker_count = std::max(1u, request.thread_count);

    auto worker = [&]() {
        while (true)
        {
            const uint32_t tile_index = next_tile.fetch_add(1);
            if (tile_index >= out_result->tiles_total)
            {
                return;
            }

            CpuRenderStats tile_stats{};
            const uint32_t tile_x = tile_index % tiles_x;
            const uint32_t tile_y = tile_index / tiles_x;
            const uint32_t x0 = tile_x * kTileSize;
            const uint32_t y0 = tile_y * kTileSize;
            const uint32_t x1 = std::min(x0 + kTileSize, request.width);
            const uint32_t y1 = std::min(y0 + kTileSize, request.height);

            for (uint32_t y = y0; y < y1; y++)
            {
                for (uint32_t x = x0; x < x1; x++)
                {
                    bool should_skip = false;
                    const uint32_t pixel_index = x + y * request.width;
                    uint32_t ray_begin = pixel_index;
                    uint32_t ray_end = pixel_index + 1;

                    if (request.use_external_rays)
                    {
                        if (pixel_index < request.external_ray_offsets.size())
                        {
                            ray_begin = pixel_index == 0 ? 0 : request.external_ray_offsets[pixel_index - 1];
                            ray_end   = request.external_ray_offsets[pixel_index];
                            ray_end   = std::min<uint32_t>(ray_end, static_cast<uint32_t>(request.external_rays.size()));
                            ray_begin = std::min(ray_begin, ray_end);
                            should_skip = ray_begin == ray_end;
                        }
                        else
                        {
                            should_skip = true;
                        }
                    }

                    CpuRay camera_ray{};
                    if (!request.use_external_rays)
                    {
                        const glm::vec2 d = (((glm::vec2(x, y) + 0.5f) / glm::vec2(request.width, request.height)) * 2.0f - 1.0f);
                        float dy          = -d.y;
                        if (request.invert_y)
                        {
                            dy *= -1.0f;
                        }
                        const glm::vec3 target = TransformPosition(request.inverse_proj, glm::vec3(d.x, dy, 1.0f));
                        camera_ray.origin      = TransformPosition(request.inverse_view, glm::vec3(0.0f, 0.0f, 0.0f));
                        camera_ray.direction   = TransformDirection(request.inverse_view, glm::normalize(target));
                        camera_ray.tmin        = 0.001f;
                        camera_ray.tmax        = 10000.0f;
                    }

                    glm::vec4 color = should_skip ? SkipColor(x, y) : ShadeMiss(y, request.height);
                    if (!should_skip)
                    {
                        glm::vec4 accumulated_color(0.0f);
                        const uint32_t rays_to_trace = request.use_external_rays ? (ray_end - ray_begin) : 1;
                        for (uint32_t ray_index = 0; ray_index < rays_to_trace; ray_index++)
                        {
                            const CpuRay& ray = request.use_external_rays ? request.external_rays[ray_begin + ray_index] : camera_ray;
                            tile_stats.rays++;
                            HitInfo hit;
                            const bool did_hit =
                                request.backend == RenderBackend::kCpuBvh
                                    ? TraceTlasBlasBvh(mesh_vertices_, instances_, blases_, tlas_refs_, tlas_nodes_, tlas_root_index_, ray.origin, ray.direction, ray.tmin, ray.tmax, &tile_stats, &hit)
                                    : TraceBruteForce(mesh_vertices_, instances_, ray.origin, ray.direction, ray.tmin, ray.tmax, &tile_stats, &hit);

                            glm::vec4 ray_color = ShadeMiss(y, request.height);
                            if (did_hit)
                            {
                                const auto& instance = instances_[hit.instance_index];
                                const auto& verts    = mesh_vertices_[instance.mesh_index];
                                ray_color = PackColor(MakeColorFromWorldNormal(verts[hit.first_vertex + 0],
                                                                               verts[hit.first_vertex + 1],
                                                                               verts[hit.first_vertex + 2],
                                                                               instance.object_to_world));
                            }
                            accumulated_color += ray_color;
                        }
                        color = accumulated_color / static_cast<float>(rays_to_trace);
                    }

                    WriteColor(color, out_result->rgba.data() + pixel_index * 4);
                }
            }

            atomic_stats.rays.fetch_add(tile_stats.rays);
            atomic_stats.bvh_steps.fetch_add(tile_stats.bvh_steps);
            atomic_stats.box_nodes.fetch_add(tile_stats.box_nodes);
            atomic_stats.tri_nodes.fetch_add(tile_stats.tri_nodes);
            atomic_stats.ray_box_tests.fetch_add(tile_stats.ray_box_tests);
            atomic_stats.ray_triangle_tests.fetch_add(tile_stats.ray_triangle_tests);
            atomic_stats.tlas_to_blas.fetch_add(tile_stats.tlas_to_blas);

            const uint32_t tiles_done = completed_tiles.fetch_add(1) + 1;
            if (publish_partial)
            {
                std::lock_guard<std::mutex> lock(publish_mutex);
                out_result->tiles_completed         = tiles_done;
                out_result->stats.rays              = atomic_stats.rays.load();
                out_result->stats.bvh_steps         = atomic_stats.bvh_steps.load();
                out_result->stats.box_nodes         = atomic_stats.box_nodes.load();
                out_result->stats.tri_nodes          = atomic_stats.tri_nodes.load();
                out_result->stats.ray_box_tests     = atomic_stats.ray_box_tests.load();
                out_result->stats.ray_triangle_tests = atomic_stats.ray_triangle_tests.load();
                out_result->stats.tlas_to_blas       = atomic_stats.tlas_to_blas.load();
                publish_partial(*out_result);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (uint32_t i = 0; i < worker_count; i++)
    {
        threads.emplace_back(worker);
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    out_result->tiles_completed = out_result->tiles_total;
    out_result->stats.rays = atomic_stats.rays.load();
    out_result->stats.bvh_steps = atomic_stats.bvh_steps.load();
    out_result->stats.box_nodes = atomic_stats.box_nodes.load();
    out_result->stats.tri_nodes = atomic_stats.tri_nodes.load();
    out_result->stats.ray_box_tests = atomic_stats.ray_box_tests.load();
    out_result->stats.ray_triangle_tests = atomic_stats.ray_triangle_tests.load();
    out_result->stats.tlas_to_blas       = atomic_stats.tlas_to_blas.load();
    out_result->complete = true;
    if (publish_partial)
    {
        publish_partial(*out_result);
    }
}

bool CpuPrimaryRayRenderer::IsReady() const
{
    return !instances_.empty();
}
