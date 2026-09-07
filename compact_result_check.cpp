#include <stdint.h>
#include <stdio.h>
#include <filesystem>
#include <fstream>

#include <glm/glm.hpp>
#include "lodepng.h"

uint32_t RT_W, RT_H;

void DumpToPNG(uint32_t w, uint32_t h, std::vector<glm::vec4>* pixels, const char* png_out_fn)
{
    {
        uint32_t             sz = pixels->size();
        assert(sz == w * h);
        std::vector<uint8_t> rgba(sz * 4);
        for (uint32_t i = 0; i < sz; i++)
        {
            glm::vec4 p = (*pixels)[i];
            rgba[i * 4]     = p.x * 255;
            rgba[i * 4 + 1] = p.y * 255;
            rgba[i * 4 + 2] = p.z * 255;
            rgba[i * 4 + 3] = p.w * 255;
        }
        
        /*Encode the image*/
        unsigned error = lodepng_encode32_file(png_out_fn, rgba.data(), w, h);

        /*if there's an error, display it*/
        if (error)
            printf("error %u: %s\n", error, lodepng_error_text(error));
    }
}

int main(int argc, char** argv)
{
    printf("RRAPlayground \"compacted\" result checker.\n");
    printf("  add `--perbatch` to dump per-batch result\n");
    printf("  add `-file-to-use FILE_TOUSE` to choose which mapping file to use\n");
    printf("                                0: ReducePixelRanges.txt\n");
    printf("                                1: ScatterMap.txt\n");
    bool should_print_per_batch{false};
    int  FILE_TO_USE = 1;
    for (uint32_t i = 0; i < argc; i++)
    {
        if (!strcmp(argv[i], "--perbatch"))
        {
            should_print_per_batch = true;
            printf("Should print per patch.\n");
        } 
        else if ((!strcmp(argv[i], "-file-to-use")) && i+1 < argc)
        {
            FILE_TO_USE = std::atoi(argv[i + 1]);
            printf("FILE_TO_USE set to %d\n", FILE_TO_USE);
        }
    }

    const char* crr_fn = "CompactRayResults.bin.txt";
    if (std::filesystem::exists(crr_fn) == false)
    {
        printf("Oh! Please run this from where `%s` is located.\n", crr_fn);
        exit(0);
    }

    std::vector<glm::vec4> out_sums;
    std::vector<uint32_t>  out_counts;

    std::ifstream ifs(crr_fn);
    while (ifs.good())
    {
        std::string line;
        std::getline(ifs, line);

        if (sscanf_s(line.c_str(), "rendertarget size: %u x %u", &RT_W, &RT_H) == 2)
        {
            printf("rendertarget: %u x %u\n", RT_W, RT_H);
        }
    }
    ifs.close();

    out_sums.resize(RT_W * RT_H);
    out_counts.resize(RT_W * RT_H);

    const char* crr_bin_fn = "CompactRayResults.bin";
    if (std::filesystem::exists(crr_bin_fn) == false)
    {
        printf("Oh! %s not found.\n", crr_bin_fn);
        exit(0);
    }

    std::vector<glm::vec4> batch_results;

    ifs                  = std::ifstream(crr_bin_fn, std::ios::binary | std::ios::ate);
    std::streamsize size = ifs.tellg();
    batch_results.resize(size / 4);
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(batch_results.data()), size);
    ifs.close();
    printf("Results size = %zu\n", size);

    std::vector<std::string> batches;
    std::ranges::for_each(std::filesystem::directory_iterator{"."}, [&](const std::filesystem::directory_entry& dir_entry) {
        if (dir_entry.is_directory())
        {
            std::string d = dir_entry.path().string();
            uint32_t dummy;
            if (sscanf_s(d.c_str(), ".\\batch%06u", &dummy) == 1)
            {
                batches.push_back(d);
            }
        }
    });
    printf("%zu batches found.\n", batches.size());

    uint32_t bidx{0};
    for (std::string b : batches)
    {
        std::vector<glm::vec4> this_batch_output;
        if (should_print_per_batch)
        {
            this_batch_output.resize(RT_W * RT_H);
        }

        uint32_t    nlines{0};

        switch (FILE_TO_USE)
        {
            case 0:
            {
                std::string b_fn = b + "\\ReducePixelRanges.txt";
                if (std::filesystem::exists(b_fn))
                {
                    ifs = std::ifstream(b_fn, std::ios::in);
                    std::string line;
                    std::getline(ifs, line);
                    while (ifs.good())
                    {
                        std::getline(ifs, line);
                        uint32_t pixel, global_begin, global_end;
                        if (sscanf(line.c_str(), "%u %u %u", &pixel, &global_begin, &global_end) == 3)
                        {
                            for (uint32_t idx = global_begin; idx < global_end; idx++)
                            {
                                if (should_print_per_batch)
                                {
                                    this_batch_output[pixel] = batch_results[idx];
                                }
                                out_sums[pixel] += batch_results[idx];
                                out_counts[pixel]++;
                            }
                            nlines++;
                        }
                    }
                    ifs.close();
                }
                break;
            }
            case 1:
            {
                std::string b_fn = b + "\\ScatterMap.txt";
                if (std::filesystem::exists(b_fn))
                {
                    ifs = std::ifstream(b_fn, std::ios::in);
                    std::string line;
                    std::getline(ifs, line);
                    while (ifs.good())
                    {
                        std::getline(ifs, line);
                        uint32_t local_ray_index, compact_ray_index, pixel_index;
                        if (sscanf(line.c_str(), "%u %u %u", &local_ray_index, &compact_ray_index, &pixel_index) == 3)
                        {
                            assert(local_ray_index == nlines);
                            if (should_print_per_batch)
                            {
                                this_batch_output[pixel_index] = batch_results[compact_ray_index];
                            }
                            out_sums[pixel_index] += batch_results[compact_ray_index];
                            out_counts[pixel_index]++;
                            nlines++;
                        }
                    }
                    ifs.close();
                }
                break;
            }
        }

        if (should_print_per_batch)
        {
            char buf[200];
            snprintf(buf, sizeof(buf), "batch%06u_out.png", bidx);
            DumpToPNG(RT_W, RT_H, &this_batch_output, buf);
        }

        printf("[%u] %s, %u pixels written\n", bidx, b.c_str(), nlines);
        bidx++;
    }

    // Save to BPM format
    const char*   ppm_out_fn = "out.png";
    for (uint32_t i = 0; i < RT_W * RT_H; i++)
    {
        out_sums[i] *= (1.0f / out_counts[i]);
    }
    DumpToPNG(RT_W, RT_H, &out_sums, ppm_out_fn);

    return 0;
}