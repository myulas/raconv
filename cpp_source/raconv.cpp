#include <torch/torch.h>

using namespace at::indexing;

at::Tensor convolution(at::Tensor patches, at::Tensor filters, at::Tensor bias)
{
    auto batch_size = patches.size(0),
         num_patches = patches.size(1),
         num_filters = filters.size(0);

    auto output = at::empty({batch_size, num_patches, num_filters}, at::kFloat);

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end)
    {
        c10::InferenceMode inference_mode;

        for(auto bi = start; bi < end; bi++)
        {
            auto output_bi = output[bi];
            auto patches_bi = patches[bi];

            for(auto pi = 0; pi < num_patches; pi++)
            {
                output_bi[pi] = bias.addmv(filters, patches_bi[pi]);
            }
        }
    });

    return output;
}

at::Tensor raconvolution_avg(at::Tensor patches, at::Tensor filters, at::Tensor bias,
                             float scale_factor, int memory_size, int min_summary)
{
    auto batch_size = patches.size(0),
         num_patches = patches.size(1),
         num_filters = filters.size(0);

    auto output = at::empty({batch_size, num_patches, num_filters}, at::kFloat);

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end)
    {
        c10::InferenceMode inference_mode;
        
        auto summaries = patches.index({Slice(start, end)}).mean(2).mul_(scale_factor).to(at::kInt);
        
        int summary, mi, *memory_ptr = new int[memory_size];
        memory_ptr -= min_summary;

        for(auto bi = start; bi < end; bi++)
        {
            auto output_bi = output[bi];
            auto patches_bi = patches[bi];
            auto summaries_bi = summaries[bi - start].data_ptr<int>();
            std::memset(memory_ptr + min_summary, -1, memory_size * sizeof(int));

            for(auto pi = 0; pi < num_patches; pi++)
            {
                summary = summaries_bi[pi];
                mi = memory_ptr[summary];

                if(mi == -1)
                {
                    output_bi[pi] = bias.addmv(filters, patches_bi[pi]);
                    memory_ptr[summary] = pi;
                }
                else
                {
                    output_bi[pi] = output_bi[mi];
                }
            }
        }

        delete[] (memory_ptr + min_summary); 
    });

    return output;
}

at::Tensor raconvolution_max(at::Tensor patches, at::Tensor filters, at::Tensor bias,
                             float scale_factor, int memory_size, int min_summary)
{
    auto batch_size = patches.size(0),
         num_patches = patches.size(1),
         num_filters = filters.size(0);

    auto output = at::empty({batch_size, num_patches, num_filters}, at::kFloat);

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end)
    {
        c10::InferenceMode inference_mode;

        auto summaries = std::get<0>(patches.index({Slice(start, end)}).max(2)).mul_(scale_factor).to(at::kInt);

        int summary, mi, *memory_ptr = new int[memory_size];
        memory_ptr -= min_summary;

        for(auto bi = start; bi < end; bi++)
        {
            auto output_bi = output[bi];
            auto patches_bi = patches[bi];
            auto summaries_bi = summaries[bi - start].data_ptr<int>();
            std::memset(memory_ptr + min_summary, -1, memory_size * sizeof(int));

            for(auto pi = 0; pi < num_patches; pi++)
            {
                summary = summaries_bi[pi];
                mi = memory_ptr[summary];

                if(mi == -1)
                {
                    output_bi[pi] = bias.addmv(filters, patches_bi[pi]);
                    memory_ptr[summary] = pi;
                }
                else
                {
                    output_bi[pi] = output_bi[mi];
                }
            }
        }

        delete[] (memory_ptr + min_summary);
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("convolution", &convolution, "Vanilla Convolution");
    m.def("raconvolution_avg", &raconvolution_avg, "Redundancy-Aware Convolution - AVG");
    m.def("raconvolution_max", &raconvolution_max, "Redundancy-Aware Convolution - MAX");
}