// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  // Backward-compatible API:
  // - quantized=false (default): key_value is a Tensor, behaves like the original op.
  // - quantized=true: key_value is a sequence of 6 tensors
  //   (k_enc, k_scale, k_mn, v_enc, v_scale, v_mn), and the op will
  //   dequantize on GPU and write directly into vLLM paged KV cache.
  m.def(
    "multi_layer_kv_transfer",
    [](py::object key_value,
       const torch::Tensor& key_value_ptrs,
       const torch::Tensor& slot_mapping,
       const torch::Device& paged_memory_device,
       const int page_buffer_size,
       const bool direction,
       const bool use_mla,
       const bool quantized,
       const int bits,
       const int group_size) {
      if (py::isinstance<py::sequence>(key_value)) {
        auto seq = key_value.cast<py::sequence>();
        TORCH_CHECK(quantized, "TensorList requires quantized=true for dequantize path");
        TORCH_CHECK(seq.size() == 6,
                    "quantized=true expects a sequence of 6 tensors: "
                    "(k_encoded, k_scale, k_mn, v_encoded, v_scale, v_mn)");

        std::vector<torch::Tensor> kv_list;
        kv_list.reserve(6);
        for (size_t i = 0; i < 6; ++i) {
          kv_list.push_back(seq[i].cast<torch::Tensor>());
        }

        return multi_layer_kv_transfer(kv_list, key_value_ptrs, slot_mapping,
                                       paged_memory_device, page_buffer_size,
                                       direction, use_mla, bits, group_size);
      }

      TORCH_CHECK(!quantized, "quantized=true expects a TensorList, got Tensor");
      auto kv = key_value.cast<torch::Tensor>();
      return multi_layer_kv_transfer(kv, key_value_ptrs, slot_mapping,
                                     paged_memory_device, page_buffer_size,
                                     direction, use_mla);
    },
    py::arg("key_value"),
    py::arg("key_value_ptrs"),
    py::arg("slot_mapping"),
    py::arg("paged_memory_device"),
    py::arg("page_buffer_size"),
    py::arg("direction"),
    py::arg("use_mla"),
    py::arg("quantized") = false,
    py::arg("bits") = 4,
    py::arg("group_size") = 128);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async);
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("alloc_numa_ptr", &alloc_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_numa_ptr", &free_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
}
