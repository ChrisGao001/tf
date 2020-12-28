#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SparseTransformer")
    .Attr("T: {int32, int64}")
    .Input("row_width: T")
    .Output("indices: int64")
    .Output("shape: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(2));
      return Status::OK();
    });

template <typename T>
class SparseTransformer : public OpKernel {
 public:
  explicit SparseTransformer(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    int64 batch_size = input.size();
    int64 total = 0;
    T max =  0;
    for (int i = 0; i < batch_size; ++i) {
		total += input(i);	
        max = max > input(i) ? max : input(i);
    }

    // Create an output tensor
    Tensor* output_tensor_indice = NULL;
    Tensor* output_tensor_shape = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, {total, 2},
                                                     &output_tensor_indice));
    OP_REQUIRES_OK(context, context->allocate_output(1, {2},
                                                     &output_tensor_shape));
    // fill shape tensor
    auto output_shape_flat = output_tensor_shape->flat<int64>();
    output_shape_flat(0) = batch_size;
    output_shape_flat(1) = max;
    // fill indice tensor
    size_t offset = 0;
    for (int i = 0; i < batch_size;  ++i) {
		int64* ix_p = &output_tensor_indice->matrix<int64>()(offset, 0);
        auto count = input(i);
		for (int j = 0;  j < count;  ++j) {
			*ix_p = i;
			*(ix_p + 1) = j;
			ix_p += 2;
		}
		offset += count;
    }
  }
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SparseTransformer").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseTransformer<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);

#undef REGISTER_KERNEL
