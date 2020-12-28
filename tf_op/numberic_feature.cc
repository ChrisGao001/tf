#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

inline float signed_log_x_plus_one(float x) {
  if (x > 0) {
    return std::log(x + 1.0);
  } else if (x < 0) {
    return -1 * std::log(-1 * x + 1.0);
  } else {
    return 0;
  }
}

inline float min_max_normalize(float x, float v_min, float v_max, float delta) {
  if (x < v_min) {
    return 0;
  } else if (x < v_max) {
    return (x - v_min) / delta ;
  } else {
    return 1.0;
  }
}

class LogMinMaxOp : public OpKernel {
public:
    explicit LogMinMaxOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("LogMin", &log_min_));
        OP_REQUIRES_OK(context, context->GetAttr("LogMax", &log_max_));
        delta_ = log_max_ - log_min_;
    }
    void Compute(OpKernelContext* context) override {
        Tensor& x_tensor = const_cast<Tensor&>(context->input(0));
        auto x = x_tensor.flat<float>();
        int64 batch_size = x.size();
        float* p1 = (float*)(&x(0));
        for (int i = 0; i < batch_size; ++i) {
            x(i) = min_max_normalize(signed_log_x_plus_one(x(i)), log_min_, log_max_, delta_);
        }
        context->set_output(0, x_tensor);
    }
private:
    float log_min_ = 0, log_max_ = 0;
    float delta_ = 0;
};

REGISTER_OP("LogMinMax")
    .Input("x: float32")
    .Output("y: float32")
    .Attr("LogMin: float")
    .Attr("LogMax: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("LogMinMax").Device(DEVICE_CPU), LogMinMaxOp)
