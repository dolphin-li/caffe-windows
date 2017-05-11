#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// Configure the kernel size, padding, stride, and inputs.
	PoolingParameter pool_param = this->layer_param_.pooling_param();
	channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
	global_pooling_ = pool_param.global_pooling();
	const int first_spatial_axis = channel_axis_ + 1;
	const int num_axes = bottom[0]->num_axes();
	const int num_kernel_dims = pool_param.kernel_size_size();
	num_spatial_axes_ = num_axes - first_spatial_axis;
	CHECK_GE(num_spatial_axes_, 0);
	const vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
	// Checking for kernel configs
	if (pool_param.global_pooling()) {
		CHECK(!(pool_param.kernel_size_size() ||
			pool_param.has_kernel_h() || pool_param.has_kernel_w()))
			<< "With Global_pooling: true Filter size cannot specified";
	}
	else {
		CHECK(!pool_param.kernel_size_size() !=
			!(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
			<< "Filter size is kernel_size OR kernel_h and kernel_w; not both";
		CHECK(pool_param.kernel_size_size() ||
			(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
			<< "For non-square filters both kernel_h and kernel_w are required.";
	}
	CHECK((!pool_param.pad_size() && pool_param.has_pad_h()
		&& pool_param.has_pad_w())
		|| (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
		<< "pad is pad OR pad_h and pad_w are required.";
	CHECK((!pool_param.stride_size() && pool_param.has_stride_h()
		&& pool_param.has_stride_w())
		|| (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
		<< "Stride is stride OR stride_h and stride_w are required.";

	// Setup filter kernel dimensions(kernel_shape_).
	kernel_shape_.Reshape(spatial_dim_blob_shape);
	if (global_pooling_) for(int i=0; i < num_spatial_axes_; i++)
		kernel_shape_.mutable_cpu_data()[i] = bottom[0]->shape(i + first_spatial_axis);
	else if(!pool_param.has_kernel_h()) for (int i = 0; i < num_spatial_axes_; i++)
		kernel_shape_.mutable_cpu_data()[i] = pool_param.kernel_size((pool_param.kernel_size_size() == 1) ? 0 : i);
	else {
		kernel_shape_.mutable_cpu_data()[0] = pool_param.kernel_h();
		kernel_shape_.mutable_cpu_data()[1] = pool_param.kernel_w();
	}
	for (int i = 0; i < num_spatial_axes_; i++)
		CHECK_GT(kernel_shape_.cpu_data()[i], 0) << "Filter dimensions cannot be zero.";

	// Setup filter pad dimensions(pad_shape_).
	pad_shape_.Reshape(spatial_dim_blob_shape);
	if (!pool_param.has_pad_h()) for (int i = 0; i < num_spatial_axes_; i++)
		pad_shape_.mutable_cpu_data()[i] = (pool_param.pad_size() == 0 ? 0 : 
		pool_param.pad((pool_param.pad_size() == 1) ? 0 : i));
	else {
		pad_shape_.mutable_cpu_data()[0] = pool_param.pad_h();
		pad_shape_.mutable_cpu_data()[1] = pool_param.pad_w();
	}

	// Setup filter stride dimensions(stride_shape_).
	stride_shape_.Reshape(spatial_dim_blob_shape);
	if (!pool_param.has_stride_h())  for (int i = 0; i < num_spatial_axes_; i++)
		stride_shape_.mutable_cpu_data()[i] = (pool_param.stride_size() == 0 ? 1 : 
		pool_param.stride((pool_param.stride_size() == 1) ? 0 : i));
	else {
		stride_shape_.mutable_cpu_data()[0] = pool_param.stride_h();
		stride_shape_.mutable_cpu_data()[1] = pool_param.stride_w();
	}
	if (global_pooling_) for (int i = 0; i < num_spatial_axes_; i++) {
		CHECK_EQ(pad_shape_.cpu_data()[i], 0) << "With Global_pooling: true; only pad = 0";
		CHECK_EQ(stride_shape_.cpu_data()[i], 1) << "With Global_pooling: true; only stride = 1";
	}
	bool pad_all_zero = false;
	for (int i = 0; i < num_spatial_axes_; i++) {
		if (pad_shape_.cpu_data()[i]) 
			pad_all_zero = true;
	}
	if (!pad_all_zero) {
		CHECK(this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_AVE
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_MAX
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_DEF
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_DEF_ALL
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_DEF_ALL2
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_DEF_ALL3
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_DEF_ALL4
			|| this->layer_param_.pooling_param().pool()
			== PoolingParameter_PoolMethod_LOWRES)
			<< "Padding implemented only for average and max pooling.";
		for (int i = 0; i < num_spatial_axes_; i++)
			CHECK_LT(pad_shape_.cpu_data()[i], stride_shape_.cpu_data()[i]);
	}
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
	const vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));

	// input/output shape
	in_channel_shape_.Reshape(bottom_dim_blob_shape);
	for (int i = 0; i < num_spatial_axes_ + 1; i++)
		in_channel_shape_.mutable_cpu_data()[i] = bottom[0]->shape(channel_axis_ + i);
	if (global_pooling_) for (int i = 0; i < num_spatial_axes_; i++)
		kernel_shape_.mutable_cpu_data()[i] = bottom[0]->shape(channel_axis_ + 1 + i);
	out_shape_.Reshape(spatial_dim_blob_shape);
	for (int i = 0; i < num_spatial_axes_; i++)
		out_shape_.mutable_cpu_data()[i] = static_cast<int>(ceil(static_cast<float>(
			bottom[0]->shape(channel_axis_ + 1 + i) + 2 * pad_shape_.cpu_data()[i]
			- kernel_shape_.cpu_data()[i]) / stride_shape_.cpu_data()[i])) + 1;

	// with pad
	bool pad_all_zero = false;
	for (int i = 0; i < num_spatial_axes_; i++) {
		if (pad_shape_.mutable_cpu_data()[i])
			pad_all_zero = true;
	}
	if (!pad_all_zero) {
		// If we have padding, ensure that the last pooling starts strictly
		// inside the image (instead of at the padding); otherwise clip the last.
		for (int i = 0; i < num_spatial_axes_; i++)
		if ((out_shape_.cpu_data()[i] - 1) * stride_shape_.cpu_data()[i] >= 
			in_channel_shape_.cpu_data()[i+1] + pad_shape_.cpu_data()[i]) 
		{
			--out_shape_.mutable_cpu_data()[i];
			CHECK_LT((out_shape_.cpu_data()[i] - 1) * stride_shape_.cpu_data()[i], 
				in_channel_shape_.cpu_data()[i + 1] + pad_shape_.cpu_data()[i]);
		}
	}
	std::vector<int> topShape(bottom[0]->num_axes(), 1);
	for (int i = 0; i < channel_axis_+1; i++)
		topShape[i] = bottom[0]->shape(i);
	for (int i = 0; i < num_spatial_axes_; i++)
		topShape[channel_axis_ + 1 + i] = out_shape_.cpu_data()[i];
	top[0]->Reshape(topShape);// set output mapsize
	if (top.size() > 1) 
		top[1]->ReshapeLike(*top[0]);
	// If max pooling, we will initialize the vector index part.
	if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX && top.size() == 1)
		max_idx_.Reshape(topShape);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	CHECK_EQ(bottom[0]->num_axes(), 4);
	const int channels = in_channel_shape_.cpu_data()[0];
	const int height = in_channel_shape_.cpu_data()[1];
	const int width = in_channel_shape_.cpu_data()[2];
	const int kernel_h = kernel_shape_.cpu_data()[0];
	const int kernel_w = kernel_shape_.cpu_data()[1];
	const int pad_h = pad_shape_.cpu_data()[0];
	const int pad_w = pad_shape_.cpu_data()[1];
	const int stride_h = stride_shape_.cpu_data()[0];
	const int stride_w = stride_shape_.cpu_data()[1];
	const int pooled_height = out_shape_.cpu_data()[0];
	const int pooled_width = out_shape_.cpu_data()[1];
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int top_count = top[0]->count();
	// We'll output the mask to top[1] if it's of size >1.
	const bool use_top_mask = top.size() > 1;
	int* mask = NULL;  // suppress warnings about uninitalized variables
	Dtype* top_mask = NULL;
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more code.
	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		// Initialize
		if (use_top_mask) {
			top_mask = top[1]->mutable_cpu_data();
			caffe_set(top_count, Dtype(-1), top_mask);
		}
		else {
			mask = max_idx_.mutable_cpu_data();
			caffe_set(top_count, -1, mask);
		}
		caffe_set(top_count, Dtype(-FLT_MAX), top_data);
		// The main loop
		for (int n = 0; n < bottom[0]->shape(0); ++n) {
			for (int c = 0; c < channels; ++c) {
				for (int ph = 0; ph < pooled_height; ++ph) {
					for (int pw = 0; pw < pooled_width; ++pw) {
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend = min(hstart + kernel_h, height);
						int wend = min(wstart + kernel_w, width);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						const int pool_index = ph * pooled_width + pw;
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								const int index = h * width + w;
								if (bottom_data[index] > top_data[pool_index]) {
									top_data[pool_index] = bottom_data[index];
									if (use_top_mask) {
										top_mask[pool_index] = static_cast<Dtype>(index);
									}
									else {
										mask[pool_index] = index;
									}
								}
							}
						}
					}
				}
				// compute offset
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
				if (use_top_mask) {
					top_mask += top[0]->offset(0, 1);
				}
				else {
					mask += top[0]->offset(0, 1);
				}
			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		for (int i = 0; i < top_count; ++i) {
			top_data[i] = 0;
		}
		// The main loop
		for (int n = 0; n < bottom[0]->shape(0); ++n) {
			for (int c = 0; c < channels; ++c) {
				for (int ph = 0; ph < pooled_height; ++ph) {
					for (int pw = 0; pw < pooled_width; ++pw) {
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend = min(hstart + kernel_h, height + pad_h);
						int wend = min(wstart + kernel_w, width + pad_w);
						int pool_size = (hend - hstart) * (wend - wstart);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						hend = min(hend, height);
						wend = min(wend, width);
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								top_data[ph * pooled_width + pw] +=
									bottom_data[h * width + w];
							}
						}
						top_data[ph * pooled_width + pw] /= pool_size;
					}
				}
				// compute offset
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	CHECK_EQ(bottom[0]->num_axes(), 4);
	const int channels = in_channel_shape_.cpu_data()[0];
	const int height = in_channel_shape_.cpu_data()[1];
	const int width = in_channel_shape_.cpu_data()[2];
	const int kernel_h = kernel_shape_.cpu_data()[0];
	const int kernel_w = kernel_shape_.cpu_data()[1];
	const int pad_h = pad_shape_.cpu_data()[0];
	const int pad_w = pad_shape_.cpu_data()[1];
	const int stride_h = stride_shape_.cpu_data()[0];
	const int stride_w = stride_shape_.cpu_data()[1];
	const int pooled_height = out_shape_.cpu_data()[0];
	const int pooled_width = out_shape_.cpu_data()[1];
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more codes.
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	// We'll output the mask to top[1] if it's of size >1.
	const bool use_top_mask = top.size() > 1;
	const int* mask = NULL;  // suppress warnings about uninitialized variables
	const Dtype* top_mask = NULL;
	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		// The main loop
		if (use_top_mask) {
			top_mask = top[1]->cpu_data();
		}
		else {
			mask = max_idx_.cpu_data();
		}
		for (int n = 0; n < top[0]->shape(0); ++n) {
			for (int c = 0; c < channels; ++c) {
				for (int ph = 0; ph < pooled_height; ++ph) {
					for (int pw = 0; pw < pooled_width; ++pw) {
						const int index = ph * pooled_width + pw;
						const int bottom_index =
							use_top_mask ? top_mask[index] : mask[index];
						bottom_diff[bottom_index] += top_diff[index];
					}
				}
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
				if (use_top_mask) {
					top_mask += top[0]->offset(0, 1);
				}
				else {
					mask += top[0]->offset(0, 1);
				}
			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		// The main loop
		for (int n = 0; n < top[0]->shape(0); ++n) {
			for (int c = 0; c < channels; ++c) {
				for (int ph = 0; ph < pooled_height; ++ph) {
					for (int pw = 0; pw < pooled_width; ++pw) {
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend = min(hstart + kernel_h, height + pad_h);
						int wend = min(wstart + kernel_w, width + pad_w);
						int pool_size = (hend - hstart) * (wend - wstart);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						hend = min(hend, height);
						wend = min(wend, width);
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								bottom_diff[h * width + w] +=
									top_diff[ph * pooled_width + pw] / pool_size;
							}
						}
					}
				}
				// offset
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}

#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
