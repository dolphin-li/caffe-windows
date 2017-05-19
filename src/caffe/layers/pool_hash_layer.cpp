#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pool_hash_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolHashLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	//CHECK size
	if (bottom.size() != 1 + HASH_STRUCTURE_SIZE + HASH_STRUCTURE_SIZE)	//data + input struct + out struct
	{
		printf("Fatal error: bottom size should be %d\n", 1 + HASH_STRUCTURE_SIZE + HASH_STRUCTURE_SIZE);
		exit(0);
	}
	if (top.size() != 1)
	{
		printf("Fatal error: top size should be 1\n");
		exit(0);
	}


	num_spatial_axes_ = 3;	//for 3D case


	
							// Configure the kernel size, padding, stride, and inputs.
	const PoolHashParameter &pool_hash_param = this->layer_param_.pool_hash_param();

	const vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
	
	// Setup filter kernel dimensions(kernel_shape_).
	kernel_shape_.Reshape(spatial_dim_blob_shape);
	int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
	{
		const int num_kernel_dims = pool_hash_param.kernel_size_size();
		CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
			<< "kernel_size must be specified once, or once per spatial dimension "
			<< "(kernel_size specified " << num_kernel_dims << " times; "
			<< num_spatial_axes_ << " spatial dims).";
		for (int i = 0; i < num_spatial_axes_; ++i) {
			kernel_shape_data[i] =
				pool_hash_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
		}
	}
	for (int i = 0; i < num_spatial_axes_; ++i) {
		CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
	}

	// Setup filter pad dimensions(pad_shape_).
	pad_shape_.Reshape(spatial_dim_blob_shape);
	for (int i = 0; i < num_spatial_axes_; i++)
		pad_shape_.mutable_cpu_data()[i] = (pool_hash_param.pad_size() == 0 ? 0 :
			pool_hash_param.pad((pool_hash_param.pad_size() == 1) ? 0 : i));
	

	// Setup filter stride dimensions(stride_shape_).
	stride_shape_.Reshape(spatial_dim_blob_shape);
	for (int i = 0; i < num_spatial_axes_; i++)
		stride_shape_.mutable_cpu_data()[i] = (pool_hash_param.stride_size() == 0 ? 1 :
			pool_hash_param.stride((pool_hash_param.stride_size() == 1) ? 0 : i));
	
	// Configure output channels and groups.
	channels_ = pool_hash_param.input_channels();
	printf("************Pooling layer: input channels %d*******\n",channels_);
	CHECK_GT(channels_, 0);

}



template <typename Dtype>
void PoolHashLayer<Dtype>::reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{

	const Blob<Dtype> *bottom_m_bar_blob = bottom[M_BAR_BLOB];
	const Blob<Dtype> *top_m_bar_blob = bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE];
	if (!bottom_m_bar_blob->num_axes() || !top_m_bar_blob->num_axes())
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
		exit(0);
		return;
	}
	const int top_channels = channels_;
	const int batch_num = bottom_m_bar_blob->shape(0);
	if (batch_num != top_m_bar_blob->shape(0))
	{
		printf("Error: bottom hash num != top hash num!\n");
		exit(0);
		return;
	}
	int batch_hash_size = 0;
	for (int i = 0; i < batch_num; i++)
	{
		int m_bar = (int)top_m_bar_blob->cpu_data()[i];
		batch_hash_size += m_bar * m_bar * m_bar;
	}
	std::vector<int> hash_data_shape(1, batch_hash_size * top_channels);
	top[HASH_DATA_BLOB]->Reshape(hash_data_shape);
	memset(top[HASH_DATA_BLOB]->mutable_cpu_data(), 0, sizeof(Dtype)*batch_hash_size * top_channels);

	//also reshape max_idx_
	
	if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX)
	{
		std::vector<int> hash_data_shape(1, batch_hash_size);
		max_idx_.Reshape(hash_data_shape);
	}	
}

template <typename Dtype>
void PoolHashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	reshape_topHashData(bottom, top);
}

template <typename Dtype>
void PoolHashLayer<Dtype>::forward_cpu_max(const float *bottom_hash, const unsigned char *bottom_offset,
	const PACKED_POSITION *bottom_posTag, int bottom_m_bar, int bottom_r_bar,
	float *top_hash, const unsigned char *top_offset,
	const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar, int dense_res)
{
	//const int top_m = top_m_bar * top_m_bar * top_m_bar;

	//const int stride_x = stride_shape_.cpu_data()[0];
	//const int stride_y = stride_shape_.cpu_data()[1];
	//const int stride_z = stride_shape_.cpu_data()[2];

	//for (int v = 0; v < top_m; v++)
	//{
	//	//if the hash voxel is undefined, skip
	//	if (!ishashVoxelDefined(&top_posTag[v]))
	//	{
	//		//data_ptr++;
	//		continue;
	//	}
	//	///////////////////////////////////////////

	//	//get the real voxel position from the position tag
	//	int cx, cy, cz;
	//	xyz_from_pack(top_posTag[v], cx, cy, cz);	//get the voxel position mapped to this hash														

	//	int min_x = cx * stride_x;
	//	int min_y = cy * stride_y;
	//	int min_z = cz * stride_z;


	//	int hend = min(hstart + kernel_h, height);
	//	int wend = min(wstart + kernel_w, width);
	//	hstart = max(hstart, 0);
	//	wstart = max(wstart, 0);
	//	
	//	int min_x = cx;
	//	int min_y = cy - hy;
	//	int min_z = cz - hz;
	//	int mx, my, mz;
	//	float *cur_row = col_ptr;
	//}
}

template <typename Dtype>
void PoolHashLayer<Dtype>::Forward_cpu_max(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	//const int top_channels = channels_;
	//const int top_count = max_idx_.shape(0);
	//int *mask = max_idx_.mutable_cpu_data();
	//caffe_set(top_count, -1, mask);
	//Dtype* top_data = top[0]->mutable_cpu_data();
	//caffe_set(top_count*top_channels, Dtype(-FLT_MAX), top_data);
	//// The main loop
	//const int batch_num = bottom_m_bar_blob->shape(0);
	//for (int n = 0; n < batch_num; ++n) 
	//{
	//	for (int c = 0; c < channels; ++c) {
	//		for (int ph = 0; ph < pooled_height; ++ph) {
	//			for (int pw = 0; pw < pooled_width; ++pw) {
	//				int hstart = ph * stride_h - pad_h;
	//				int wstart = pw * stride_w - pad_w;
	//				int hend = min(hstart + kernel_h, height);
	//				int wend = min(wstart + kernel_w, width);
	//				hstart = max(hstart, 0);
	//				wstart = max(wstart, 0);
	//				const int pool_index = ph * pooled_width + pw;
	//				for (int h = hstart; h < hend; ++h) {
	//					for (int w = wstart; w < wend; ++w) {
	//						const int index = h * width + w;
	//						if (bottom_data[index] > top_data[pool_index]) {
	//							top_data[pool_index] = bottom_data[index];
	//							if (use_top_mask) {
	//								top_mask[pool_index] = static_cast<Dtype>(index);
	//							}
	//							else {
	//								mask[pool_index] = index;
	//							}
	//						}
	//					}
	//				}
	//			}
	//		}
	//		// compute offset
	//		bottom_data += bottom[0]->offset(0, 1);
	//		top_data += top[0]->offset(0, 1);
	//		if (use_top_mask) {
	//			top_mask += top[0]->offset(0, 1);
	//		}
	//		else {
	//			mask += top[0]->offset(0, 1);
	//		}
	//	}
	//}
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	//CHECK_EQ(bottom[0]->num_axes(), 4);
	//const int channels = in_channel_shape_.cpu_data()[0];
	//const int height = in_channel_shape_.cpu_data()[1];
	//const int width = in_channel_shape_.cpu_data()[2];
	//const int kernel_h = kernel_shape_.cpu_data()[0];
	//const int kernel_w = kernel_shape_.cpu_data()[1];
	//const int pad_h = pad_shape_.cpu_data()[0];
	//const int pad_w = pad_shape_.cpu_data()[1];
	//const int stride_h = stride_shape_.cpu_data()[0];
	//const int stride_w = stride_shape_.cpu_data()[1];
	//const int pooled_height = out_shape_.cpu_data()[0];
	//const int pooled_width = out_shape_.cpu_data()[1];
	//const Dtype* bottom_data = bottom[0]->cpu_data();
	//Dtype* top_data = top[0]->mutable_cpu_data();
	//const int top_count = top[0]->count();
	//// We'll output the mask to top[1] if it's of size >1.
	//const bool use_top_mask = top.size() > 1;
	//int* mask = NULL;  // suppress warnings about uninitalized variables
	//Dtype* top_mask = NULL;
	//// Different pooling methods. We explicitly do the switch outside the for
	//// loop to save time, although this results in more code.
	//switch (this->layer_param_.pooling_param().pool()) {
	//case PoolingParameter_PoolMethod_MAX:
	//	// Initialize
	//	if (use_top_mask) {
	//		top_mask = top[1]->mutable_cpu_data();
	//		caffe_set(top_count, Dtype(-1), top_mask);
	//	}
	//	else {
	//		mask = max_idx_.mutable_cpu_data();
	//		caffe_set(top_count, -1, mask);
	//	}
	//	caffe_set(top_count, Dtype(-FLT_MAX), top_data);
	//	// The main loop
	//	for (int n = 0; n < bottom[0]->shape(0); ++n) {
	//		for (int c = 0; c < channels; ++c) {
	//			for (int ph = 0; ph < pooled_height; ++ph) {
	//				for (int pw = 0; pw < pooled_width; ++pw) {
	//					int hstart = ph * stride_h - pad_h;
	//					int wstart = pw * stride_w - pad_w;
	//					int hend = min(hstart + kernel_h, height);
	//					int wend = min(wstart + kernel_w, width);
	//					hstart = max(hstart, 0);
	//					wstart = max(wstart, 0);
	//					const int pool_index = ph * pooled_width + pw;
	//					for (int h = hstart; h < hend; ++h) {
	//						for (int w = wstart; w < wend; ++w) {
	//							const int index = h * width + w;
	//							if (bottom_data[index] > top_data[pool_index]) {
	//								top_data[pool_index] = bottom_data[index];
	//								if (use_top_mask) {
	//									top_mask[pool_index] = static_cast<Dtype>(index);
	//								}
	//								else {
	//									mask[pool_index] = index;
	//								}
	//							}
	//						}
	//					}
	//				}
	//			}
	//			// compute offset
	//			bottom_data += bottom[0]->offset(0, 1);
	//			top_data += top[0]->offset(0, 1);
	//			if (use_top_mask) {
	//				top_mask += top[0]->offset(0, 1);
	//			}
	//			else {
	//				mask += top[0]->offset(0, 1);
	//			}
	//		}
	//	}
	//	break;
	//case PoolingParameter_PoolMethod_AVE:
	//	for (int i = 0; i < top_count; ++i) {
	//		top_data[i] = 0;
	//	}
	//	// The main loop
	//	for (int n = 0; n < bottom[0]->shape(0); ++n) {
	//		for (int c = 0; c < channels; ++c) {
	//			for (int ph = 0; ph < pooled_height; ++ph) {
	//				for (int pw = 0; pw < pooled_width; ++pw) {
	//					int hstart = ph * stride_h - pad_h;
	//					int wstart = pw * stride_w - pad_w;
	//					int hend = min(hstart + kernel_h, height + pad_h);
	//					int wend = min(wstart + kernel_w, width + pad_w);
	//					int pool_size = (hend - hstart) * (wend - wstart);
	//					hstart = max(hstart, 0);
	//					wstart = max(wstart, 0);
	//					hend = min(hend, height);
	//					wend = min(wend, width);
	//					for (int h = hstart; h < hend; ++h) {
	//						for (int w = wstart; w < wend; ++w) {
	//							top_data[ph * pooled_width + pw] +=
	//								bottom_data[h * width + w];
	//						}
	//					}
	//					top_data[ph * pooled_width + pw] /= pool_size;
	//				}
	//			}
	//			// compute offset
	//			bottom_data += bottom[0]->offset(0, 1);
	//			top_data += top[0]->offset(0, 1);
	//		}
	//	}
	//	break;
	//case PoolingParameter_PoolMethod_STOCHASTIC:
	//	NOT_IMPLEMENTED;
	//	break;
	//default:
	//	LOG(FATAL) << "Unknown pooling method.";
	//}
}

template <typename Dtype>
void PoolHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//if (!propagate_down[0]) {
	//	return;
	//}
	//CHECK_EQ(bottom[0]->num_axes(), 4);
	//const int channels = in_channel_shape_.cpu_data()[0];
	//const int height = in_channel_shape_.cpu_data()[1];
	//const int width = in_channel_shape_.cpu_data()[2];
	//const int kernel_h = kernel_shape_.cpu_data()[0];
	//const int kernel_w = kernel_shape_.cpu_data()[1];
	//const int pad_h = pad_shape_.cpu_data()[0];
	//const int pad_w = pad_shape_.cpu_data()[1];
	//const int stride_h = stride_shape_.cpu_data()[0];
	//const int stride_w = stride_shape_.cpu_data()[1];
	//const int pooled_height = out_shape_.cpu_data()[0];
	//const int pooled_width = out_shape_.cpu_data()[1];
	//const Dtype* top_diff = top[0]->cpu_diff();
	//Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	//// Different pooling methods. We explicitly do the switch outside the for
	//// loop to save time, although this results in more codes.
	//caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	//// We'll output the mask to top[1] if it's of size >1.
	//const bool use_top_mask = top.size() > 1;
	//const int* mask = NULL;  // suppress warnings about uninitialized variables
	//const Dtype* top_mask = NULL;
	//switch (this->layer_param_.pooling_param().pool()) {
	//case PoolingParameter_PoolMethod_MAX:
	//	// The main loop
	//	if (use_top_mask) {
	//		top_mask = top[1]->cpu_data();
	//	}
	//	else {
	//		mask = max_idx_.cpu_data();
	//	}
	//	for (int n = 0; n < top[0]->shape(0); ++n) {
	//		for (int c = 0; c < channels; ++c) {
	//			for (int ph = 0; ph < pooled_height; ++ph) {
	//				for (int pw = 0; pw < pooled_width; ++pw) {
	//					const int index = ph * pooled_width + pw;
	//					const int bottom_index =
	//						use_top_mask ? top_mask[index] : mask[index];
	//					bottom_diff[bottom_index] += top_diff[index];
	//				}
	//			}
	//			bottom_diff += bottom[0]->offset(0, 1);
	//			top_diff += top[0]->offset(0, 1);
	//			if (use_top_mask) {
	//				top_mask += top[0]->offset(0, 1);
	//			}
	//			else {
	//				mask += top[0]->offset(0, 1);
	//			}
	//		}
	//	}
	//	break;
	//case PoolingParameter_PoolMethod_AVE:
	//	// The main loop
	//	for (int n = 0; n < top[0]->shape(0); ++n) {
	//		for (int c = 0; c < channels; ++c) {
	//			for (int ph = 0; ph < pooled_height; ++ph) {
	//				for (int pw = 0; pw < pooled_width; ++pw) {
	//					int hstart = ph * stride_h - pad_h;
	//					int wstart = pw * stride_w - pad_w;
	//					int hend = min(hstart + kernel_h, height + pad_h);
	//					int wend = min(wstart + kernel_w, width + pad_w);
	//					int pool_size = (hend - hstart) * (wend - wstart);
	//					hstart = max(hstart, 0);
	//					wstart = max(wstart, 0);
	//					hend = min(hend, height);
	//					wend = min(wend, width);
	//					for (int h = hstart; h < hend; ++h) {
	//						for (int w = wstart; w < wend; ++w) {
	//							bottom_diff[h * width + w] +=
	//								top_diff[ph * pooled_width + pw] / pool_size;
	//						}
	//					}
	//				}
	//			}
	//			// offset
	//			bottom_diff += bottom[0]->offset(0, 1);
	//			top_diff += top[0]->offset(0, 1);
	//		}
	//	}
	//	break;
	//case PoolingParameter_PoolMethod_STOCHASTIC:
	//	NOT_IMPLEMENTED;
	//	break;
	//default:
	//	LOG(FATAL) << "Unknown pooling method.";
	//}
}

#ifdef CPU_ONLY
STUB_GPU(PoolHashLayer);
#endif
template <typename Dtype>
void PoolHashLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void PoolHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

}

INSTANTIATE_CLASS(PoolHashLayer);

}  // namespace caffe
