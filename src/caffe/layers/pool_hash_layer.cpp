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
	if (bottom.size() != HASH_DATA_SIZE + HASH_STRUCTURE_SIZE + HASH_STRUCTURE_SIZE)	//data + input struct + out struct
	{
		printf("Fatal error: bottom size should be %d\n", HASH_DATA_SIZE + HASH_STRUCTURE_SIZE + HASH_STRUCTURE_SIZE);
		exit(0);
	}
	if (top.size() != HASH_DATA_SIZE)
	{
		printf("Fatal error: top size should be 1\n");
		exit(0);
	}


	num_spatial_axes_ = 3;	//for 3D case


	
							// Configure the kernel size, padding, stride, and inputs.
	const PoolHashParameter &pool_hash_param = this->layer_param_.pool_hash_param();

	const vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
	
	//// Setup filter kernel dimensions(kernel_shape_).
	//kernel_shape_.Reshape(spatial_dim_blob_shape);
	//int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
	//{
	//	const int num_kernel_dims = pool_hash_param.kernel_size_size();
	//	CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
	//		<< "kernel_size must be specified once, or once per spatial dimension "
	//		<< "(kernel_size specified " << num_kernel_dims << " times; "
	//		<< num_spatial_axes_ << " spatial dims).";
	//	for (int i = 0; i < num_spatial_axes_; ++i) {
	//		kernel_shape_data[i] =
	//			pool_hash_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
	//	}
	//}
	//for (int i = 0; i < num_spatial_axes_; ++i) {
	//	CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
	//}

	//// Setup filter pad dimensions(pad_shape_).
	//pad_shape_.Reshape(spatial_dim_blob_shape);
	//for (int i = 0; i < num_spatial_axes_; i++)
	//	pad_shape_.mutable_cpu_data()[i] = (pool_hash_param.pad_size() == 0 ? 0 :
	//		pool_hash_param.pad((pool_hash_param.pad_size() == 1) ? 0 : i));
	

	// Setup filter stride dimensions(stride_shape_).
	stride_shape_.Reshape(spatial_dim_blob_shape);
	for (int i = 0; i < num_spatial_axes_; i++)
		stride_shape_.mutable_cpu_data()[i] = (pool_hash_param.stride_size() == 0 ? 1 :
			pool_hash_param.stride((pool_hash_param.stride_size() == 1) ? 0 : i));
}



template <typename Dtype>
void PoolHashLayer<Dtype>::reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	// Configure output channels and groups.
	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	printf("************Pooling layer: input channels %d*******\n", channels_);
	CHECK_GT(channels_, 0);

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
		max_idx_.Reshape(hash_data_shape);
	}	

	//reshape top channel and dense res
	std::vector<int> scalar_shape(1, 1);
	top[CHANNEL_BLOB]->Reshape(scalar_shape);
	top[DENSE_RES_BLOB]->Reshape(scalar_shape);
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
	const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar, 
	int *mask,
	int channels, int bt_dense_res)
{
	const int top_m = top_m_bar * top_m_bar * top_m_bar;
	const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
	const int stride_x = stride_shape_.cpu_data()[0];
	const int stride_y = stride_shape_.cpu_data()[1];
	const int stride_z = stride_shape_.cpu_data()[2];

	if (stride_x!=stride_y || stride_x!=stride_z)
	{
		printf("Fatal error: we only consider same strides!\n");
		exit(0);
	}

	const int in = bt_dense_res;//input dense res
	const int in2 = bt_dense_res*bt_dense_res;

	const int bottom_r2 = bottom_r_bar * bottom_r_bar;
	const int bottom_m2 = bottom_m_bar * bottom_m_bar;
	//init mask
	caffe_set(top_m*channels, -1, mask);
	for (int v = 0; v < top_m; v++)
	{
		//if the hash voxel is undefined, skip
		if (!ishashVoxelDefined(&top_posTag[v]))
		{
			continue;
		}
		///////////////////////////////////////////

		float *tp_hash_ptr = &top_hash[v];
		//init to min
		for (int c = 0;c<channels;c++)
		{
			*tp_hash_ptr = -FLT_MAX;
			tp_hash_ptr += top_m;
		}

		//get the real voxel position from the position tag
		int cx, cy, cz;
		xyz_from_pack(top_posTag[v], cx, cy, cz);	//get the voxel position mapped to this hash														

		int min_x = cx * stride_x;
		int min_y = cy * stride_y;
		int min_z = cz * stride_z;

		int x_end = min(min_x + stride_x, bt_dense_res);
		int y_end = min(min_y + stride_y, bt_dense_res);
		int z_end = min(min_z + stride_z, bt_dense_res);
		
		min_x = max(min_x, 0);
		min_y = max(min_y, 0);
		min_z = max(min_z, 0);
		
		int bt_mx, bt_my, bt_mz;

		for (int nz = min_z; nz < z_end; ++nz)
		{
			const int depth_idx = nz * in2;
			for (int ny = min_y; ny < y_end; ++ny)
			{
				const int height_idx = ny * in;
				for (int nx = min_x; nx< x_end;++nx)
				{
					const int dense_idx = depth_idx + height_idx + nx;

					//hash to get hash position
					Hash(nx, ny, nz, bt_mx, bt_my, bt_mz,
						bottom_offset, bottom_m_bar, bottom_r_bar, bottom_r2);
					const int bt_m_idx = NXYZ2I(bt_mx, bt_my, bt_mz, bottom_m_bar, bottom_m2);

					if (!ishashVoxelDefined(&bottom_posTag[bt_m_idx]))	//the bottom hash voxel is undefined
					{
						continue;
					}

					int stored_x, stored_y, stored_z;
					xyz_from_pack(bottom_posTag[bt_m_idx], stored_x, stored_y, stored_z);
					if (nx != stored_x || ny != stored_y || nz != stored_z)	//undefined dense voxel
					{
						continue;
					}
					
					const float *bt_hash_ptr = &bottom_hash[bt_m_idx];
					tp_hash_ptr = &top_hash[v];	
					int *mask_ptr = &mask[v];
					for (int c = 0; c < channels; c++)
					{
						if (*tp_hash_ptr < *bt_hash_ptr)
						{
							*tp_hash_ptr = *bt_hash_ptr;
							*mask_ptr = bt_m_idx;
						}
						tp_hash_ptr += top_m;
						mask_ptr += top_m;
						bt_hash_ptr += bottom_m;
					}
				}
			}
		}
	}
}

template <typename Dtype>
void PoolHashLayer<Dtype>::Forward_cpu_max(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const float *bt_hash = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
	const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();

	float *tp_hash = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
	const unsigned char*tp_offset = (const unsigned char *)bottom[OFFSET_BLOB + HASH_STRUCTURE_SIZE]->cpu_data();
	const PACKED_POSITION *tp_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB + HASH_STRUCTURE_SIZE]->cpu_data();

	int *mask = max_idx_.mutable_cpu_data();

	int batch_num = bottom[M_BAR_BLOB]->shape(0);
	const int bt_dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
	const int tp_dense_res = (int)top[DENSE_RES_BLOB]->cpu_data()[0];
	for (int i = 0; i < batch_num; ++i)
	{
		const float* cur_bt_hash = bt_hash;
		const unsigned char* cur_bt_offset = bt_offset;
		const PACKED_POSITION *cur_bt_postag = bt_posTag;
		const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		

		float *cur_tp_hash = tp_hash;
		int *cur_mask = mask;
		const unsigned char*cur_tp_offset = tp_offset;
		const PACKED_POSITION *cur_tp_postag = tp_posTag;
		const int tp_m_bar = (int)bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];
		const int tp_r_bar = (int)bottom[R_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];



		forward_cpu_max(cur_bt_hash, cur_bt_offset, cur_bt_postag, bt_m_bar, bt_r_bar,
			cur_tp_hash, cur_tp_offset, cur_tp_postag, tp_m_bar, tp_r_bar,
			cur_mask, channels_, bt_dense_res);

#if 1	//for debug
		float *bt_dense_buf = new float[bt_dense_res * bt_dense_res * bt_dense_res * channels_];
		hash_2_dense(cur_bt_hash, cur_bt_postag, cur_bt_offset, bt_m_bar, 
			bt_r_bar, channels_, bt_dense_buf, bt_dense_res);
		char buf[128];
		sprintf(buf,"bottom_%d.grid",i);
		writeDense_2_Grid(bt_dense_buf, bt_dense_res, channels_, buf);
		delete[]bt_dense_buf;

		float *tp_dense_buf = new float[tp_dense_res * tp_dense_res * tp_dense_res * channels_];
		hash_2_dense(cur_tp_hash, cur_tp_postag, cur_tp_offset, tp_m_bar,
			tp_r_bar, channels_, tp_dense_buf, tp_dense_res);
		sprintf(buf, "top_%d.grid", i);
		writeDense_2_Grid(tp_dense_buf, tp_dense_res, channels_, buf);
		delete[]tp_dense_buf;
#endif
		
		//to next hash
		const int bt_m = bt_m_bar * bt_m_bar * bt_m_bar;
		const int bt_r = bt_r_bar * bt_r_bar * bt_r_bar;
		bt_hash += bt_m * channels_;
		bt_offset += bt_r * 3;
		bt_posTag += bt_m;

		const int tp_m = tp_m_bar * tp_m_bar * tp_m_bar;
		const int tp_r = tp_r_bar * tp_r_bar * tp_r_bar;
		tp_hash += tp_m * channels_;
		mask += tp_m * channels_;
		tp_offset += tp_r * 3;
		tp_posTag += tp_m;
	}
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	//fill the channel and dense for next layer
	top[CHANNEL_BLOB]->mutable_cpu_data()[0] = bottom[CHANNEL_BLOB]->cpu_data()[0];

	const int stride = stride_shape_.cpu_data()[0];

	top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = ((int)bottom[DENSE_RES_BLOB]->cpu_data()[0]/ stride);
	printf("top dense resolution %d\n", (int)top[DENSE_RES_BLOB]->cpu_data()[0]);

	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		Forward_cpu_max(bottom, top);
		break;
	case PoolingParameter_PoolMethod_AVE:
		NOT_IMPLEMENTED;
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
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

INSTANTIATE_CLASS(PoolHashLayer);

}  // namespace caffe
