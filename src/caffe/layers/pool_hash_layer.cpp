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
	//printf("************Pooling layer: input channels %d*******\n", channels_);
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
	//memset(top[HASH_DATA_BLOB]->mutable_cpu_data(), 0, sizeof(Dtype)*batch_hash_size * top_channels);

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
void PoolHashLayer<Dtype>::backward_cpu_max(float *bottom_dif, int bottom_m_bar,
	const float *top_dif, const PACKED_POSITION *top_posTag, int top_m_bar,
	const int *mask, int channels)
{
	const int top_m = top_m_bar * top_m_bar * top_m_bar;
	const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
	//init dif to zero
	caffe_set(bottom_m*channels, (float)0, bottom_dif);
	for (int v = 0; v < top_m; v++)
	{
		//if the hash voxel is undefined, skip
		if (!ishashVoxelDefined(&top_posTag[v]))
		{
			continue;
		}
		///////////////////////////////////////////

		const float *tp_dif_ptr = &top_dif[v];
		const int *mask_ptr = &mask[v];
		float *bt_dif_start = bottom_dif;
		//init to min
		for (int c = 0; c < channels; c++)
		{
			const int bt_m_idx = *mask_ptr;
			
			bt_dif_start[bt_m_idx] += *tp_dif_ptr;

			tp_dif_ptr += top_m;
			mask_ptr += top_m;
			bt_dif_start += bottom_m;
		}
	}
}

template <typename Dtype>
void PoolHashLayer<Dtype>::Backward_cpu_max(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	float *bt_hash_dif = (float*)bottom[HASH_DATA_BLOB]->mutable_cpu_diff();
	const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();

	const float *tp_hash_dif = (const float*)top[HASH_DATA_BLOB]->cpu_diff();
	const unsigned char*tp_offset = (const unsigned char *)bottom[OFFSET_BLOB + HASH_STRUCTURE_SIZE]->cpu_data();
	const PACKED_POSITION *tp_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB + HASH_STRUCTURE_SIZE]->cpu_data();

	int *mask = max_idx_.mutable_cpu_data();

	int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int bt_dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
	const int tp_dense_res = (int)top[DENSE_RES_BLOB]->cpu_data()[0];

	for (int i = 0; i < batch_num; ++i)
	{
		float* cur_bt_dif = bt_hash_dif;
		const unsigned char* cur_bt_offset = bt_offset;
		const PACKED_POSITION *cur_bt_postag = bt_posTag;
		const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];


		const float *cur_tp_dif = tp_hash_dif;
		int *cur_mask = mask;
		const unsigned char*cur_tp_offset = tp_offset;
		const PACKED_POSITION *cur_tp_postag = tp_posTag;
		const int tp_m_bar = (int)bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];
		const int tp_r_bar = (int)bottom[R_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];

		backward_cpu_max(cur_bt_dif, bt_m_bar, 
			cur_tp_dif, cur_tp_postag, tp_m_bar, 
			cur_mask, channels_);

#if 1	//for debug
		float *bt_dense_buf = new float[bt_dense_res * bt_dense_res * bt_dense_res * channels_];
		hash_2_dense(cur_bt_dif, cur_bt_postag, cur_bt_offset, bt_m_bar,
			bt_r_bar, channels_, bt_dense_buf, bt_dense_res);
		char buf[128];
		sprintf(buf, "bottom_dif_%d.grid", i);
		writeDense_2_Grid(bt_dense_buf, bt_dense_res, channels_, buf);
		

		float *tp_dense_buf = new float[tp_dense_res * tp_dense_res * tp_dense_res * channels_];
		hash_2_dense(cur_tp_dif, cur_tp_postag, cur_tp_offset, tp_m_bar,
			tp_r_bar, channels_, tp_dense_buf, tp_dense_res);
		int *tp_idx_buf = new int[tp_dense_res * tp_dense_res * tp_dense_res * channels_];
		topMask_2_dense(cur_mask, cur_tp_postag, cur_tp_offset,
			tp_m_bar, tp_r_bar, channels_, tp_dense_res, cur_bt_postag, bt_dense_res, tp_idx_buf);
		

		float *bt_dense_debug = new float[bt_dense_res * bt_dense_res * bt_dense_res * channels_];
		bp_max_dense(tp_dense_buf, tp_idx_buf, bt_dense_debug, tp_dense_res, bt_dense_res, channels_);

		//for (int tt=0;tt<bt_dense_res * bt_dense_res * bt_dense_res * channels_;tt++)
		//{
		//	if (bt_dense_debug[tt]!=bt_dense_buf[tt])
		//	{
		//		printf("Error!\n");
		//	}
		//}

		sprintf(buf, "bottom_dif_dense_%d.grid", i);
		writeDense_2_Grid(bt_dense_debug, bt_dense_res, channels_, buf);

		delete[]tp_dense_buf;
		delete[]bt_dense_buf;
		delete[]tp_idx_buf;
		delete[]bt_dense_debug;

#endif

		//to next hash
		const int bt_m = bt_m_bar * bt_m_bar * bt_m_bar;
		const int bt_r = bt_r_bar * bt_r_bar * bt_r_bar;
		bt_hash_dif += bt_m * channels_;
		bt_offset += bt_r * 3;
		bt_posTag += bt_m;

		const int tp_m = tp_m_bar * tp_m_bar * tp_m_bar;
		const int tp_r = tp_r_bar * tp_r_bar * tp_r_bar;
		tp_hash_dif += tp_m * channels_;
		mask += tp_m * channels_;
		tp_offset += tp_r * 3;
		tp_posTag += tp_m;
	}
}

template <typename Dtype>
void PoolHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		Backward_cpu_max(top, propagate_down,bottom);
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

#ifdef CPU_ONLY
STUB_GPU(PoolHashLayer);
#endif

INSTANTIATE_CLASS(PoolHashLayer);

}  // namespace caffe


void bp_max_dense(const float *top_dif, const int *top_mask, float *bottom_dif, int top_res, int bottom_res, int channels)
{
	int top_n = top_res * top_res * top_res;
	int bottom_n = bottom_res * bottom_res * bottom_res;
	memset(bottom_dif,0,sizeof(float)*channels*bottom_n);

	for (int c=0;c<channels;c++)
	{
		const float *cur_top_dif = top_dif + top_n*c;
		float *cur_bottom_dif = bottom_dif + bottom_n*c;
		const int *cur_mask = top_mask + top_n*c;
		for (int ti=0;ti<top_n;ti++)
		{
			int bi = cur_mask[ti];
			if (bi<0)
			{
				continue;
			}
			cur_bottom_dif[bi] += cur_top_dif[ti];
		}
	}
}