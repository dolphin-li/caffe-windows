#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pool_hash_layer.hpp"
#include "caffe\util\HashData.h"

namespace caffe {

	__global__ void forward_gpu_max_kernel(const float *bottom_hash, const unsigned char *bottom_offset,
		const PACKED_POSITION *bottom_posTag, int bottom_m_bar, int bottom_r_bar,
		float *top_hash, const unsigned char *top_offset,
		const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar,
		int *mask, int channels, int bt_dense_res, int3 stride)
	{
		const int top_m = top_m_bar * top_m_bar * top_m_bar;
		const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
		const int bottom_r2 = bottom_r_bar * bottom_r_bar;
		const int bottom_m2 = bottom_m_bar * bottom_m_bar;
		CUDA_KERNEL_LOOP(v, top_m)
		{
			//if the hash voxel is undefined, skip
			if (!ishashVoxelDefined_g(top_posTag[v]))
				continue;

			float *tp_hash_ptr = top_hash + v;

			//init to min
			for (int c = 0; c < channels; c++)
			{
				*tp_hash_ptr = -FLT_MAX;
				tp_hash_ptr += top_m;
			}

			//get the real voxel position from the position tag
			int cx, cy, cz;
			xyz_from_pack_g(top_posTag[v], cx, cy, cz);	//get the voxel position mapped to this hash														

			const int min_x = max(0, cx * stride.x);
			const int min_y = max(0, cy * stride.y);
			const int min_z = max(0, cz * stride.z);
			const int x_end = min(min_x + stride.x, bt_dense_res);
			const int y_end = min(min_y + stride.y, bt_dense_res);
			const int z_end = min(min_z + stride.z, bt_dense_res);

			int bt_mx, bt_my, bt_mz;
			for (int nz = min_z; nz < z_end; ++nz)
			for (int ny = min_y; ny < y_end; ++ny)
				for (int nx = min_x; nx < x_end; ++nx)
				{
					//hash to get hash position
					Hash_g(nx, ny, nz, bt_mx, bt_my, bt_mz,
						bottom_offset, bottom_m_bar, bottom_r_bar, bottom_r2);
					const int bt_m_idx = NXYZ2I_g(bt_mx, bt_my, bt_mz, bottom_m_bar, bottom_m2);

					//the bottom hash voxel is undefined
					if (!ishashVoxelDefined_g(bottom_posTag[bt_m_idx]))
						continue;

					int stored_x, stored_y, stored_z;
					xyz_from_pack_g(bottom_posTag[bt_m_idx], stored_x, stored_y, stored_z);
					if (nx != stored_x || ny != stored_y || nz != stored_z)	//undefined dense voxel
						continue;

					const float *bt_hash_ptr = bottom_hash + bt_m_idx;
					tp_hash_ptr = top_hash + v;
					int *mask_ptr = mask + v;
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
				} // end for nx, ny, nz
		} // end for caffe_kernel_loop
	}

	template<class Dtype>
	void PoolHashLayer<Dtype>::forward_gpu_max(const float *bottom_hash, const unsigned char *bottom_offset,
		const PACKED_POSITION *bottom_posTag, int bottom_m_bar, int bottom_r_bar,
		float *top_hash, const unsigned char *top_offset,
		const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar,
		int *mask, int channels, int bt_dense_res)
	{
		const int top_m = top_m_bar * top_m_bar * top_m_bar;
		const int stride_x = stride_shape_.cpu_data()[0];
		const int stride_y = stride_shape_.cpu_data()[1];
		const int stride_z = stride_shape_.cpu_data()[2];
		if (stride_x != stride_y || stride_x != stride_z)
		{
			printf("Fatal error: we only consider same strides!\n");
			exit(0);
		}

		//init mask
		caffe_gpu_set(top_m*channels, -1, mask);

		//max pool kernel
		forward_gpu_max_kernel << <CAFFE_GET_BLOCKS(top_m), CAFFE_CUDA_NUM_THREADS >> > (
			bottom_hash, bottom_offset, bottom_posTag, bottom_m_bar, bottom_r_bar,
			top_hash, top_offset, top_posTag, top_m_bar, top_r_bar,
			mask, channels, bt_dense_res, make_int3(stride_x, stride_y, stride_z)
			);	
	}

	template void PoolHashLayer<float>::forward_gpu_max(const float *bottom_hash, const unsigned char *bottom_offset,
		const PACKED_POSITION *bottom_posTag, int bottom_m_bar, int bottom_r_bar,
		float *top_hash, const unsigned char *top_offset,
		const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar,
		int *mask, int channels, int dense_res);

	template <typename Dtype>
	void PoolHashLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		//fill the channel and dense for next layer
		top[CHANNEL_BLOB]->mutable_cpu_data()[0] = bottom[CHANNEL_BLOB]->cpu_data()[0];
		const int stride = stride_shape_.cpu_data()[0];
		top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = ((int)bottom[DENSE_RES_BLOB]->cpu_data()[0] / stride);

		switch (this->layer_param_.pooling_param().pool()) {
		case PoolingParameter_PoolMethod_MAX:
			Forward_gpu_max(bottom, top);
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
	void PoolHashLayer<Dtype>::Forward_gpu_max(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const float *bt_hash = (const float*)bottom[HASH_DATA_BLOB]->gpu_data();
		const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();

		float *tp_hash = (float*)top[HASH_DATA_BLOB]->mutable_gpu_data();
		const unsigned char*tp_offset = (const unsigned char *)bottom[OFFSET_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const PACKED_POSITION *tp_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();

		int *mask = max_idx_.mutable_gpu_data();

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

			forward_gpu_max(cur_bt_hash, cur_bt_offset, cur_bt_postag, bt_m_bar, bt_r_bar,
				cur_tp_hash, cur_tp_offset, cur_tp_postag, tp_m_bar, tp_r_bar,
				cur_mask, channels_, bt_dense_res);

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

	template <typename Dtype>
	void PoolHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{

	}

	INSTANTIATE_LAYER_GPU_FUNCS(PoolHashLayer);
}  // namespace caffe
