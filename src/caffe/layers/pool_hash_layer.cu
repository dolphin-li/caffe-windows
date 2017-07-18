#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pool_hash_layer.hpp"
#include "caffe\util\HashData.h"

namespace caffe {

	template<class Dtype>
	__device__ __forceinline__ Dtype cubic(Dtype a)
	{
		return a*a*a;
	}

	template<class Dtype>
	__global__ void batch_forward_gpu_max_kernel(
		const Dtype *bottom_hash_0, const PACKED_POSITION *bottom_posTag_0, const unsigned char *bottom_offset_0, 
		const Dtype* bottom_m_bar_ptr, const Dtype* bottom_m_sum_ptr, 
		const Dtype* bottom_r_bar_ptr, const Dtype* bottom_r_sum_ptr,
		Dtype *top_hash_0, const PACKED_POSITION *top_posTag_0,
		const Dtype* top_m_bar_ptr, const Dtype* top_m_sum_ptr, int *mask_0,
		const VolumeIndexType* volIdx_ptr, const int* validPos,
		const int channels, const int bt_dense_res, const int3 stride,
		const int total_top_defNUm, const int nThreads, const int channelPerGroup)
	{
		CUDA_KERNEL_LOOP(threadId, nThreads)
		{
			const int group = threadId / total_top_defNUm;
			const int valid_v = threadId - group * total_top_defNUm;
			const VolumeIndexType volIdx = volIdx_ptr[valid_v];
			const int firstChannel = group * channelPerGroup;
			const int lastChannel = min(firstChannel + channelPerGroup, channels);
			const int top_m_sum = top_m_sum_ptr[volIdx];
			const int top_m = cubic(int(top_m_bar_ptr[volIdx]));
			const int bottom_m_bar = bottom_m_bar_ptr[volIdx];
			const int bottom_m_sum = bottom_m_sum_ptr[volIdx];
			const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
			const int bottom_r_bar = bottom_r_bar_ptr[volIdx];
			const int v = validPos[valid_v];
			const Dtype* bottom_hash = bottom_hash_0 + bottom_m_sum * channels;
			Dtype* top_hash = top_hash_0 + top_m_sum * channels + v;
			int* mask = mask_0 + top_m_sum * channels + v;
			const unsigned char* bottom_offset = bottom_offset_0 + int(bottom_r_sum_ptr[volIdx]) * 3;

			const int3 center = xyz_from_pack_g(top_posTag_0[v + top_m_sum]);

			//init to min
			for (int c = firstChannel; c < lastChannel; c++)
				top_hash[top_m*c] = -FLT_MAX;

			const int min_x = max(0, center.x * stride.x);
			const int min_y = max(0, center.y * stride.y);
			const int min_z = max(0, center.z * stride.z);
			const int x_end = min(min_x + stride.x, bt_dense_res);
			const int y_end = min(min_y + stride.y, bt_dense_res);
			const int z_end = min(min_z + stride.z, bt_dense_res);

			for (int nz = min_z; nz < z_end; ++nz)
				for (int ny = min_y; ny < y_end; ++ny)
					for (int nx = min_x; nx < x_end; ++nx)
					{
						//hash to get hash position
						const int bt_m_idx = NXYZ2I_g(Hash_g(nx, ny, nz, bottom_offset, 
							bottom_m_bar, bottom_r_bar), bottom_m_bar);
						const int3 stored = xyz_from_pack_g(bottom_posTag_0[bt_m_idx + bottom_m_sum]);

						//the bottom hash voxel is undefined
						if (!ishashVoxelDefined_g(stored) || nx != stored.x || ny != stored.y || nz != stored.z)
							continue;
						for (int c = firstChannel; c < lastChannel; c++)
						{
							const Dtype bt_hash_val = bottom_hash[bt_m_idx + bottom_m*c];
							if (top_hash[top_m*c] < bt_hash_val)
							{
								top_hash[top_m*c] = bt_hash_val;
								mask[top_m*c] = bt_m_idx;
							}
						}
					} // end for nx, ny, nz
#if USE_EMPTY_VALID_REGION
			if (top_hash[top_m*firstChannel] == -FLT_MAX)//expanded regions, no parent
			{
				for (int c = firstChannel; c < lastChannel; c++)
					top_hash[top_m*c] = 0.f;
			}
#endif
		} // end for caffe_kernel_loop
	}


	//added for outputing mask to top
	template<class Dtype>
	__global__ void batch_forward_gpu_max_kernel(
		const Dtype *bottom_hash_0, const PACKED_POSITION *bottom_posTag_0, const unsigned char *bottom_offset_0,
		const Dtype* bottom_m_bar_ptr, const Dtype* bottom_m_sum_ptr,
		const Dtype* bottom_r_bar_ptr, const Dtype* bottom_r_sum_ptr,
		Dtype *top_hash_0, const PACKED_POSITION *top_posTag_0,
		const Dtype* top_m_bar_ptr, const Dtype* top_m_sum_ptr, Dtype *top_mask_0,
		const VolumeIndexType* volIdx_ptr, const int* validPos,
		const int channels, const int bt_dense_res, const int3 stride,
		const int total_top_defNUm, const int nThreads, const int channelPerGroup)
	{
		CUDA_KERNEL_LOOP(threadId, nThreads)
		{
			const int group = threadId / total_top_defNUm;
			const int valid_v = threadId - group * total_top_defNUm;
			const VolumeIndexType volIdx = volIdx_ptr[valid_v];
			const int firstChannel = group * channelPerGroup;
			const int lastChannel = min(firstChannel + channelPerGroup, channels);
			const int top_m_sum = top_m_sum_ptr[volIdx];
			const int top_m = cubic(int(top_m_bar_ptr[volIdx]));
			const int bottom_m_bar = bottom_m_bar_ptr[volIdx];
			const int bottom_m_sum = bottom_m_sum_ptr[volIdx];
			const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
			const int bottom_r_bar = bottom_r_bar_ptr[volIdx];
			const int v = validPos[valid_v];
			const Dtype* bottom_hash = bottom_hash_0 + bottom_m_sum * channels;
			Dtype* top_hash = top_hash_0 + top_m_sum * channels + v;
			Dtype* top_mask = top_mask_0 + top_m_sum * channels + v;
			const unsigned char* bottom_offset = bottom_offset_0 + int(bottom_r_sum_ptr[volIdx]) * 3;

			const int3 center = xyz_from_pack_g(top_posTag_0[v + top_m_sum]);

			//init to min
			for (int c = firstChannel; c < lastChannel; c++)
				top_hash[top_m*c] = -FLT_MAX;

			const int min_x = max(0, center.x * stride.x);
			const int min_y = max(0, center.y * stride.y);
			const int min_z = max(0, center.z * stride.z);
			const int x_end = min(min_x + stride.x, bt_dense_res);
			const int y_end = min(min_y + stride.y, bt_dense_res);
			const int z_end = min(min_z + stride.z, bt_dense_res);

			for (int nz = min_z; nz < z_end; ++nz)
				for (int ny = min_y; ny < y_end; ++ny)
					for (int nx = min_x; nx < x_end; ++nx)
					{
						//hash to get hash position
						const int bt_m_idx = NXYZ2I_g(Hash_g(nx, ny, nz, bottom_offset,
							bottom_m_bar, bottom_r_bar), bottom_m_bar);
						const int3 stored = xyz_from_pack_g(bottom_posTag_0[bt_m_idx + bottom_m_sum]);

						//the bottom hash voxel is undefined
						if (!ishashVoxelDefined_g(stored) || nx != stored.x || ny != stored.y || nz != stored.z)
							continue;
						for (int c = firstChannel; c < lastChannel; c++)
						{
							const Dtype bt_hash_val = bottom_hash[bt_m_idx + bottom_m*c];
							if (top_hash[top_m*c] < bt_hash_val)
							{
								top_hash[top_m*c] = bt_hash_val;
								top_mask[top_m*c] = bt_m_idx;
							}
						}
					} // end for nx, ny, nz
#if USE_EMPTY_VALID_REGION
			if (top_hash[top_m*firstChannel] == -FLT_MAX)//expanded regions, no parent
			{
				for (int c = firstChannel; c < lastChannel; c++)
					top_hash[top_m*c] = 0.f;
			}
#endif
		} // end for caffe_kernel_loop
	}


	template<class Dtype>
	__global__ void batch_backward_gpu_max_kernel(
		Dtype *bottom_dif, const Dtype* bottom_m_bar_ptr, const Dtype* bottom_m_sum_ptr,
		const Dtype *top_dif, const Dtype* top_m_bar_ptr, const Dtype* top_m_sum_ptr,
		const int *mask, 
		const int *validPos,
		const VolumeIndexType* volIdx_ptr,
		const int channels, 
		const int top_total_def_num)
	{
		CUDA_KERNEL_LOOP(threadId, top_total_def_num*channels)
		{
			const int c = threadId / top_total_def_num;
			const int valid_v = threadId - top_total_def_num * c;
			const VolumeIndexType volIdx = volIdx_ptr[valid_v];
			const int top_m_bar = top_m_bar_ptr[volIdx];
			const int top_m_sum = top_m_sum_ptr[volIdx];
			const int top_m = top_m_bar*top_m_bar*top_m_bar;
			const int bottom_m_bar = bottom_m_bar_ptr[volIdx];
			const int bottom_m_sum = bottom_m_sum_ptr[volIdx];
			const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
			const int v = validPos[valid_v] + c*top_m + channels*top_m_sum;
			const int bt_m_idx = mask[v];
#if USE_EMPTY_VALID_REGION
			if(bt_m_idx!=-1)
				bottom_dif[bt_m_idx + bottom_m * c + channels*bottom_m_sum] = top_dif[v];
#else
			bottom_dif[bt_m_idx + bottom_m * c + channels*bottom_m_sum] = top_dif[v];
#endif
		}
	}

	//added for outputing mask to top
	template<class Dtype>
	__global__ void batch_backward_gpu_max_kernel(
		Dtype *bottom_dif, const Dtype* bottom_m_bar_ptr, const Dtype* bottom_m_sum_ptr,
		const Dtype *top_dif, const Dtype* top_m_bar_ptr, const Dtype* top_m_sum_ptr,
		const Dtype *top_mask,
		const int *validPos,
		const VolumeIndexType* volIdx_ptr,
		const int channels,
		const int top_total_def_num)
	{
		CUDA_KERNEL_LOOP(threadId, top_total_def_num*channels)
		{
			const int c = threadId / top_total_def_num;
			const int valid_v = threadId - top_total_def_num * c;
			const VolumeIndexType volIdx = volIdx_ptr[valid_v];
			const int top_m_bar = top_m_bar_ptr[volIdx];
			const int top_m_sum = top_m_sum_ptr[volIdx];
			const int top_m = top_m_bar*top_m_bar*top_m_bar;
			const int bottom_m_bar = bottom_m_bar_ptr[volIdx];
			const int bottom_m_sum = bottom_m_sum_ptr[volIdx];
			const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
			const int v = validPos[valid_v] + c*top_m + channels*top_m_sum;
			const Dtype bt_m_idx = top_mask[v];
#if USE_EMPTY_VALID_REGION
			if (bt_m_idx != (Dtype)-1)
				bottom_dif[(int)bt_m_idx + bottom_m * c + channels*bottom_m_sum] = top_dif[v];
#else
			bottom_dif[bt_m_idx + bottom_m * c + channels*bottom_m_sum] = top_dif[v];
#endif
		}
	}

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
		const int stride_x = stride_shape_.cpu_data()[0];
		const int stride_y = stride_shape_.cpu_data()[1];
		const int stride_z = stride_shape_.cpu_data()[2];
		CHECK(stride_x == stride_y || stride_x != stride_z);
		const int bt_dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
		const int total_top_defNUm = bottom[DEFNUM_SUM_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[
			bottom[DEFNUM_SUM_BLOB + HASH_STRUCTURE_SIZE]->count() - 1];
		CHECK_GT(total_top_defNUm, 0);
		const int groups = (channels_ + CHANNEL_GROUP_NUM - 1) / CHANNEL_GROUP_NUM;
		const int nThreads = total_top_defNUm * groups;
		const int channelPerGroup = (channels_ + groups - 1) / groups;

		const Dtype *bt_hash = bottom[HASH_DATA_BLOB]->gpu_data();
		const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();
		const Dtype* bt_m_bar_ptr = bottom[M_BAR_BLOB]->gpu_data();
		const Dtype* bt_m_sum_ptr = bottom[M_SUM_BLOB]->gpu_data();
		const Dtype* bt_r_bar_ptr = bottom[R_BAR_BLOB]->gpu_data();
		const Dtype* bt_r_sum_ptr = bottom[R_SUM_BLOB]->gpu_data();

		Dtype *tp_hash = (Dtype*)top[HASH_DATA_BLOB]->mutable_gpu_data();
		const PACKED_POSITION *tp_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const Dtype *tp_m_bar_ptr = bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const Dtype *tp_m_sum_ptr = bottom[M_SUM_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();

		const VolumeIndexType* volIdx_ptr = (const VolumeIndexType*)bottom[VOLUME_IDX_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const int* validPos = (const int*)bottom[VALID_POS_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();


		const bool use_top_mask = top.size() == HASH_DATA_SIZE + 1;
		int* mask = NULL;  // suppress warnings about uninitalized variables
		Dtype* top_mask = NULL;
		if (use_top_mask) {
			top_mask = top[HASH_DATA_SIZE]->mutable_gpu_data();
			caffe_gpu_set(top[HASH_DATA_SIZE]->count(), Dtype(-1), top_mask);

			batch_forward_gpu_max_kernel << <CAFFE_GET_BLOCKS(nThreads), CAFFE_CUDA_NUM_THREADS >> > (
				bt_hash, bt_posTag, bt_offset, bt_m_bar_ptr, bt_m_sum_ptr, bt_r_bar_ptr, bt_r_sum_ptr,
				tp_hash, tp_posTag, tp_m_bar_ptr, tp_m_sum_ptr, top_mask, volIdx_ptr, validPos,
				channels_, bt_dense_res, make_int3(stride_x, stride_y, stride_z),
				total_top_defNUm, nThreads, channelPerGroup
				);
		}
		else {
			mask = max_idx_.mutable_gpu_data();
			caffe_gpu_set(max_idx_.count(), -1, mask);

			batch_forward_gpu_max_kernel << <CAFFE_GET_BLOCKS(nThreads), CAFFE_CUDA_NUM_THREADS >> > (
				bt_hash, bt_posTag, bt_offset, bt_m_bar_ptr, bt_m_sum_ptr, bt_r_bar_ptr, bt_r_sum_ptr,
				tp_hash, tp_posTag, tp_m_bar_ptr, tp_m_sum_ptr, mask, volIdx_ptr, validPos,
				channels_, bt_dense_res, make_int3(stride_x, stride_y, stride_z),
				total_top_defNUm, nThreads, channelPerGroup
				);
		}
	}

	template <typename Dtype>
	void PoolHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		switch (this->layer_param_.pooling_param().pool()) {
		case PoolingParameter_PoolMethod_MAX:
			Backward_gpu_max(top, propagate_down, bottom);
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
	void PoolHashLayer<Dtype>::Backward_gpu_max(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Dtype *bt_hash_dif = (Dtype*)bottom[HASH_DATA_BLOB]->mutable_gpu_diff();
		const Dtype* bt_m_bar_ptr = bottom[M_BAR_BLOB]->gpu_data();
		const Dtype* bt_m_sum_ptr = bottom[M_SUM_BLOB]->gpu_data();

		const Dtype *tp_hash_dif = top[HASH_DATA_BLOB]->gpu_diff();
		const int *tp_validPos = (const int *)bottom[VALID_POS_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const Dtype* tp_m_bar_ptr = bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const Dtype* tp_m_sum_ptr = bottom[M_SUM_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const VolumeIndexType* tp_volIdx = (const VolumeIndexType*)
			bottom[VOLUME_IDX_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const int tp_totalDefNum = bottom[DEFNUM_SUM_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[
			bottom[DEFNUM_SUM_BLOB + HASH_STRUCTURE_SIZE]->count()-1];
		CHECK_GT(tp_totalDefNum, 0);

		const bool use_top_mask = top.size() == HASH_DATA_SIZE + 1;
		const int* mask = NULL;  // suppress warnings about uninitalized variables
		const Dtype* top_mask = NULL;
		caffe_gpu_set(bottom[HASH_DATA_BLOB]->count(), (Dtype)0, bt_hash_dif);
		if (use_top_mask)
		{
			top_mask = top[HASH_DATA_SIZE]->gpu_data();
			batch_backward_gpu_max_kernel << <CAFFE_GET_BLOCKS(tp_totalDefNum*channels_), CAFFE_CUDA_NUM_THREADS >> > (
				bt_hash_dif, bt_m_bar_ptr, bt_m_sum_ptr, tp_hash_dif, tp_m_bar_ptr, tp_m_sum_ptr,
				top_mask, tp_validPos, tp_volIdx, channels_, tp_totalDefNum
				);
		}
		else
		{
			mask = max_idx_.gpu_data();
			batch_backward_gpu_max_kernel << <CAFFE_GET_BLOCKS(tp_totalDefNum*channels_), CAFFE_CUDA_NUM_THREADS >> > (
				bt_hash_dif, bt_m_bar_ptr, bt_m_sum_ptr, tp_hash_dif, tp_m_bar_ptr, tp_m_sum_ptr,
				mask, tp_validPos, tp_volIdx, channels_, tp_totalDefNum
				);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(PoolHashLayer);
}  // namespace caffe
