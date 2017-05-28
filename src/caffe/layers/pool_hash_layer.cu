#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pool_hash_layer.hpp"
#include "caffe\util\HashData.h"

namespace caffe {

	__global__ void forward_gpu_max_kernel(
		const float *bottom_hash, 
		const unsigned char *bottom_offset,
		const PACKED_POSITION *bottom_posTag, 
		const int bottom_m_bar, 
		const int bottom_r_bar,
		float *top_hash, 
		const unsigned char *top_offset,
		const PACKED_POSITION *top_posTag, 
		const int top_m_bar, 
		const int top_r_bar,
		int *mask, 
		const int channels, 
		const int bt_dense_res, 
		const int3 stride,
		const int nThreads,
		const int channelPerGroup)
	{
		const int top_m = top_m_bar * top_m_bar * top_m_bar;
		const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
		CUDA_KERNEL_LOOP(v_threads, nThreads)
		{
			const int group = v_threads / top_m;
			const int v = v_threads - group * top_m;
			const int firstChannel = group * channelPerGroup;
			const int nChannels = min(firstChannel + channelPerGroup, channels) - firstChannel;
			const int3 center = xyz_from_pack_g(top_posTag[v]);
			//if the hash voxel is undefined, skip
			if (!ishashVoxelDefined_g(center))
				continue;

			//init to min
			for (int c = 0; c < nChannels; c++)
				top_hash[top_m*(c + firstChannel) + v] = -FLT_MAX;

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
					const int3 bt_xyz = Hash_g(nx, ny, nz, bottom_offset, bottom_m_bar, bottom_r_bar);
					const int bt_m_idx = NXYZ2I_g(bt_xyz.x, bt_xyz.y, bt_xyz.z, bottom_m_bar);
					const int3 stored = xyz_from_pack_g(bottom_posTag[bt_m_idx]);

					//the bottom hash voxel is undefined
					if (!ishashVoxelDefined_g(stored) || nx != stored.x || ny != stored.y || nz != stored.z)
						continue;
					for (int c = 0; c < nChannels; c++)
					{
						const float bt_hash_val = bottom_hash[bt_m_idx + bottom_m*(c + firstChannel)];
						if (top_hash[top_m*(c + firstChannel) + v] < bt_hash_val)
						{
							top_hash[top_m*(c + firstChannel) + v] = bt_hash_val;
							mask[top_m*(c + firstChannel) + v] = bt_m_idx;
						}
					}
				} // end for nx, ny, nz
		} // end for caffe_kernel_loop
	}

	__global__ void backward_gpu_max_kernel(float *bottom_dif, int bottom_m_bar,
		const float *top_dif, const PACKED_POSITION *top_posTag, int top_m_bar,
		const int *mask, int channels)
	{
		const int top_m = top_m_bar * top_m_bar * top_m_bar;
		const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
		CUDA_KERNEL_LOOP(v_channels, top_m*channels)
		{
			const int c = v_channels / top_m;
			const int v = v_channels - c * top_m;
			//if the hash voxel is undefined, skip
			if (!ishashVoxelDefined_g(top_posTag[v]))
				continue;
			const int bt_m_idx = mask[v + c*top_m];
			bottom_dif[bt_m_idx + bottom_m * c] = top_dif[v + c*top_m];
		}
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
		CHECK(stride_x == stride_y || stride_x != stride_z);

		const int groups = (channels + CHANNEL_GROUP_NUM - 1) / CHANNEL_GROUP_NUM;
		const int nThreads = top_m* groups;
		const int channelPerGroup = (channels + groups - 1) / groups;
		//max pool kernel
		forward_gpu_max_kernel << <CAFFE_GET_BLOCKS(nThreads), CAFFE_CUDA_NUM_THREADS >> > (
			bottom_hash, bottom_offset, bottom_posTag, bottom_m_bar, bottom_r_bar,
			top_hash, top_offset, top_posTag, top_m_bar, top_r_bar,
			mask, channels, bt_dense_res, make_int3(stride_x, stride_y, stride_z), 
			nThreads, channelPerGroup
			);	
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
			const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int tp_m_bar = (int)bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];
			const int tp_r_bar = (int)bottom[R_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];

			forward_gpu_max(bt_hash, bt_offset, bt_posTag, bt_m_bar, bt_r_bar,
				tp_hash, tp_offset, tp_posTag, tp_m_bar, tp_r_bar, mask, channels_, bt_dense_res);

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
		float *bt_hash_dif = (float*)bottom[HASH_DATA_BLOB]->mutable_gpu_diff();
		const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();

		const float *tp_hash_dif = (const float*)top[HASH_DATA_BLOB]->gpu_diff();
		const unsigned char*tp_offset = (const unsigned char *)bottom[OFFSET_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();
		const PACKED_POSITION *tp_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB + HASH_STRUCTURE_SIZE]->gpu_data();

		const int *mask = max_idx_.gpu_data();

		int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
		const int bt_dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
		const int tp_dense_res = (int)top[DENSE_RES_BLOB]->cpu_data()[0];

		for (int i = 0; i < batch_num; ++i)
		{
			const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int tp_m_bar = (int)bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];
			const int tp_r_bar = (int)bottom[R_BAR_BLOB + HASH_STRUCTURE_SIZE]->cpu_data()[i];

			backward_gpu_max(bt_hash_dif, bt_m_bar, tp_hash_dif, tp_posTag, tp_m_bar, mask, channels_);

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
	void PoolHashLayer<Dtype>::backward_gpu_max(float *bottom_dif, int bottom_m_bar,
		const float *top_dif, const PACKED_POSITION *top_posTag, int top_m_bar,
		const int *mask, int channels)
	{
		const int top_m = top_m_bar * top_m_bar * top_m_bar;
		const int bottom_m = bottom_m_bar * bottom_m_bar * bottom_m_bar;
		
		//init dif to zero
		caffe_gpu_set(bottom_m*channels, (float)0, bottom_dif);
		backward_gpu_max_kernel << <CAFFE_GET_BLOCKS(top_m*channels), CAFFE_CUDA_NUM_THREADS >> > (
			bottom_dif, bottom_m_bar, top_dif, top_posTag, top_m_bar, mask, channels
			);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(PoolHashLayer);
}  // namespace caffe
