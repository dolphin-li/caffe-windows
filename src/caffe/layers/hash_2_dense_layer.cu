#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hash_2_dense_layer.hpp"
#include "caffe\util\HashData.h"

namespace caffe {
	template<class Dtype>
	__global__ void hash_2_dense_kernel(const Dtype *hash_data,
		const PACKED_POSITION *position_tags, 
		const unsigned char *m_offset_data,
		int m_bar, int r_bar, int channels,
		Dtype *dense_data, int res)
	{
		const int res3 = res*res*res;
		const int m = m_bar * m_bar * m_bar;
		CUDA_KERNEL_LOOP(i, m)
		{
			if (!ishashVoxelDefined_g(position_tags[i]))
				continue;
			int x, y, z;
			xyz_from_pack_g(position_tags[i], x, y, z);
			int ni = NXYZ2I_g(x, y, z, res, res*res);
			Dtype *cur_dense_ptr = dense_data + ni;
			const Dtype *cur_hash_ptr = hash_data + i;
			for (int c = 0; c < channels; c++)
			{
				*cur_dense_ptr = *cur_hash_ptr;
				cur_dense_ptr += res3;
				cur_hash_ptr += m;
			}
		}
	}

	template <typename Dtype>
	void Hash2DenseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype *bt_hash = bottom[HASH_DATA_BLOB]->gpu_data();
		const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();
		const int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
		const int channel_spatial_dim = channels_ * dense_res * dense_res * dense_res;

		Dtype *dense_buf = (Dtype*)top[0]->mutable_gpu_data();
		cudaMemset(dense_buf, 0, sizeof(Dtype)*channel_spatial_dim*batch_num);
		for (int i = 0; i < batch_num; ++i)
		{
			const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int bt_m = bt_m_bar * bt_m_bar * bt_m_bar;
			const int bt_r = bt_r_bar * bt_r_bar * bt_r_bar;

			hash_2_dense_kernel << <CAFFE_GET_BLOCKS(bt_m), CAFFE_CUDA_NUM_THREADS >> >(
				bt_hash, bt_posTag, bt_offset, bt_m_bar, bt_r_bar, channels_, dense_buf, dense_res);

			//to next hash
			bt_hash += bt_m * channels_;
			bt_offset += bt_r * 3;
			bt_posTag += bt_m;
			//to next dense
			dense_buf += channel_spatial_dim;
		}
	}

	template <typename Dtype>
	__global__ void dense_2_hash_kernel(Dtype *hash_data, 
		const PACKED_POSITION *position_tags, 
		const unsigned char *m_offset_data,
		int m_bar, int r_bar, int channels,
		const Dtype *dense_data, int res)
	{
		const int res3 = res*res*res;
		const int m = m_bar * m_bar * m_bar;

		CUDA_KERNEL_LOOP(i, m)
		{
			if (!ishashVoxelDefined_g(position_tags[i]))
				continue;
			int x, y, z;
			xyz_from_pack_g(position_tags[i], x, y, z);
			const int ni = NXYZ2I_g(x, y, z, res, res*res);
			const Dtype *cur_dense_ptr = dense_data + ni;
			Dtype *cur_hash_ptr = hash_data + i;
			for (int c = 0; c < channels; c++)
			{
				*cur_hash_ptr = *cur_dense_ptr;
				cur_dense_ptr += res3;
				cur_hash_ptr += m;
			}
		}
	}

	template <typename Dtype>
	void Hash2DenseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Dtype *bt_hash = (Dtype*)bottom[HASH_DATA_BLOB]->mutable_gpu_diff();
		const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();
		const int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
		const int channel_spatial_dim = channels_ * dense_res * dense_res * dense_res;

		const Dtype *dense_buf = top[0]->gpu_diff();
		for (int i = 0; i < batch_num; ++i)
		{
			const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int bt_m = bt_m_bar * bt_m_bar * bt_m_bar;
			const int bt_r = bt_r_bar * bt_r_bar * bt_r_bar;

			cudaMemset(bt_hash, 0, sizeof(Dtype)*bt_m * channels_);
			dense_2_hash_kernel << <CAFFE_GET_BLOCKS(bt_m), CAFFE_CUDA_NUM_THREADS >> >(
				bt_hash, bt_posTag, bt_offset, bt_m_bar, bt_r_bar, channels_, dense_buf, dense_res);

			//to next hash
			bt_hash += bt_m * channels_;
			bt_offset += bt_r * 3;
			bt_posTag += bt_m;
			//to next dense
			dense_buf += channel_spatial_dim;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(Hash2DenseLayer);
}  // namespace caffe
