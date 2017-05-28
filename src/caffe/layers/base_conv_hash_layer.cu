#include <algorithm>
#include <vector>
#include "caffe/util/math_functions.hpp"

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_hash_layer.hpp"
#include "caffe/util/MyMacro.h"

namespace caffe {
	__global__ void conv_hash2col_gpu_kernel(
		const float* hash_data,
		const unsigned char *offset_data,
		const PACKED_POSITION *position_tags,
		const int* valid_positions,
		const int3 kernel_shape,	//D, H, W
		const int m_bar, 
		const int r_bar, 
		const int channels, 
		const int cols,
		const int dense_res, 
		const int nThreads,
		const int channelPerGroup,
		float* col_buff)
	{
		const int m = m_bar * m_bar * m_bar;
		const int cross_channel_stride = cols * kernel_shape.x * kernel_shape.y * kernel_shape.z;

		CUDA_KERNEL_LOOP(valid_v_threads, nThreads)
		{
			const int group = valid_v_threads / cols;
			const int valid_v = valid_v_threads - group * cols;
			const int firstChannel = group * channelPerGroup;
			const int nChannels = min(firstChannel + channelPerGroup, channels) - firstChannel;
			const int v = valid_positions[valid_v];
			//if (!ishashVoxelDefined_g(position_tags[v]))
			//	printf("error: valid given is non-valid: %d->%d\n", valid_v, v);

			const int3 center = xyz_from_pack_g(position_tags[v]);	//get the voxel position mapped to this hash
			const int min_x = center.x - (kernel_shape.x >> 1);
			const int min_y = center.y - (kernel_shape.y >> 1);
			const int min_z = center.z - (kernel_shape.z >> 1);

			float *cur_row = col_buff + valid_v + firstChannel * cross_channel_stride;
			const float* hash_ptr = hash_data + firstChannel * m;
			for (int nz = min_z; nz < min_z + kernel_shape.z; nz++)
			for (int ny = min_y; ny < min_y + kernel_shape.y; ny++)
				for (int nx = min_x; nx < min_x + kernel_shape.x; nx++, cur_row += cols)
				{
					for (int c = 0; c < nChannels; c++)
						cur_row[c * cross_channel_stride] = 0;
					if (nx < 0 || ny < 0 || nz < 0 || nx >= dense_res || ny >= dense_res || nz >= dense_res)
						continue;

					//hash to get hash position
					const int3 mxyz = Hash_g(nx, ny, nz, offset_data, m_bar, r_bar);
					const int m_idx = NXYZ2I_g(mxyz.x, mxyz.y, mxyz.z, m_bar);
					const int3 stored = xyz_from_pack_g(position_tags[m_idx]);
					if (!ishashVoxelDefined_g(stored) || nx != stored.x || ny != stored.y || nz != stored.z)
						continue;

					//fill the value at cur_row and corresponding channels
					for (int c = 0; c < nChannels; c++)
						cur_row[c * cross_channel_stride] = hash_ptr[c*m + m_idx];
				} // end for xyz
		} // end for cuda_kernel_loop
	}

	int conv_hash2col_gpu(const float* hash_data, 
		const unsigned char *offset_data, 
		const PACKED_POSITION *position_tags,
		const int* valid_positions,
		const int kernel_shape[3],	//D, H, W
		int m_bar, int r_bar, int channels, int defined_num,
		int dense_res, float* col_buff)
	{
		const int groups = (channels + CHANNEL_GROUP_NUM - 1) / CHANNEL_GROUP_NUM;
		const int nThreads = defined_num* groups;
		const int channelPerGroup = (channels + groups - 1) / groups;

		conv_hash2col_gpu_kernel << <CAFFE_GET_BLOCKS(nThreads), CAFFE_CUDA_NUM_THREADS >> > (
			hash_data, offset_data, position_tags, valid_positions,
			make_int3(kernel_shape[0], kernel_shape[1], kernel_shape[2]),
			m_bar, r_bar, channels, defined_num, dense_res, nThreads, channelPerGroup, col_buff
			);
		return 1;
	}

	__global__ void conv_col2hash_gpu_kernel(
		const int* valid_positions, float *out_hash_data,
		const int m, const int out_channels_mul_defined_num, 
		const int defined_num, const float* col_buff)
	{
		CUDA_KERNEL_LOOP(valid_v_channels, out_channels_mul_defined_num)
		{
			const int channel = valid_v_channels / defined_num;
			const int valid_v = valid_v_channels - channel * defined_num;
			const int v = valid_positions[valid_v];
			out_hash_data[channel*m + v] = col_buff[channel*defined_num + valid_v];
		}
	}

	int conv_col2hash_gpu(const PACKED_POSITION *pos_tags, 
		const int* valid_positions, float *out_hash_data,
		int m_bar, int out_channels, int defined_num, const float* col_buff)
	{
		conv_col2hash_gpu_kernel << <CAFFE_GET_BLOCKS(defined_num * out_channels), CAFFE_CUDA_NUM_THREADS >> > (
			valid_positions, out_hash_data, m_bar*m_bar*m_bar, out_channels*defined_num, defined_num, col_buff
			);
		return 1;
	}

	__global__ void top_hash2col_gpu_kernel(const float *hash_data, const PACKED_POSITION *pos_tags,
		const int* valid_positions, int m, int out_channels_mul_defNum, int defined_num, float* col_buff)
	{
		CUDA_KERNEL_LOOP(valid_v_channels, out_channels_mul_defNum)
		{
			const int channel = valid_v_channels / defined_num;
			const int valid_v = valid_v_channels - channel * defined_num;
			const int v = valid_positions[valid_v];
			col_buff[defined_num*channel + valid_v] = hash_data[m*channel + v];
		}
	}

	//used for BP, convert the top (dif) to col
	int top_hash2col_gpu(const float *hash_data, const PACKED_POSITION *pos_tags,
		const int* valid_positions, int m_bar, int out_channels, int defined_num, float* col_buff)
	{
		top_hash2col_gpu_kernel << <CAFFE_GET_BLOCKS(defined_num*out_channels), CAFFE_CUDA_NUM_THREADS >> > (
			hash_data, pos_tags, valid_positions, m_bar*m_bar*m_bar, defined_num*out_channels, defined_num, col_buff
			);
		return 1;
	}

	__global__ void bottom_col2hash_gpu_kernel(
		float* hash_data,
		const unsigned char *offset_data,
		const PACKED_POSITION *position_tags,
		const int* valid_positions, 
		const int3 kernel_shape,	//D, H, W
		const int m_bar, 
		const int r_bar, 
		const int channels, 
		const int defined_num,
		const int dense_res, 
		const int nThreads, 
		const int channelPerGroup,
		const float* col_buff)
	{
		//col is reception field: input_channels * KD * KH * KW; row: spatial domain
		const int m = m_bar * m_bar * m_bar;
		const int cross_channel_stride = defined_num * kernel_shape.x * kernel_shape.y * kernel_shape.z;

		CUDA_KERNEL_LOOP(valid_v_threads, nThreads)
		{
			const int group = valid_v_threads / defined_num;
			const int valid_v = valid_v_threads - group * defined_num;
			const int firstChannel = group * channelPerGroup;
			const int nChannels = min(firstChannel + channelPerGroup, channels) - firstChannel;
			const int v = valid_positions[valid_v];

			//get the real voxel position from the position tag
			const int3 center = xyz_from_pack_g(position_tags[v]);		
			const int min_x = center.x - (kernel_shape.x >> 1);
			const int min_y = center.y - (kernel_shape.y >> 1);
			const int min_z = center.z - (kernel_shape.z >> 1);
			const float *cur_row = col_buff + valid_v + firstChannel * cross_channel_stride;
			float* hash_ptr = hash_data + firstChannel * m;
			for (int nz = min_z; nz < min_z + kernel_shape.z; nz++)
			for (int ny = min_y; ny < min_y + kernel_shape.y; ny++)
				for (int nx = min_x; nx < min_x + kernel_shape.x; nx++, cur_row += defined_num)
				{
					if (nx < 0 || ny < 0 || nz < 0 || nx >= dense_res || ny >= dense_res || nz >= dense_res)
						continue;

					//hash to get hash position
					const int3 mxyz = Hash_g(nx, ny, nz, offset_data, m_bar, r_bar);
					const int m_idx = NXYZ2I_g(mxyz.x, mxyz.y, mxyz.z, m_bar);
					const int3 stored = xyz_from_pack_g(position_tags[m_idx]);

					if (!ishashVoxelDefined_g(stored) || nx != stored.x || ny != stored.y || nz != stored.z)
						continue;

					//accumulate the value at cur_row and corresponding channels to the hash data
					float *hash_ptr_fill = hash_ptr + m_idx;
					const float *fill_row_ptr = cur_row;
					for (int c = 0; c < nChannels; c++)
					{
						// TODO: how to parallel the operation instead of using automics?
						//*hash_ptr += *fill_row_ptr;
						atomicAdd(hash_ptr_fill, *fill_row_ptr);	//accumulate the value to the hash
						fill_row_ptr += cross_channel_stride;
						hash_ptr_fill += m;
					}
				} // end for nx, ny, nz
		} // end for caffe kernel loop
	}

	int bottom_col2hash_gpu(float* hash_data, const unsigned char *offset_data, const PACKED_POSITION *position_tags,
		const int* valid_positions, const int kernel_shape[3],	//D, H, W
		int m_bar, int r_bar, int channels, int defined_num,
		int dense_res, const float* col_buff)
	{
		const int groups = (channels + CHANNEL_GROUP_NUM - 1) / CHANNEL_GROUP_NUM;
		const int nThreads = defined_num* groups;
		const int channelPerGroup = (channels + groups - 1) / groups;
		bottom_col2hash_gpu_kernel << <CAFFE_GET_BLOCKS(nThreads), CAFFE_CUDA_NUM_THREADS >> > (
			hash_data, offset_data, position_tags, valid_positions, 
			make_int3(kernel_shape[0], kernel_shape[1], kernel_shape[2]), m_bar,
			r_bar, channels, defined_num, dense_res, nThreads, channelPerGroup, col_buff
			);
		return 1;
	}
}  // namespace caffe
