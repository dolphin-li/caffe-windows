#include <algorithm>
#include <vector>
#include "caffe/util/math_functions.hpp"

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_hash_layer.hpp"
#include "caffe/util/MyMacro.h"

#define DUMP_2_TXT 0

namespace caffe {

	__device__ __host__ inline void xyz_from_pack_g(PACKED_POSITION packed_val, int &x, int &y, int &z)
	{
		const int packed_xy = *(const int*)(packed_val._val);
		x = (packed_xy & 0x0000FFFF);
		y = (packed_xy & 0xFFFF0000) >> 16;

		const int packed_z = *(const int*)(packed_val._val + 2);
		z = (packed_z & 0xFFFF0000) >> 16;
	}

	__device__ __host__ inline bool ishashVoxelDefined_g(PACKED_POSITION pos_tag)
	{
		int x, y, z;
		xyz_from_pack_g(pos_tag, x, y, z);
		return !(x == INVALID_POSTAG || y == INVALID_POSTAG || z == INVALID_POSTAG);
	}

	__device__ __host__ inline int NXYZ2I_g(int nx, int ny, int nz, int n, int n2)
	{
		return nz*n2 + ny*n + nx;
	};

	__device__ __host__ inline void Hash_g(int nx, int ny, int nz, int& mx, int& my, int& mz,
		const unsigned char *offset_data, int m_bar, int r_bar, int r2)
	{
		int rx = nx%r_bar;
		int ry = ny%r_bar;
		int rz = nz%r_bar;

		const unsigned char *offset = offset_data + NXYZ2I_g(rx, ry, rz, r_bar, r2) * 3;

		mx = (nx + offset[0]) % m_bar;
		my = (ny + offset[1]) % m_bar;
		mz = (nz + offset[2]) % m_bar;
	}

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
		float* col_buff)
	{
		const int m = m_bar * m_bar * m_bar;
		const int kernel_dim = kernel_shape.x * kernel_shape.y * kernel_shape.z;
		const int r2 = r_bar * r_bar;
		const int m2 = m_bar * m_bar;
		const int cross_channel_stride = cols * kernel_dim;

		CUDA_KERNEL_LOOP(valid_v, cols)
		{
			const int v = valid_positions[valid_v];
			if (!ishashVoxelDefined_g(position_tags[v]))
				printf("error: valid given is non-valid: %d->%d\n", valid_v, v);

			int cx, cy, cz;
			xyz_from_pack_g(position_tags[v], cx, cy, cz);	//get the voxel position mapped to this hash
			const int min_x = cx - (kernel_shape.x >> 1);
			const int min_y = cy - (kernel_shape.y >> 1);
			const int min_z = cz - (kernel_shape.z >> 1);

			float *cur_row = col_buff + valid_v;
			for (int nz = min_z; nz < min_z + kernel_shape.z; nz++)
			for (int ny = min_y; ny < min_y + kernel_shape.y; ny++)
				for (int nx = min_x; nx < min_x + kernel_shape.x; nx++)
				{
					if (nx < 0 || ny < 0 || nz < 0 || nx >= dense_res || ny >= dense_res || nz >= dense_res)
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}
					//hash to get hash position
					int mx, my, mz;
					Hash_g(nx, ny, nz, mx, my, mz, offset_data, m_bar, r_bar, r2);
					const int m_idx = NXYZ2I_g(mx, my, mz, m_bar, m2);

					if (!ishashVoxelDefined_g(position_tags[m_idx]))	//the hash voxel is undefined
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}

					int stored_x, stored_y, stored_z;
					xyz_from_pack_g(position_tags[m_idx], stored_x, stored_y, stored_z);
					if (nx != stored_x || ny != stored_y || nz != stored_z)	//the neighbor is an undefined voxel
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}

					//fill the value at cur_row and corresponding channels
					const float *hash_ptr = hash_data + m_idx;
					float *fill_row_ptr = cur_row;
					for (int c = 0; c < channels; c++)
					{
						*fill_row_ptr = *hash_ptr;
						fill_row_ptr += cross_channel_stride;
						hash_ptr += m;
					}
					cur_row += cols;
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
		//col is reception field: input_channels * KD * KH * KW; row: spatial domain
		const int kernel_dim = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
		const int rows = kernel_dim * channels;
		const int cols = defined_num;
		
		//init vals to zero
		cudaMemset(col_buff, 0, sizeof(float)*cols * rows);

		conv_hash2col_gpu_kernel << <CAFFE_GET_BLOCKS(cols), CAFFE_CUDA_NUM_THREADS >> > (
			hash_data, offset_data, position_tags, valid_positions,
			make_int3(kernel_shape[0], kernel_shape[1], kernel_shape[2]),
			m_bar, r_bar, channels, cols, dense_res, col_buff
			);
	
		CHECK_EQ(cudaDeviceSynchronize(), CUDA_SUCCESS);

#if 1
		static int test_a = 0;
		if (test_a++ == 0)
		{
			std::vector<float> tmp_col(rows*cols);
			caffe_copy(rows*cols, col_buff, tmp_col.data());
			FILE* pFile = fopen("col_buf_gpu.txt", "w");
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
					fprintf(pFile, "%f ", tmp_col[r*cols + c]);
				fprintf(pFile, "\n");
			}
			fclose(pFile);

			pFile = fopen("col_buf_valid_pos_gpu.txt", "w");
			std::vector<int> tmp_valid(defined_num);
			caffe_copy(defined_num, valid_positions, tmp_valid.data());
			for (int i = 0; i < defined_num; i++)
				fprintf(pFile, "%d\n", tmp_valid[i]);
			fclose(pFile);
		}
#endif

		return 1;
	}

	__global__ void conv_col2hash_gpu_kernel(const PACKED_POSITION *pos_tags, 
		const int* valid_positions, float *out_hash_data,
		const int m_bar, const int out_channels, const int defined_num, const float* col_buff)
	{
		const int m = m_bar * m_bar * m_bar;
		CUDA_KERNEL_LOOP(valid_v, defined_num)
		{
			const int v = valid_positions[valid_v];
			//if the hash voxel is undefined, skip
			if (!ishashVoxelDefined_g(pos_tags[v]))
				printf("error: valid given is non-valid: %d->%d\n", valid_v, v);

			float *cur_out_ptr = out_hash_data + v;
			const float *cur_row_ptr = col_buff + valid_v;
			for (int c = 0; c < out_channels; c++)
			{
				*cur_out_ptr = *cur_row_ptr;
				cur_out_ptr += m;
				cur_row_ptr += defined_num;
			}
		}
	}

	int conv_col2hash_gpu(const PACKED_POSITION *pos_tags, 
		const int* valid_positions, float *out_hash_data,
		int m_bar, int out_channels, int defined_num, const float* col_buff)
	{
		//col is reception field: input_channels * KD * KH * KW; row: spatial domain
		const int m = m_bar * m_bar * m_bar;
		//to be safe, init out to zero
		cudaMemset(out_hash_data, 0, sizeof(float)*m*out_channels);
		conv_col2hash_gpu_kernel << <CAFFE_GET_BLOCKS(defined_num), CAFFE_CUDA_NUM_THREADS >> > (
			pos_tags, valid_positions, out_hash_data, m_bar, out_channels, defined_num, col_buff
			);
		return 1;
	}

}  // namespace caffe
