#include <algorithm>
#include <vector>

#include "caffe/layers/bn_hash_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe\util\HashData.h"

namespace caffe {

	template <typename Dtype>
	__global__ void hash2temp_kernel(const Dtype *hash, const int* validPos, int m_bar,
		int channels, int def_num, int total_def_num, Dtype *temp)
	{
		const int m = m_bar * m_bar * m_bar;
		CUDA_KERNEL_LOOP(valid_v, def_num)
		{
			const int v = validPos[valid_v];
			const Dtype *hash_ptr = hash + v;
			Dtype *temp_ptr = temp + valid_v;
			for (int c = 0; c < channels; c++)
			{
				*temp_ptr = *hash_ptr;
				hash_ptr += m;
				temp_ptr += total_def_num;
			}
		}
	}

	template <typename Dtype>
	__global__ void temp2hash_kernel(Dtype *hash, const int* validPos, int m_bar,
		int channels, int def_num, int total_def_num, const Dtype *temp)
	{
		const int m = m_bar * m_bar * m_bar;
		CUDA_KERNEL_LOOP(valid_v, def_num)
		{
			const int v = validPos[valid_v];
			Dtype *hash_ptr = hash + v;
			const Dtype *temp_ptr = temp + valid_v;
			for (int c = 0; c < channels; c++)
			{
				*hash_ptr = *temp_ptr;
				hash_ptr += m;
				temp_ptr += total_def_num;
			}
		}
	}

	template <typename Dtype>
	__global__ void substract_mean_kernel(Dtype* temp, int channels, int total_defNum, const Dtype* mean)
	{
		CUDA_KERNEL_LOOP(index, channels*total_defNum)
		{
			const int c = index / total_defNum;
			temp[index] -= mean[c];
		}
	}

	template <typename Dtype>
	__global__ void inv_sqrt_eps_var_kernel(Dtype* temp, int channels, int total_defNum, const Dtype* var, Dtype eps)
	{
		CUDA_KERNEL_LOOP(index, channels*total_defNum)
		{
			const int c = index / total_defNum;
			temp[index] /= sqrt(var[c] + eps);
		}
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::forward_hash2temp_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype *hash = bottom[HASH_DATA_BLOB]->gpu_data();
		const int *validPos = (const int*)bottom[VALID_POS_BLOB]->gpu_data();
		Dtype* temp = temp_.mutable_gpu_data();
		const int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int total_def_num = temp_.shape(1);
		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];

			hash2temp_kernel << <CAFFE_GET_BLOCKS(def_num), CAFFE_CUDA_NUM_THREADS >> > (
				hash, validPos, m_bar, channels_, def_num, total_def_num, temp
				);

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			hash += m * channels_;
			validPos += m;
			temp += def_num;
		}
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::forward_temp2hash_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Dtype *hash = (Dtype*)top[HASH_DATA_BLOB]->mutable_gpu_data();
		const int *validPos = (const int*)bottom[VALID_POS_BLOB]->gpu_data();
		const Dtype* temp = temp_.gpu_data();
		const int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int total_def_num = temp_.shape(1);
		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];

			temp2hash_kernel << <CAFFE_GET_BLOCKS(def_num), CAFFE_CUDA_NUM_THREADS >> > (
				hash, validPos, m_bar, channels_, def_num, total_def_num, temp
				);

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			hash += m * channels_;
			validPos += m;
			temp += def_num;
		}
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		//total num
		const int total_defNum = temp_.shape(1);
		const Dtype mean_div = Dtype(1) / Dtype(total_defNum);
		//const Dtype var_div = Dtype(1) / Dtype(std::max(1, total_defNum - 1));
		const Dtype var_div = mean_div;	//will be bias-corrected when adding to blob[1]

		// prepare temp_ array
		forward_hash2temp_gpu(bottom, top);

		if (use_global_stats_) 
		{
			// use the stored mean/variance estimates.
			const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
				0 : 1 / this->blobs_[2]->cpu_data()[0];
			caffe_gpu_scale(variance_.count(), scale_factor,
				this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
			caffe_gpu_scale(variance_.count(), scale_factor,
				this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
		}
		else
		{
			/********1. compute the mean EX for each channel *************/
			cudaMemset(mean_.mutable_gpu_data(), 0, sizeof(Dtype)*channels_);
			caffe_gpu_gemv(CblasNoTrans, channels_, total_defNum, mean_div, 
				temp_.gpu_data(), mean_multiplier_.gpu_data(), Dtype(0), mean_.mutable_gpu_data());
		}

		/**********************2 substract mean****************/
		substract_mean_kernel << <CAFFE_GET_BLOCKS(channels_*total_defNum), CAFFE_CUDA_NUM_THREADS >> > (
			temp_.mutable_gpu_data(), channels_, total_defNum, mean_.gpu_data()
			);

		/********************3. compute variance using var(X) = E((X-EX)^2)***********************/
		if (!use_global_stats_)
		{
			cudaMemset(variance_.mutable_gpu_data(), 0, sizeof(Dtype)*channels_);
			caffe_gpu_mul(temp_.count(), temp_.gpu_data(), temp_.gpu_data(), temp2_.mutable_gpu_data());
			caffe_gpu_gemv(CblasNoTrans, channels_, total_defNum, var_div,
				temp2_.gpu_data(), mean_multiplier_.gpu_data(), Dtype(0), variance_.mutable_gpu_data());

			// compute and save moving average
			this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
			this->blobs_[2]->mutable_cpu_data()[0] += 1;

			caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
				moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());

			Dtype bias_correction_factor = total_defNum > 1 ? Dtype(total_defNum) / (total_defNum - 1) : 1;
			caffe_gpu_axpby(variance_.count(), bias_correction_factor,
				variance_.gpu_data(), moving_average_fraction_, this->blobs_[1]->mutable_gpu_data());

		}

		/********************4. compute final top (X-mean(X))/(sqrt(var(X)+eps))***********************/
		// normalize variance
		// div by sqrt(var(X)+eps)
		inv_sqrt_eps_var_kernel << <CAFFE_GET_BLOCKS(channels_*total_defNum), CAFFE_CUDA_NUM_THREADS >> > (
			temp_.mutable_gpu_data(), channels_, total_defNum, variance_.gpu_data(), eps_
			);

		forward_temp2hash_gpu(bottom, top);
		
		caffe_copy(bottom[CHANNEL_BLOB]->count(), bottom[CHANNEL_BLOB]->cpu_data(),
			top[CHANNEL_BLOB]->mutable_cpu_data());
		caffe_copy(bottom[DENSE_RES_BLOB]->count(), bottom[DENSE_RES_BLOB]->cpu_data(),
			top[DENSE_RES_BLOB]->mutable_cpu_data());
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* top_diff = top[HASH_DATA_BLOB]->gpu_diff();
		Dtype* bottom_diff = bottom[HASH_DATA_BLOB]->mutable_gpu_diff();
		const int total_defNum = temp_.shape(1);
		const Dtype mean_div = Dtype(1) / Dtype(total_defNum);

		//convert top_dif to tmp
		backward_topDif2temp_gpu(bottom, top);
		if (use_global_stats_)
		{
			// replicate inv_variance to input size
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
				(Dtype)1, inv_sqrt_var_.gpu_data(), mean_multiplier_.gpu_data(), (Dtype)0,
				temp2_.mutable_gpu_data());
			caffe_gpu_mul(temp_.count(), temp_.gpu_data(), temp2_.gpu_data(), temp_.mutable_gpu_data());
			backward_temp2BottomDif_gpu(bottom, top);
			return;
		}


		// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
		//
		// dE(Y)/dX =
		//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
		//     ./ sqrt(var(X) + eps)
		//
		// where \cdot and ./ are hadamard product and elementwise division,
		// respectively, dE/dY is the top diff, and mean/var/sum are all computed
		// along all dimensions except the channels dimension.  In the above
		// equation, the operations allow for expansion (i.e. broadcast) along all
		// dimensions except the channels dimension where required.
		// --------------------------------------------------------
		// If disable_vairance is set, the derivative change to
		// dE(Y)/dX = dE/dY - mean(dE/dY)
		// If disable_mean is set, derivative becomes
		// dE(Y)/dX =
		//   (dE/dY - mean(dE/dY \cdot Y) \cdot Y)
		//     ./ sqrt(var(X) + eps)

		//step1. mean(dE/dY \cdot Y)
		top_2_buf_gpu(bottom, top, temp2_);	//convert Y to temp2_
		//dE/dY \cdot Y; // NOTE: here temp_ is modified
		caffe_gpu_mul(temp_.count(), temp_.gpu_data(), temp2_.gpu_data(), temp_.mutable_gpu_data());
		//mean
		caffe_gpu_gemv(CblasNoTrans, channels_, total_defNum, mean_div,
			temp_.gpu_data(), mean_multiplier_.gpu_data(), Dtype(0), mean_.mutable_gpu_data());

		//step2. mean(dE/dY \cdot Y) \cdot Y
		//reshape mean to input size
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
			(Dtype)1, mean_.gpu_data(), mean_multiplier_.gpu_data(), (Dtype)0,
			temp_.mutable_gpu_data());
		// mean(dE/dY \cdot Y) \cdot Y
		caffe_gpu_mul(temp_.count(), temp_.gpu_data(), temp2_.gpu_data(), temp2_.mutable_gpu_data());

		//step3. dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y
		//convert top_dif to tmp
		backward_topDif2temp_gpu(bottom, top);
		//mean(dE/dY)
		caffe_gpu_gemv(CblasNoTrans, channels_, total_defNum, mean_div,
			temp_.gpu_data(), mean_multiplier_.gpu_data(), Dtype(0), mean_.mutable_gpu_data());

		//dE/dY - mean(dE/dY \cdot Y) \cdot Y
		caffe_gpu_sub(temp_.count(), temp_.gpu_data(), temp2_.gpu_data(), temp_.mutable_gpu_data());

		//-= mean(dE/dY)
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
			total_defNum, 1, (Dtype)-1, mean_.gpu_data(), mean_multiplier_.gpu_data(),
			(Dtype)1, temp_.mutable_gpu_data());

		//step4. (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y) / sqrt(var(X) + eps)
		// replicate inv_variance to input size
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
			(Dtype)1, inv_sqrt_var_.gpu_data(), mean_multiplier_.gpu_data(), (Dtype)0,
			temp2_.mutable_gpu_data());
		caffe_gpu_mul(temp_.count(), temp_.gpu_data(), temp2_.gpu_data(), temp_.mutable_gpu_data());
		backward_temp2BottomDif_gpu(bottom, top);
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::backward_topDif2temp_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype *hash_dif = top[HASH_DATA_BLOB]->gpu_diff();
		const int *validPos = (const int*)bottom[VALID_POS_BLOB]->gpu_data();
		Dtype* temp = temp_.mutable_gpu_data();
		const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
		const int total_def_num = temp_.shape(1);
		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
			const int m = m_bar * m_bar * m_bar;

			hash2temp_kernel << <CAFFE_GET_BLOCKS(def_num), CAFFE_CUDA_NUM_THREADS >> > (
				hash_dif, validPos, m_bar, channels_, def_num, total_def_num, temp
				);

			//to next hash
			hash_dif += m * channels_;
			validPos += m;
			temp += def_num;
		}
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::backward_temp2BottomDif_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Dtype *hash_dif = bottom[HASH_DATA_BLOB]->mutable_gpu_diff();
		const int *validPos = (const int*)bottom[VALID_POS_BLOB]->gpu_data();
		const Dtype* temp = temp_.gpu_data();
		const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
		const int total_def_num = temp_.shape(1);
		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
			const int m = m_bar * m_bar * m_bar;

			temp2hash_kernel << <CAFFE_GET_BLOCKS(def_num), CAFFE_CUDA_NUM_THREADS >> > (
				hash_dif, validPos, m_bar, channels_, def_num, total_def_num, temp
				);

			//to next hash
			hash_dif += m * channels_;
			validPos += m;
			temp += def_num;
		}
	}

	template <typename Dtype>
	void BNHashLayer<Dtype>::top_2_buf_gpu(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top, Blob<Dtype> &buf)
	{
		const Dtype *hash = top[HASH_DATA_BLOB]->gpu_data();
		const int *validPos = (const int*)bottom[VALID_POS_BLOB]->gpu_data();
		Dtype* buf_ptr = buf.mutable_gpu_data();
		const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
		const int total_def_num = buf.shape(1);
		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
			const int m = m_bar * m_bar * m_bar;

			hash2temp_kernel << <CAFFE_GET_BLOCKS(def_num), CAFFE_CUDA_NUM_THREADS >> > (
				hash, validPos, m_bar, channels_, def_num, total_def_num, buf_ptr
				);

			//to next hash
			hash += m * channels_;
			validPos += m;
			buf_ptr += def_num;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(BNHashLayer);
}  // namespace caffe
