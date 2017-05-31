#include <vector>

#include "caffe/layers/conv_hash_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ConvHashLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) 
	{
		const float *batch_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->gpu_data();
		const unsigned char*batch_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *batch_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();
		const int* batch_validPos_ptr = (const int*)bottom[VALID_POS_BLOB]->gpu_data();
		float *top_batch_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_gpu_data();
		const int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];

		//forward channel and dense res
		top[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)num_output_;
		top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];

		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
			float *out_buf = (float*)out_col_buffer_.mutable_gpu_data();

			// LDP: why empty volume??? bugs???
			// current_row_ == 4096, batch_idx=7
			if (defNum == 0)
				continue;

			forward_gpu_gemm(batch_hash_ptr, batch_offset_ptr, batch_posTag_ptr, batch_validPos_ptr, m_bar, r_bar,
				channels_, num_output_, defNum, dense_res, out_buf);

			if (this->bias_term_) 
			{
				const float* bias = (const float*)this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(out_buf, bias, defNum);
			}

			//convert out col buf to top
			conv_col2hash_gpu(batch_posTag_ptr, batch_validPos_ptr, 
				top_batch_hash_ptr, m_bar, num_output_, defNum, out_buf);

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			const int r = r_bar * r_bar * r_bar;
			batch_hash_ptr += m * channels_;
			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
			batch_validPos_ptr += defNum;
			top_batch_hash_ptr += m * num_output_;
		}
	}

	template <typename Dtype>
	void ConvHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		//memset to zero, as the weight_diff will be accumulated within the batch
		caffe_gpu_set(this->blobs_[0]->count(), (Dtype)0, weight_diff);
		Dtype* bias_diff = NULL;
		if (this->bias_term_)
		{
			caffe_gpu_set(this->blobs_[1]->count(), (Dtype)0, this->blobs_[1]->mutable_gpu_diff());
			bias_diff = this->blobs_[1]->mutable_gpu_diff();
		}

		const float *top_hash_dif = (const float*)top[HASH_DATA_BLOB]->gpu_diff();
		const float *bt_hash = (const float*)bottom[HASH_DATA_BLOB]->gpu_data();
		float *bt_hash_dif = (float*)bottom[HASH_DATA_BLOB]->mutable_gpu_diff();
		cudaMemset(bt_hash_dif, 0, bottom[HASH_DATA_BLOB]->count()*sizeof(float));

		const unsigned char* offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->gpu_data();
		const PACKED_POSITION *posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->gpu_data();
		const int* validPos_ptr = (const int*)bottom[VALID_POS_BLOB]->gpu_data();

		int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
		for (int i = 0; i < batch_num; ++i)
		{
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

			if (defNum == 0)
				continue;

			//convert top dif to out_col_buf
			top_hash2col_gpu(top_hash_dif, posTag_ptr, validPos_ptr, m_bar,
				num_output_, defNum, (float*)out_col_buffer_.mutable_gpu_data());

			//convert bottom data to col_buf
			conv_hash2col_gpu(bt_hash, offset_ptr, posTag_ptr, validPos_ptr, kernel_shape_.cpu_data(),
				m_bar, r_bar, channels_, defNum, dense_res, (float*)col_buffer_.mutable_gpu_data());

			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1])
				this->backward_gpu_bias((float*)bias_diff, (const float*)out_col_buffer_.gpu_data(), defNum);

			if (this->param_propagate_down_[0])// || propagate_down[i]) 
			{
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					this->weight_gpu_gemm((const float*)col_buffer_.gpu_data(), (const float*)out_col_buffer_.gpu_data(),
						(float*)weight_diff, channels_, num_output_, defNum);
				}
				// gradient w.r.t. bottom data, if necessary.
				//if (propagate_down[i]) 
				{
					//NOTE: now col_buf is updated to bottom_dif in backward_cpu_gemm
					this->backward_gpu_gemm((const float*)out_col_buffer_.gpu_data(), channels_,
						num_output_, defNum, (float*)col_buffer_.mutable_gpu_data());
					//convert the col_buf to bottom dif
					bottom_col2hash_gpu(bt_hash_dif, offset_ptr, posTag_ptr, validPos_ptr, kernel_shape_.cpu_data(),
						m_bar, r_bar, channels_, defNum, dense_res, (const float*)col_buffer_.gpu_data());
				}
			}

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			const int r = r_bar * r_bar * r_bar;

			bt_hash += m * channels_;
			bt_hash_dif += m * channels_;
			offset_ptr += r * 3;
			posTag_ptr += m;
			validPos_ptr += defNum;

			top_hash_dif += m * num_output_;
		}

#if 0//for debug
		writeDenseKernelDif_2_HF5("kernel_dif_gpu.hf5");
		writeBiasDif_2_HF5("bias_dif_gpu.hf5");
#endif
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvHashLayer);
}  // namespace caffe
