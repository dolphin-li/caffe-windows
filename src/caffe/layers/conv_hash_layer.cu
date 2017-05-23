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
		int batch_num = bottom[M_BAR_BLOB]->shape(0);
		const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
		for (int i = 0; i < batch_num; ++i)
		{
			const float* bottom_data = batch_hash_ptr;
			const unsigned char*offset_data = batch_offset_ptr;
			const PACKED_POSITION *pos_tags = batch_posTag_ptr;
			const int* valid_positions = batch_validPos_ptr;
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

			float *top_data = top_batch_hash_ptr;
			float *out_buf = (float*)out_col_buffer_.mutable_gpu_data();

			forward_gpu_gemm(bottom_data, offset_data, pos_tags, valid_positions, m_bar, r_bar,
				channels_, num_output_, defNum, dense_res, out_buf);

			if (this->bias_term_) {
				const float* bias = (const float*)this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(out_buf, bias, defNum);
			}

			//convert out col buf to top
			conv_col2hash_gpu(pos_tags, valid_positions, top_data, m_bar, num_output_, defNum, out_buf);

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			const int r = r_bar * r_bar * r_bar;

			batch_hash_ptr += m * channels_;
			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
			batch_validPos_ptr += m;

			top_batch_hash_ptr += m * num_output_;
		}
	}

	template <typename Dtype>
	void ConvHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		//const Dtype* weight = this->blobs_[0]->gpu_data();
		//Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		//for (int i = 0; i < top.size(); ++i) {
		//	const Dtype* top_diff = top[i]->gpu_diff();
		//	// Bias gradient, if necessary.
		//	if (this->bias_term_ && this->param_propagate_down_[1]) {
		//		Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
		//		for (int n = 0; n < this->num_; ++n) {
		//			this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
		//		}
		//	}
		//	if (this->param_propagate_down_[0] || propagate_down[i]) {
		//		const Dtype* bottom_data = bottom[i]->gpu_data();
		//		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		//		for (int n = 0; n < this->num_; ++n) {
		//			// gradient w.r.t. weight. Note that we will accumulate diffs.
		//			if (this->param_propagate_down_[0]) {
		//				this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
		//					top_diff + n * this->top_dim_, weight_diff);
		//			}
		//			// gradient w.r.t. bottom data, if necessary.
		//			if (propagate_down[i]) {
		//				this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
		//					bottom_diff + n * this->bottom_dim_);
		//			}
		//		}
		//	}
		//}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvHashLayer);
}  // namespace caffe
