#include <vector>

#include "caffe/layers/conv_hash_layer.hpp"
#include "caffe/util/MyMacro.h"

namespace caffe {

template <typename Dtype>
void ConvHashLayer<Dtype>::compute_output_shape() 
{
  //the output shape can be got from top structure blobs
}

template <typename Dtype>
void ConvHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  
  const float *batch_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
  const unsigned char*batch_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
  const PACKED_POSITION *batch_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
 
  float *top_batch_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
  int batch_num = bottom[M_BAR_BLOB]->shape(0);

  for (int i = 0; i < batch_num; ++i)
  {
	  const float* bottom_data = batch_hash_ptr;
	  const unsigned char*offset_data = batch_offset_ptr;
	  const PACKED_POSITION *pos_tags = batch_posTag_ptr;
	  const int m_bar = bottom[M_BAR_BLOB]->cpu_data()[i];
	  const int r_bar = bottom[R_BAR_BLOB]->cpu_data()[i];
	  const int defNum = bottom[DEFNUM_BLOB]->cpu_data()[i];

	  float *top_data = top_batch_hash_ptr;
	  float *out_buf = (float*)out_col_buffer_.mutable_cpu_data();

	  forward_cpu_gemm(bottom_data, offset_data, pos_tags, m_bar, r_bar,
		  channels_, num_output_, defNum, out_buf);

	  if (this->bias_term_) {
		  const float* bias = (const float*)this->blobs_[1]->cpu_data();
		  this->forward_cpu_bias(out_buf, bias,defNum);
	  }

	  //convert out col buf to top
	  conv_col2hash_cpu(pos_tags, top_data, m_bar, num_output_, defNum, out_buf);

	  //to next hash
	  const int m = m_bar * m_bar * m_bar;
	  const int r = r_bar * r_bar * r_bar;

	  batch_hash_ptr += m * channels_;
	  batch_offset_ptr += r * 3;
	  batch_posTag_ptr += m;

	  top_batch_hash_ptr += m * num_output_;
  }

}

template <typename Dtype>
void ConvHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//TODO:
  //const Dtype* weight = this->blobs_[0]->cpu_data();
  //Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  //for (int i = 0; i < top.size(); ++i) {
  //  const Dtype* top_diff = top[i]->cpu_diff();
  //  const Dtype* bottom_data = bottom[i]->cpu_data();
  //  Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
  //  // Bias gradient, if necessary.
  //  if (this->bias_term_ && this->param_propagate_down_[1]) {
  //    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
  //    for (int n = 0; n < this->num_; ++n) {
  //      this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
  //    }
  //  }
  //  if (this->param_propagate_down_[0] || propagate_down[i]) {
  //    for (int n = 0; n < this->num_; ++n) {
  //      // gradient w.r.t. weight. Note that we will accumulate diffs.
  //      if (this->param_propagate_down_[0]) {
  //        this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
  //            top_diff + n * this->top_dim_, weight_diff);
  //      }
  //      // gradient w.r.t. bottom data, if necessary.
  //      if (propagate_down[i]) {
  //        this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
  //            bottom_diff + n * this->bottom_dim_);
  //      }
  //    }
  //  }
  //}
}

#ifdef CPU_ONLY
STUB_GPU(ConvHashLayer);
#endif

INSTANTIATE_CLASS(ConvHashLayer);

}  // namespace caffe
