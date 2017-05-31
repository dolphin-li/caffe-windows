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
  const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];

  //forward channel and dense res
  top[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)num_output_;
  top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];

  for (int i = 0; i < batch_num; ++i)
  {
	  const float* bottom_data = batch_hash_ptr;
	  const unsigned char*offset_data = batch_offset_ptr;
	  const PACKED_POSITION *pos_tags = batch_posTag_ptr;
	  const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
	  const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
	  const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

	  float *top_data = top_batch_hash_ptr;
	  float *out_buf = (float*)out_col_buffer_.mutable_cpu_data();

	  forward_cpu_gemm(bottom_data, offset_data, pos_tags, m_bar, r_bar,
		  channels_, num_output_, defNum, dense_res,out_buf);

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
  

#if DUMP_2_TXT//for debug
  writeDenseKernel_2_HF5("kernel.hf5");
  writeBias_2_HF5("bias.hf5");
#endif
}

template <typename Dtype>
void ConvHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  //memset to zero, as the weight_diff will be accumulated within the batch
  caffe_set(this->blobs_[0]->count(), (Dtype)0,
	  weight_diff);
  Dtype* bias_diff = NULL;
  if (this->bias_term_)
  {
	  caffe_set(this->blobs_[1]->count(), (Dtype)0,
		  this->blobs_[1]->mutable_cpu_diff());
	  bias_diff = this->blobs_[1]->mutable_cpu_diff();
  }

  const float *top_hash_dif = (const float*)top[HASH_DATA_BLOB]->cpu_diff();
  const float *bt_hash = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
  float *bt_hash_dif = (float*)bottom[HASH_DATA_BLOB]->mutable_cpu_diff();

  const unsigned char* offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
  const PACKED_POSITION *posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();

  
  int batch_num = bottom[M_BAR_BLOB]->shape(0);
  const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
  for (int i = 0; i < batch_num; ++i)
  {
	  const float* bottom_data = bt_hash;
	  float *bottom_dif = bt_hash_dif;
	  const float *top_dif = top_hash_dif;

	  const unsigned char* offset_data = offset_ptr;
	  const PACKED_POSITION *pos_tags = posTag_ptr;
	  const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
	  const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
	  const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

	  //convert top dif to out_col_buf
	  top_hash2col_cpu(top_dif,pos_tags,m_bar,num_output_,defNum,(float*)out_col_buffer_.mutable_cpu_data());
	  //convert bottom data to col_buf
	  conv_hash2col_cpu(bottom_data, offset_data, pos_tags, kernel_shape_.cpu_data(),
		  m_bar, r_bar, channels_, defNum, dense_res, (float*)col_buffer_.mutable_cpu_data());

	  // Bias gradient, if necessary.
	  if (this->bias_term_ && this->param_propagate_down_[1]) 
	  {
		  this->backward_cpu_bias((float*)bias_diff, (const float*)out_col_buffer_.cpu_data(),defNum);
	  }
	  if (this->param_propagate_down_[0])// || propagate_down[i]) 
	  {  
			// gradient w.r.t. weight. Note that we will accumulate diffs.
			if (this->param_propagate_down_[0]) {
				this->weight_cpu_gemm((const float*)col_buffer_.cpu_data(),(const float*)out_col_buffer_.cpu_data(),
					(float*)weight_diff,channels_,num_output_,defNum);
			}
			// gradient w.r.t. bottom data, if necessary.
			//if (propagate_down[i]) 
			{
				//NOTE: now col_buf is updated to bottom_dif in backward_cpu_gemm
				this->backward_cpu_gemm((const float*)out_col_buffer_.cpu_data(), channels_,
					num_output_,defNum, (float*)col_buffer_.mutable_cpu_data());
#if 0
				char name[128];
				sprintf(name,"col_buf_%d.hf5",i);
				writeDense_2_HF5((const float*)col_buffer_.cpu_data(), col_buffer_.shape(0), 1, col_buffer_.shape(1), name);
#endif
				//convert the col_buf to bottom dif
				bottom_col2hash_cpu(bottom_dif, offset_data, pos_tags, kernel_shape_.cpu_data(),
					m_bar, r_bar, channels_, defNum, dense_res, (const float*)col_buffer_.cpu_data());
			}
	  }
	  

	  //to next hash
	  const int m = m_bar * m_bar * m_bar;
	  const int r = r_bar * r_bar * r_bar;

	  bt_hash += m * channels_;
	  bt_hash_dif += m * channels_;
	  offset_ptr += r * 3;
	  posTag_ptr += m;

	  top_hash_dif += m * num_output_;
  }

#if DUMP_2_TXT//for debug
  writeDenseKernelDif_2_HF5("kernel_dif.hf5");
  writeBiasDif_2_HF5("bias_dif.hf5");
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ConvHashLayer);
#endif

INSTANTIATE_CLASS(ConvHashLayer);
REGISTER_LAYER_CLASS(ConvHash);
}  // namespace caffe
