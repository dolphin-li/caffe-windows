#include <algorithm>
#include <vector>

#include "caffe/layers/bn_hash_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/HashData.h"

namespace caffe {

template <typename Dtype>
void BNHashLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const BNHashParameter &param = this->layer_param_.bn_hash_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  disable_mean_ = param.disable_mean();
  disable_variance_ = param.disable_variance();
  
  eps_ = param.eps();
  
  //if (param.engine() != BatchNormParameter_Engine_CUDNN) 
  {
    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction to zero.
    for (int i = 0; i < this->blobs_.size(); ++i) {
      if (this->layer_param_.param_size() == i) {
        ParamSpec* fixed_param_spec = this->layer_param_.add_param();
        fixed_param_spec->set_lr_mult(0.f);
      }
      else {
        CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
      }
    }
  }
}

template <typename Dtype>
void BNHashLayer<Dtype>::init_self_blob(const vector<Blob<Dtype> *>& bottom)
{
	if (m_self_blob_init_flag)	//only init once
	{
		return;
	}
	m_self_blob_init_flag = true;

	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	CHECK_GT(channels_, 0);

	this->blobs_.resize(3);
	vector<int> sz;
	sz.push_back(channels_);
	this->blobs_[0].reset(new Blob<Dtype>(sz));
	this->blobs_[1].reset(new Blob<Dtype>(sz));
	sz[0] = 1;
	this->blobs_[2].reset(new Blob<Dtype>(sz));
	for (int i = 0; i < 3; ++i) 
	{
		caffe_set(this->blobs_[i]->count(), Dtype(0),
			this->blobs_[i]->mutable_cpu_data());
	}
}



template <typename Dtype>
void BNHashLayer<Dtype>::reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	// Configure output channels and groups.
	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	
	const Blob<Dtype> *m_bar_blob = bottom[M_BAR_BLOB];
	if (!m_bar_blob->num_axes())
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
		exit(0);
		return;
	}
	
	const int batch_num = m_bar_blob->shape(0);
	batch_hash_size_ = 0;
	for (int i = 0; i < batch_num; i++)
	{
		int m_bar = (int)m_bar_blob->cpu_data()[i];
		batch_hash_size_ += m_bar * m_bar * m_bar;
	}
	std::vector<int> hash_data_shape(1, batch_hash_size_ * channels_);
	top[HASH_DATA_BLOB]->Reshape(hash_data_shape);
	memset(top[HASH_DATA_BLOB]->mutable_cpu_data(), 0, sizeof(Dtype)*batch_hash_size_ * channels_);

	//reshape top channel and dense res
	std::vector<int> scalar_shape(1, 1);
	top[CHANNEL_BLOB]->Reshape(scalar_shape);
	top[DENSE_RES_BLOB]->Reshape(scalar_shape);
}


template <typename Dtype>
void BNHashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	init_self_blob(bottom);
	reshape_topHashData(bottom, top);

	vector<int> sz;
	sz.push_back(channels_);
	mean_.Reshape(sz);
	variance_.Reshape(sz);
	inv_sqrt_var_.Reshape(sz);

	// reshape temp to be channels * defNums
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	int total_defNum = 0;
	for (int i = 0; i < batch_num; i++)
		total_defNum += (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
	CHECK_GT(total_defNum, 0);

	// tmp to store the valid hash data, tmp2 store tmp^2
	vector<int> sz2;
	sz2.push_back(channels_);
	sz2.push_back(total_defNum);
	temp_.Reshape(sz2);
	temp2_.Reshape(sz2);

	// a multipler for Mv
	vector<int> sz3;
	sz3.push_back(total_defNum);
	mean_multiplier_.Reshape(sz3);
	caffe_set(mean_multiplier_.count(), Dtype(1), mean_multiplier_.mutable_cpu_data());
}


template <typename Dtype>
void BNHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	//total num
	const int batch_num = bottom[CHANNEL_BLOB]->shape(0);
	int total_defNum = 0;
	for (int i = 0; i < batch_num; i++)
	{
		total_defNum += (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
	}
	CHECK_GT(total_defNum, 0);
	const float ave_w = 1.f / (float)total_defNum;

	if (use_global_stats_) {
		// use the stored mean/variance estimates.
		const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
			0 : 1 / this->blobs_[2]->cpu_data()[0];
		caffe_cpu_scale(variance_.count(), scale_factor,
			this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
		caffe_cpu_scale(variance_.count(), scale_factor,
			this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
	}
	else
	{
		/********1. compute the mean EX for each channel *************/
		const float *bt_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
		const unsigned char* bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
		const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();

		memset(mean_.mutable_cpu_data(), 0, sizeof(float)*channels_);
		for (int i = 0; i < batch_num; ++i)
		{
			const float* bottom_data = bt_hash_ptr;
			const unsigned char*offset_data = bt_offset_ptr;
			const PACKED_POSITION *pos_tags = bt_posTag_ptr;
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

			calc_sum(bottom_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, ave_w, (float*)temp_.mutable_cpu_data());
			for (int c = 0; c < channels_; c++)
			{
				mean_.mutable_cpu_data()[c] += temp_.cpu_data()[c];
			}

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			const int r = r_bar * r_bar * r_bar;

			bt_hash_ptr += m * channels_;
			bt_offset_ptr += r * 3;
			bt_posTag_ptr += m;
		}
	}
	
	
	/**********************2 substract mean****************/
	float *tp_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
	const float *bt_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
	const unsigned char *bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();

	//copy bottom to top
	memcpy(tp_hash_ptr,bt_hash_ptr,sizeof(float)*batch_hash_size_ * channels_);
	//subtract mean
	for (int i = 0; i < batch_num; ++i)
	{
		float* top_data = tp_hash_ptr;
		const unsigned char* offset_data = bt_offset_ptr;
		const PACKED_POSITION *pos_tags = bt_posTag_ptr;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

		hash_subtract_scalar(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, (const float*)mean_.cpu_data());
		

		//to next hash
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		tp_hash_ptr += m * channels_;
		bt_offset_ptr += r * 3;
		bt_posTag_ptr += m;
	}
	
	/********************3. compute variance using var(X) = E((X-EX)^2)***********************/
	if (!use_global_stats_) 
	{
		memset(variance_.mutable_cpu_data(), 0, sizeof(float)*channels_);
		const float *tp_hash_ptr = (const float*)top[HASH_DATA_BLOB]->cpu_data();
		const unsigned char* bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
		const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
		// E((X-EX)^2)
		for (int i = 0; i < batch_num; ++i)
		{
			const float* top_data = tp_hash_ptr;	//here top is already X - EX
			const unsigned char* offset_data = bt_offset_ptr;
			const PACKED_POSITION *pos_tags = bt_posTag_ptr;
			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

			calc_square_sum(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, ave_w, (float*)temp_.mutable_cpu_data());
			for (int c = 0; c < channels_; c++)
			{
				variance_.mutable_cpu_data()[c] += temp_.cpu_data()[c];
			}

			//to next hash
			const int m = m_bar * m_bar * m_bar;
			const int r = r_bar * r_bar * r_bar;

			tp_hash_ptr += m * channels_;
			bt_offset_ptr += r * 3;
			bt_posTag_ptr += m;
		}


		// compute and save moving average
		this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
		this->blobs_[2]->mutable_cpu_data()[0] += 1;
		
		caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
						moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
		
		
		Dtype bias_correction_factor = total_defNum > 1 ? Dtype(total_defNum) / (total_defNum - 1) : 1;
		caffe_cpu_axpby(variance_.count(), bias_correction_factor,
						variance_.cpu_data(), moving_average_fraction_,
						this->blobs_[1]->mutable_cpu_data());
	}


	/********************3. compute final top (X-mean(X))/(sqrt(var(X)+eps))***********************/
	// normalize variance
	caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
	caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
				variance_.mutable_cpu_data());

	for (int c=0;c<channels_;c++)
	{
		inv_sqrt_var_.mutable_cpu_data()[c] = 1.f / variance_.cpu_data()[c];
	}
	// div by sqrt(var(X)+eps)
	tp_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
	bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
	for (int i = 0; i < batch_num; ++i)
	{
		float* top_data = tp_hash_ptr;
		const unsigned char* offset_data = bt_offset_ptr;
		const PACKED_POSITION *pos_tags = bt_posTag_ptr;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];

		hash_mult_scalar(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, (const float*)inv_sqrt_var_.cpu_data());


		//to next hash
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		tp_hash_ptr += m * channels_;
		bt_offset_ptr += r * 3;
		bt_posTag_ptr += m;
	}
}

template <typename Dtype>
void BNHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //const Dtype* top_diff;
  //if (bottom[0] != top[0]) {
  //  top_diff = top[0]->cpu_diff();
  //} else {
  //  caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
  //  top_diff = x_norm_.cpu_diff();
  //}
  //Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //if (use_global_stats_) {
  //  if (disable_variance_) {
  //    if (bottom[0] != top[0]) {
  //      caffe_copy(top[0]->count(), top[0]->cpu_diff(), bottom_diff);
  //    }
  //  }
  //  else {
  //    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
  //  }
  //  return;
  //}
  //const Dtype* top_data = x_norm_.cpu_data();
  //int num = bottom[0]->shape()[0];
  //int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  //// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  ////
  //// dE(Y)/dX =
  ////   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  ////     ./ sqrt(var(X) + eps)
  ////
  //// where \cdot and ./ are hadamard product and elementwise division,
  //// respectively, dE/dY is the top diff, and mean/var/sum are all computed
  //// along all dimensions except the channels dimension.  In the above
  //// equation, the operations allow for expansion (i.e. broadcast) along all
  //// dimensions except the channels dimension where required.
  //// --------------------------------------------------------
  //// If disable_vairance is set, the derivative change to
  //// dE(Y)/dX = dE/dY - mean(dE/dY)
  //// If disable_mean is set, derivative becomes
  //// dE(Y)/dX =
  ////   (dE/dY - mean(dE/dY \cdot Y) \cdot Y)
  ////     ./ sqrt(var(X) + eps)

  //if (!disable_variance_) {
  //  // sum(dE/dY \cdot Y)
  //  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  //  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
  //                        bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
  //                        num_by_chans_.mutable_cpu_data());
  //  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
  //                        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
  //                        mean_.mutable_cpu_data());

  //  // reshape (broadcast) the above
  //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
  //                        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
  //                        num_by_chans_.mutable_cpu_data());
  //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
  //                        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
  //                        spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  //  // sum(dE/dY \cdot Y) \cdot Y
  //  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);
  //}
  //else {
  //  caffe_set(temp_.count(), Dtype(0), bottom_diff);
  //}

  //if (!disable_mean_) {
  //  // sum(dE/dY)
  //  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
  //                        top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
  //                        num_by_chans_.mutable_cpu_data());
  //  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
  //                        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
  //                        mean_.mutable_cpu_data());
  //  // reshape (broadcast) the above to make
  //  // sum(dE/dY)+sum(dE/dY \cdot Y) \cdot Y
  //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
  //                        batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
  //                        num_by_chans_.mutable_cpu_data());
  //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
  //                        spatial_dim, 1, 1., num_by_chans_.cpu_data(),
  //                        spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);
  //}

  //// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  //caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
  //    Dtype(-1. / (num * spatial_dim)), bottom_diff);

  //if (!disable_variance_) {
  //  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  //  // pass.
  //  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
  //}
}


#ifdef CPU_ONLY
STUB_GPU(BNHashLayer);
#endif

INSTANTIATE_CLASS(BNHashLayer);
REGISTER_LAYER_CLASS(BNHash);
}  // namespace caffe
