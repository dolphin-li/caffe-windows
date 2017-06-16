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
}



template <typename Dtype>
void BNHashLayer<Dtype>::reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	//reshape top channel and dense res
	std::vector<int> scalar_shape(1, 1);
	top[CHANNEL_BLOB]->Reshape(scalar_shape);
	top[DENSE_RES_BLOB]->Reshape(scalar_shape);
	//have to fill value here for subsequent layer setup
	top[CHANNEL_BLOB]->mutable_cpu_data()[0] = bottom[CHANNEL_BLOB]->cpu_data()[0];
	top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];
	// Configure output channels and groups.
	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	
	const Blob<Dtype> *m_bar_blob = bottom[M_BAR_BLOB];
	if (!m_bar_blob->num_axes())
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
		exit(0);
		return;
	}
	//NOTE: the copied split blob will not have the valid m before forward().
	if (!bottom[M_BAR_BLOB]->cpu_data()[0])
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
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

	//printf("batch num %d, top channel %d, dense_res %d\n",
	//	batch_num,
	//	(int)top[CHANNEL_BLOB]->cpu_data()[0],
	//	(int)top[DENSE_RES_BLOB]->cpu_data()[0]);
}


template <typename Dtype>
void BNHashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	// Shape the tops from the bottom hash structure info: offset, pos_tag, m_bar...
	if (bottom[HASH_DATA_BLOB]->count() == 1)		//data not transferred
	{
		printf("*************Data not transferred. cannot reshape topHashData!**********\n");
		printf("*************We just simply init top with shape(1,1,1,1)****************\n");
		std::vector<int> scalar_shape(1, 1);
		top[CHANNEL_BLOB]->Reshape(scalar_shape);
		top[DENSE_RES_BLOB]->Reshape(scalar_shape);
		top[HASH_DATA_BLOB]->Reshape(scalar_shape);

		//have to fill value here for subsequent layer setup
		top[CHANNEL_BLOB]->mutable_cpu_data()[0] = bottom[CHANNEL_BLOB]->cpu_data()[0];
		top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];


		return;
	}

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
	switch (Caffe::mode())
	{
	default:
		break;
	case Caffe::GPU:
		caffe_gpu_set(mean_multiplier_.count(), Dtype(1), mean_multiplier_.mutable_gpu_data());
		break;
	case Caffe::CPU:
		caffe_set(mean_multiplier_.count(), Dtype(1), mean_multiplier_.mutable_cpu_data());
		break;
	}
}


//template <typename Dtype>
//void BNHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top) 
//{
//	//total num
//	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
//	int total_defNum = 0;
//	for (int i = 0; i < batch_num; i++)
//	{
//		total_defNum += (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
//	}
//	CHECK_GT(total_defNum, 0);
//	const float ave_w = 1.f / (float)total_defNum;
//
//	if (use_global_stats_) {
//		// use the stored mean/variance estimates.
//		const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
//			0 : 1 / this->blobs_[2]->cpu_data()[0];
//		caffe_cpu_scale(variance_.count(), scale_factor,
//			this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
//		caffe_cpu_scale(variance_.count(), scale_factor,
//			this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
//	}
//	else
//	{
//		/********1. compute the mean EX for each channel *************/
//		const float *bt_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
//		const unsigned char* bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
//		const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
//
//		memset(mean_.mutable_cpu_data(), 0, sizeof(float)*channels_);
//		for (int i = 0; i < batch_num; ++i)
//		{
//			const float* bottom_data = bt_hash_ptr;
//			const unsigned char*offset_data = bt_offset_ptr;
//			const PACKED_POSITION *pos_tags = bt_posTag_ptr;
//			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
//			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
//			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
//
//			calc_sum(bottom_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, ave_w, (float*)temp_.mutable_cpu_data());
//			for (int c = 0; c < channels_; c++)
//			{
//				mean_.mutable_cpu_data()[c] += temp_.cpu_data()[c];
//			}
//
//			//to next hash
//			const int m = m_bar * m_bar * m_bar;
//			const int r = r_bar * r_bar * r_bar;
//
//			bt_hash_ptr += m * channels_;
//			bt_offset_ptr += r * 3;
//			bt_posTag_ptr += m;
//		}
//	}
//	
//	
//	/**********************2 substract mean****************/
//	float *tp_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
//	const float *bt_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
//	const unsigned char *bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
//	const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
//
//	//copy bottom to top
//	memcpy(tp_hash_ptr,bt_hash_ptr,sizeof(float)*batch_hash_size_ * channels_);
//	//subtract mean
//	for (int i = 0; i < batch_num; ++i)
//	{
//		float* top_data = tp_hash_ptr;
//		const unsigned char* offset_data = bt_offset_ptr;
//		const PACKED_POSITION *pos_tags = bt_posTag_ptr;
//		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
//		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
//		const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
//
//		hash_subtract_scalar(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, (const float*)mean_.cpu_data());
//		
//		//to next hash
//		const int m = m_bar * m_bar * m_bar;
//		const int r = r_bar * r_bar * r_bar;
//
//		tp_hash_ptr += m * channels_;
//		bt_offset_ptr += r * 3;
//		bt_posTag_ptr += m;
//	}
//	
//	/********************3. compute variance using var(X) = E((X-EX)^2)***********************/
//	if (!use_global_stats_) 
//	{
//		memset(variance_.mutable_cpu_data(), 0, sizeof(float)*channels_);
//		const float *tp_hash_ptr = (const float*)top[HASH_DATA_BLOB]->cpu_data();
//		const unsigned char* bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
//		const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
//		// E((X-EX)^2)
//		for (int i = 0; i < batch_num; ++i)
//		{
//			const float* top_data = tp_hash_ptr;	//here top is already X - EX
//			const unsigned char* offset_data = bt_offset_ptr;
//			const PACKED_POSITION *pos_tags = bt_posTag_ptr;
//			const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
//			const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
//			const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
//
//			calc_square_sum(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, ave_w, (float*)temp_.mutable_cpu_data());
//			for (int c = 0; c < channels_; c++)
//			{
//				variance_.mutable_cpu_data()[c] += temp_.cpu_data()[c];
//			}
//
//			//to next hash
//			const int m = m_bar * m_bar * m_bar;
//			const int r = r_bar * r_bar * r_bar;
//
//			tp_hash_ptr += m * channels_;
//			bt_offset_ptr += r * 3;
//			bt_posTag_ptr += m;
//		}
//
//
//		// compute and save moving average
//		this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
//		this->blobs_[2]->mutable_cpu_data()[0] += 1;
//		
//		caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
//						moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
//		
//		
//		Dtype bias_correction_factor = total_defNum > 1 ? Dtype(total_defNum) / (total_defNum - 1) : 1;
//		caffe_cpu_axpby(variance_.count(), bias_correction_factor,
//						variance_.cpu_data(), moving_average_fraction_,
//						this->blobs_[1]->mutable_cpu_data());
//	}
//
//
//	/********************3. compute final top (X-mean(X))/(sqrt(var(X)+eps))***********************/
//	// normalize variance
//	caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
//	caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
//				variance_.mutable_cpu_data());
//
//	for (int c=0;c<channels_;c++)
//	{
//		inv_sqrt_var_.mutable_cpu_data()[c] = 1.f / variance_.cpu_data()[c];
//	}
//	// div by sqrt(var(X)+eps)
//	tp_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
//	bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
//	bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
//	for (int i = 0; i < batch_num; ++i)
//	{
//		float* top_data = tp_hash_ptr;
//		const unsigned char* offset_data = bt_offset_ptr;
//		const PACKED_POSITION *pos_tags = bt_posTag_ptr;
//		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
//		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
//		const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
//
//		hash_mult_scalar(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, (const float*)inv_sqrt_var_.cpu_data());
//
//
//		//to next hash
//		const int m = m_bar * m_bar * m_bar;
//		const int r = r_bar * r_bar * r_bar;
//
//		tp_hash_ptr += m * channels_;
//		bt_offset_ptr += r * 3;
//		bt_posTag_ptr += m;
//	}
//}

template <typename Dtype>
void BNHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	//printf("******************BNHash forward begin\n");


	//forward channel and dense res
	top[CHANNEL_BLOB]->mutable_cpu_data()[0] = bottom[CHANNEL_BLOB]->cpu_data()[0];
	top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];

	//total num
	const int total_defNum = temp_.shape(1);
	const Dtype mean_div = Dtype(1) / Dtype(total_defNum);
	//const Dtype var_div = Dtype(1) / Dtype(std::max(1, total_defNum - 1));
	const Dtype var_div = mean_div;	//will be bias-corrected when adding to blob[1]

									// prepare temp_ array
	forward_hash2temp_cpu(bottom, top);
	//writeDense_2_TXT((const float*)temp_.cpu_data(),temp_.count(),"temp_cpu.txt");

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
		//memset(mean_.mutable_cpu_data(), 0, sizeof(float)*channels_);
		caffe_cpu_gemv(CblasNoTrans, channels_, total_defNum, mean_div,
			temp_.cpu_data(), mean_multiplier_.cpu_data(), Dtype(0), mean_.mutable_cpu_data());
		//writeDense_2_TXT((const float*)mean_.cpu_data(), mean_.count(), "mean_cpu.txt");
	}


	/**********************2 substract mean****************/
	// subtract mean
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
		total_defNum, 1, (Dtype)-1, mean_.cpu_data(), mean_multiplier_.cpu_data(),
		(Dtype)1, temp_.mutable_cpu_data());
	//writeDense_2_TXT((const float*)temp_.cpu_data(), temp_.count(), "temp_cpu.txt");
	/********************3. compute variance using var(X) = E((X-EX)^2)***********************/
	if (!use_global_stats_)
	{
		//memset(variance_.mutable_cpu_data(), 0, sizeof(float)*channels_);
		caffe_powx(temp_.count(), temp_.cpu_data(), Dtype(2),
			temp2_.mutable_cpu_data());  // (X-EX)^2
		caffe_cpu_gemv(CblasNoTrans, channels_, total_defNum, var_div,
			temp2_.cpu_data(), mean_multiplier_.cpu_data(), Dtype(0), variance_.mutable_cpu_data());

		// compute and save moving average
		this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
		this->blobs_[2]->mutable_cpu_data()[0] += 1;

		caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
			moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());

		Dtype bias_correction_factor = total_defNum > 1 ? Dtype(total_defNum) / Dtype(total_defNum - 1) : 1;
		caffe_cpu_axpby(variance_.count(), bias_correction_factor,
			variance_.cpu_data(), moving_average_fraction_, this->blobs_[1]->mutable_cpu_data());
	}

	/********************4. compute final top (X-mean(X))/(sqrt(var(X)+eps))***********************/
	// normalize variance
	caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
	caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
		variance_.mutable_cpu_data());

	//set variance to inV
	for (int c = 0; c < channels_; c++)
	{
		inv_sqrt_var_.mutable_cpu_data()[c] = 1.f / variance_.cpu_data()[c];
	}

	// replicate inv_variance to input size
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
		(Dtype)1, inv_sqrt_var_.cpu_data(), mean_multiplier_.cpu_data(), (Dtype)0,
		temp2_.mutable_cpu_data());
	
	caffe_mul(temp_.count(), temp_.cpu_data(), temp2_.cpu_data(), temp_.mutable_cpu_data());
	//writeDense_2_TXT((const float*)temp_.cpu_data(), temp_.count(), "temp_cpu.txt");

	caffe_set(top[HASH_DATA_BLOB]->count(), (Dtype)0,
		top[HASH_DATA_BLOB]->mutable_cpu_data());

	forward_temp2hash_cpu(bottom, top);

	//printf("******************BNHash forward end\n");

#if TIANJIA_DEBUG_GPU	//debug GPU

	
	printf("\n===============CHECKING BNHash Forward GPU CPU, Res %d======================\n",(int)top[DENSE_RES_BLOB]->cpu_data()[0]);
	vector<Blob<Dtype>*>  gpu_top(top.size());

	for (int i = 0; i < (int)top.size(); i++)
	{
		gpu_top[i] = new Blob<Dtype>();
		gpu_top[i]->ReshapeLike(*top[i]);
	}

	cudaThreadSynchronize();
	Forward_gpu(bottom, gpu_top);
	cudaThreadSynchronize();
	//check
	float eps = 1e-6f;
	//writeDense_2_TXT((const float*)gpu_top[HASH_DATA_BLOB]->cpu_data(), gpu_top[HASH_DATA_BLOB]->count(), "gpu_top.txt");
	//writeDense_2_TXT((const float*)top[HASH_DATA_BLOB]->cpu_data(), top[HASH_DATA_BLOB]->count(), "cpu_top.txt");
	for (int i = 0; i < (int)top.size(); i++)
	{
		for (int j = 0; j < top[i]->count(); j++)
		{
			//if (fabs(top[i]->cpu_data()[j] - gpu_top[i]->cpu_data()[j]) / (fabs(top[i]->cpu_data()[j]) + 1e-7f) > eps)
			if (fabs(top[i]->cpu_data()[j] - gpu_top[i]->cpu_data()[j]) > eps)
			{
				printf("Error: BNHash Forward cpu gpu not match! cpu: %.7f, gpu: %.7f!\n", top[i]->cpu_data()[j], gpu_top[i]->cpu_data()[j]);
			}
		}
	}

	for (int i = 0; i < (int)top.size(); i++)
	{
		delete gpu_top[i];
	}
	printf("===============CHECKING BNHash Forward GPU CPU  DONE======================\n");
#endif
}

template <typename Dtype>
void BNHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
	//printf("******************BNHash backward begin\n");

	const Dtype* top_diff = top[HASH_DATA_BLOB]->cpu_diff();
	Dtype* bottom_diff = bottom[HASH_DATA_BLOB]->mutable_cpu_diff();
	const int total_defNum = temp_.shape(1);
	const Dtype mean_div = Dtype(1) / Dtype(total_defNum);
	//const Dtype var_div = Dtype(1) / Dtype(std::max(1, total_defNum - 1));
	const Dtype var_div = mean_div;	//will be bias-corrected when adding to blob[1]

	//convert top_dif to tmp
	backward_topDif2temp_cpu(bottom, top);
	if (use_global_stats_) 
	{
		// replicate inv_variance to input size
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
			(Dtype)1, inv_sqrt_var_.cpu_data(), mean_multiplier_.cpu_data(), (Dtype)0,
			temp2_.mutable_cpu_data());
		caffe_mul(temp_.count(), temp_.cpu_data(), temp2_.cpu_data(), temp_.mutable_cpu_data());
		backward_temp2BottomDif_cpu(bottom, top);
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
	top_2_buf(bottom, top, temp2_);	//convert Y to temp2_
	//dE/dY \cdot Y; // NOTE: here temp_ is modified
	caffe_mul(temp_.count(), temp_.cpu_data(), temp2_.cpu_data(), temp_.mutable_cpu_data());
	//mean
	caffe_cpu_gemv(CblasNoTrans, channels_, total_defNum, mean_div,
		temp_.cpu_data(), mean_multiplier_.cpu_data(), Dtype(0), mean_.mutable_cpu_data());
	
	//step2. mean(dE/dY \cdot Y) \cdot Y
	//reshape mean to input size
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
		(Dtype)1, mean_.cpu_data(), mean_multiplier_.cpu_data(), (Dtype)0,
		temp_.mutable_cpu_data());
	// mean(dE/dY \cdot Y) \cdot Y
	caffe_mul(temp_.count(), temp_.cpu_data(), temp2_.cpu_data(), temp2_.mutable_cpu_data());

	//step3. dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y
	//convert top_dif to tmp
	backward_topDif2temp_cpu(bottom, top);
	//mean(dE/dY)
	caffe_cpu_gemv(CblasNoTrans, channels_, total_defNum, mean_div,
		temp_.cpu_data(), mean_multiplier_.cpu_data(), Dtype(0), mean_.mutable_cpu_data());
	
	//dE/dY - mean(dE/dY \cdot Y) \cdot Y
	caffe_sub(temp_.count(), temp_.cpu_data(), temp2_.cpu_data(), temp_.mutable_cpu_data());

	//-= mean(dE/dY)
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
		total_defNum, 1, (Dtype)-1, mean_.cpu_data(), mean_multiplier_.cpu_data(),
		(Dtype)1, temp_.mutable_cpu_data());

	//step4. (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y) / sqrt(var(X) + eps)
	// replicate inv_variance to input size
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, total_defNum, 1,
		(Dtype)1, inv_sqrt_var_.cpu_data(), mean_multiplier_.cpu_data(), (Dtype)0,
		temp2_.mutable_cpu_data());
	caffe_mul(temp_.count(), temp_.cpu_data(), temp2_.cpu_data(), temp_.mutable_cpu_data());

	caffe_set(bottom[HASH_DATA_BLOB]->count(), (Dtype)0,
		bottom[HASH_DATA_BLOB]->mutable_cpu_diff());
	backward_temp2BottomDif_cpu(bottom, top);

	//printf("******************BNHash backward end\n");

#if TIANJIA_DEBUG_GPU	//debug GPU
	printf("\n===============CHECKING BNHash Backward GPU CPU======================\n");


	vector<Blob<Dtype>*>  gpu_bottom(bottom.size());
	for (int i = 0; i < (int)bottom.size(); i++)
	{
		gpu_bottom[i] = new Blob<Dtype>();
		gpu_bottom[i]->ReshapeLike(*bottom[i]);
		caffe::caffe_copy(gpu_bottom[i]->count(), bottom[i]->cpu_data(), gpu_bottom[i]->mutable_cpu_data());
		caffe::caffe_copy(gpu_bottom[i]->count(), bottom[i]->cpu_diff(), gpu_bottom[i]->mutable_cpu_diff());
	}

	cudaThreadSynchronize();
	Backward_gpu(top, propagate_down, gpu_bottom);
	cudaThreadSynchronize();
	//check
	float eps = 1e-6f;

	for (int i = 0; i < (int)bottom.size(); i++)
	{
		for (int j = 0; j < bottom[i]->count(); j++)
		{
			//if (fabs(bottom[i]->cpu_diff()[j] - gpu_bottom[i]->cpu_diff()[j]) / (fabs(bottom[i]->cpu_diff()[j]) + 1e-7f) > eps)
			if (fabs(bottom[i]->cpu_diff()[j] - gpu_bottom[i]->cpu_diff()[j]) > eps)
			{
				printf("Error: BNHash Backward cpu gpu not match! cpu: %.7f, gpu: %.7f!\n", bottom[i]->cpu_diff()[j], gpu_bottom[i]->cpu_diff()[j]);
			}
		}
	}
	for (int i = 0; i < (int)top.size(); i++)
	{
		delete gpu_bottom[i];
	}

	printf("===============CHECKING BNHash Backward GPU CPU DONE======================\n");
#endif
}


template <typename Dtype>
void BNHashLayer<Dtype>::forward_hash2temp_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const float *hash = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	Dtype* temp = temp_.mutable_cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = temp_.shape(1);
	for (int i = 0; i < batch_num; ++i)
	{
		const float *cur_hash = hash;
		const unsigned char* cur_offset = offsets;
		const PACKED_POSITION *cur_posTag = posTags;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		Dtype *cur_tmp = temp;

		for (int v=0;v<m;v++)
		{
			if (!ishashVoxelDefined(&cur_posTag[v]))
			{
				continue;
			}
			const float *data_ptr = &cur_hash[v];
			Dtype *temp_ptr = cur_tmp;
			for (int c = 0; c < channels_; c++)
			{
				*temp_ptr = (Dtype)*data_ptr;
				data_ptr += m;
				temp_ptr += total_def_num;
			}
			cur_tmp++;
		}

		//to next hash
		hash += m * channels_;
		offsets += r*3;
		posTags += m;

		temp += def_num;
	}
}

template <typename Dtype>
void BNHashLayer<Dtype>::forward_temp2hash_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	float *hash = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	const Dtype* temp = temp_.cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = temp_.shape(1);
	for (int i = 0; i < batch_num; ++i)
	{
		float *cur_hash = hash;
		const unsigned char* cur_offset = offsets;
		const PACKED_POSITION *cur_posTag = posTags;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		const Dtype *cur_tmp = temp;

		for (int v = 0; v < m; v++)
		{
			if (!ishashVoxelDefined(&cur_posTag[v]))
			{
				continue;
			}
			float *data_ptr = &cur_hash[v];
			const Dtype *temp_ptr = cur_tmp;
			for (int c = 0; c < channels_; c++)
			{
				*data_ptr = (Dtype)*temp_ptr;
				data_ptr += m;
				temp_ptr += total_def_num;
			}
			cur_tmp++;
		}

		//to next hash
		hash += m * channels_;
		offsets += r * 3;
		posTags += m;

		temp += def_num;
	}
}



template <typename Dtype>
void BNHashLayer<Dtype>::backward_topDif2temp_cpu(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top)
{
	const float *hash_dif = (const float*)top[HASH_DATA_BLOB]->cpu_diff();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	Dtype* temp = temp_.mutable_cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = temp_.shape(1);
	for (int i = 0; i < batch_num; ++i)
	{
		const float *cur_hash_dif = hash_dif;
		const unsigned char* cur_offset = offsets;
		const PACKED_POSITION *cur_posTag = posTags;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		Dtype *cur_tmp = temp;

		for (int v = 0; v < m; v++)
		{
			if (!ishashVoxelDefined(&cur_posTag[v]))
			{
				continue;
			}
			const float *data_ptr = &cur_hash_dif[v];
			Dtype *temp_ptr = cur_tmp;
			for (int c = 0; c < channels_; c++)
			{
				*temp_ptr = (Dtype)*data_ptr;
				data_ptr += m;
				temp_ptr += total_def_num;
			}
			cur_tmp++;
		}

		//to next hash
		hash_dif += m * channels_;
		offsets += r * 3;
		posTags += m;

		temp += def_num;
	}
}



template <typename Dtype>
void BNHashLayer<Dtype>::backward_temp2BottomDif_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	float *hash_dif = (float*)bottom[HASH_DATA_BLOB]->mutable_cpu_diff();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	const Dtype* temp = temp_.cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = temp_.shape(1);
	for (int i = 0; i < batch_num; ++i)
	{
		float *cur_hash_dif = hash_dif;
		const unsigned char* cur_offset = offsets;
		const PACKED_POSITION *cur_posTag = posTags;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		const Dtype *cur_tmp = temp;

		for (int v = 0; v < m; v++)
		{
			if (!ishashVoxelDefined(&cur_posTag[v]))
			{
				continue;
			}
			float *data_ptr = &cur_hash_dif[v];
			const Dtype *temp_ptr = cur_tmp;
			for (int c = 0; c < channels_; c++)
			{
				*data_ptr = (Dtype)*temp_ptr;
				data_ptr += m;
				temp_ptr += total_def_num;
			}
			cur_tmp++;
		}

		//to next hash
		hash_dif += m * channels_;
		offsets += r * 3;
		posTags += m;

		temp += def_num;
	}
}


template <typename Dtype>
void BNHashLayer<Dtype>::top_2_buf(const vector<Blob<Dtype>*>& bottom, 
	const vector<Blob<Dtype>*>& top, Blob<Dtype> &buf)
{
	const float *hash = (const float*)top[HASH_DATA_BLOB]->cpu_data();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	Dtype* buf_ptr = buf.mutable_cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = buf.shape(1);
	for (int i = 0; i < batch_num; ++i)
	{
		const float *cur_hash = hash;
		const unsigned char* cur_offset = offsets;
		const PACKED_POSITION *cur_posTag = posTags;
		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
		const int def_num = bottom[DEFNUM_BLOB]->cpu_data()[i];
		const int m = m_bar * m_bar * m_bar;
		const int r = r_bar * r_bar * r_bar;

		Dtype *cur_buf = buf_ptr;

		for (int v = 0; v < m; v++)
		{
			if (!ishashVoxelDefined(&cur_posTag[v]))
			{
				continue;
			}
			const float *data_ptr = &cur_hash[v];
			Dtype *temp_ptr = cur_buf;
			for (int c = 0; c < channels_; c++)
			{
				*temp_ptr = (Dtype)*data_ptr;
				data_ptr += m;
				temp_ptr += total_def_num;
			}
			cur_buf++;
		}

		//to next hash
		hash += m * channels_;
		offsets += r * 3;
		posTags += m;

		buf_ptr += def_num;
	}
}

#ifdef CPU_ONLY
STUB_GPU(BNHashLayer);
#endif

INSTANTIATE_CLASS(BNHashLayer);
REGISTER_LAYER_CLASS(BNHash);
}  // namespace caffe
