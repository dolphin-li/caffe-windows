#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_hash_layer.hpp"
#include "caffe/util/MyMacro.h"


namespace caffe {

template <typename Dtype>
void BaseConvHashLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	//CHECK size
	if (bottom.size()!= HASH_DATA_SIZE + HASH_STRUCTURE_SIZE)
	{
		printf("Fatal error: bottom size should be %d\n", HASH_DATA_SIZE + HASH_STRUCTURE_SIZE);
		exit(0);
	}
	if (top.size()!= HASH_DATA_SIZE)
	{
		printf("Fatal error: top size should be %d\n", HASH_DATA_SIZE);
		exit(0);
	}

  // Configure the kernel size, padding, stride, and inputs.
  const ConvHashParameter &conv_param = this->layer_param_.conv_hash_param();
  
  num_spatial_axes_ = 3;	//for 3D convolution

  vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  {
	  const int num_kernel_dims = conv_param.kernel_size_size();
	  CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
		  << "kernel_size must be specified once, or once per spatial dimension "
		  << "(kernel_size specified " << num_kernel_dims << " times; "
		  << num_spatial_axes_ << " spatial dims).";
	  for (int i = 0; i < num_spatial_axes_; ++i) {
		  kernel_shape_data[i] =
			  conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
	  }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }

  //NOTE: the blobs must be resize here, as the net.cpp will compare blob num and param num
  bias_term_ = conv_param.bias_term();
  if (bias_term_) {
	  this->blobs_.resize(2);
  }
  else {
	  this->blobs_.resize(1);
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
#if 0	//Not considered yet
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
#endif

#if 0	//in our hash case, the channels cannot be obtained from the blob shape, but only from the data
		//so we have init them in reshape stage
  // Configure output channels and groups.
  channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
  CHECK_GT(channels_, 0);
  num_output_ = conv_param.num_output();
  CHECK_GT(num_output_, 0);
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = conv_param.bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
#endif
}

//In this layer, the top structure is the same as the bottom structure, 
//so we avoid to copy the structure blobs to save memroy
template <typename Dtype>
void BaseConvHashLayer<Dtype>::reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const ConvHashParameter &conv_param = this->layer_param_.conv_hash_param();
	std::vector<int> scalar_shape(1, 1);
	top[CHANNEL_BLOB]->Reshape(scalar_shape);
	top[DENSE_RES_BLOB]->Reshape(scalar_shape);
	//NOTE: transfer channel and denseRes for subsequent layer setup
	top[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)conv_param.num_output();
	top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];

	if (!bottom[M_BAR_BLOB]->num_axes())
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
		return;
	}

	const int top_channels = num_output_;
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	int batch_hash_size = 0;
	for (int i = 0; i < batch_num; i++)
	{
		int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		batch_hash_size += m_bar * m_bar * m_bar;
	}
	std::vector<int> hash_data_shape(1, batch_hash_size * top_channels);
	top[HASH_DATA_BLOB]->Reshape(hash_data_shape);
	//memset(top[HASH_DATA_BLOB]->mutable_cpu_data(), 0, sizeof(Dtype)*batch_hash_size * top_channels);

	//printf("batch num %d, top channel %d, dense_res %d\n",
	//	batch_num,
	//	(int)top[CHANNEL_BLOB]->cpu_data()[0],
	//	(int)top[DENSE_RES_BLOB]->cpu_data()[0]);

}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{  
  // Shape the tops from the bottom hash structure info: offset, pos_tag, m_bar...
  if (bottom[HASH_DATA_BLOB]->count()==1)		//data not transferred
  {
	  printf("*************Data not transferred. cannot reshape topHashData!**********\n");
	  printf("*************We just simply init top with scalar shape****************\n");
	
	  const ConvHashParameter &conv_param = this->layer_param_.conv_hash_param();

	  std::vector<int> scalar_shape(1, 1);
	  top[CHANNEL_BLOB]->Reshape(scalar_shape);
	  //fill the channel blob for subsequent layer setup
	  top[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)conv_param.num_output();
	  top[DENSE_RES_BLOB]->Reshape(scalar_shape);
	  top[DENSE_RES_BLOB]->mutable_cpu_data()[0] = bottom[DENSE_RES_BLOB]->cpu_data()[0];
	  top[HASH_DATA_BLOB]->Reshape(scalar_shape);	//simple fill data blob to avoid empty error
	  return;
  }
  
  //init weight kernal in reshape stage
  reshape_weight_bias(bottom);

  //reshape the top hash data
  reshape_topHashData(bottom, top);

  //reshape the top hash data
  reshape_colBuf(bottom);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::reshape_colBuf(const vector<Blob<Dtype>*>& bottom)
{
	if (!bottom[M_BAR_BLOB]->num_axes())
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
		return;
	}
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	//get the max size of current batch
	int max_defined_size = 0;
	for (int i = 0; i < batch_num; i++)
	{
		int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
		if (defNum > max_defined_size)
		{
			max_defined_size = defNum;
		}
	}

	const int* kernel_shape_data = kernel_shape_.cpu_data();
	int filter_size = 1;
	for (int i=0;i<num_spatial_axes_;i++)
	{
		filter_size *= kernel_shape_data[i];
	}
	const int input_channels = channels_;
	
	std::vector<int> col_buf_shape;
	col_buf_shape.push_back(input_channels * filter_size);
	col_buf_shape.push_back(max_defined_size);
	col_buffer_.Reshape(col_buf_shape);
	//memset(col_buffer_.mutable_cpu_data(), 0, sizeof(Dtype)*input_channels * filter_size * max_defined_size);

	//also reshape the out col buf
	const int out_channels = num_output_;
	std::vector<int> out_colBuf_shape;
	out_colBuf_shape.push_back(out_channels);
	out_colBuf_shape.push_back(max_defined_size);
	out_col_buffer_.Reshape(out_colBuf_shape);
	//memset(out_col_buffer_.mutable_cpu_data(), 0, sizeof(Dtype)*out_channels * max_defined_size);

	//also reshape the bias multiplier
	if (bias_term_) 
	{
		vector<int> bias_multiplier_shape(1, max_defined_size);
		bias_multiplier_.Reshape(bias_multiplier_shape);
		switch (Caffe::mode())
		{
		default:
			break;
		case Caffe::CPU:
			caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
			break;
		case Caffe::GPU:
			caffe_gpu_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_gpu_data());
			break;
		}
	}
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::reshape_weight_bias(const vector<Blob<Dtype> *>& bottom)
{
	//only reshape init once
	if (m_weight_bias_inited_flag)
	{
		return;
	}
	m_weight_bias_inited_flag = true;


	const ConvHashParameter &conv_param = this->layer_param_.conv_hash_param();
	const int* kernel_shape_data = kernel_shape_.cpu_data();

	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	CHECK_GT(channels_, 0);
	num_output_ = conv_param.num_output();
	CHECK_GT(num_output_, 0);
	if (reverse_dimensions()) {
		conv_out_channels_ = channels_;
		conv_in_channels_ = num_output_;
	}
	else {
		conv_out_channels_ = num_output_;
		conv_in_channels_ = channels_;
	}
	// Handle the parameters: weights and biases.
	// - blobs_[0] holds the filter weights
	// - blobs_[1] holds the biases (optional)
	vector<int> weight_shape(2);
	weight_shape[0] = conv_out_channels_;
	weight_shape[1] = conv_in_channels_;
	for (int i = 0; i < num_spatial_axes_; ++i) {
		weight_shape.push_back(kernel_shape_data[i]);
	}
	bias_term_ = conv_param.bias_term();
	vector<int> bias_shape(bias_term_, num_output_);

	//NOTE: blobs_ must be resized  in setup, as the net.cpp will compare the blob num and param num
	//Though resized, the blobs is not allocated
/*	if (this->blobs_.size() > 0) {
		CHECK_EQ(1 + bias_term_, this->blobs_.size())
			<< "Incorrect number of weight blobs.";
		if (this->blobs_[0]->count() && weight_shape != this->blobs_[0]->shape()) {
			Blob<Dtype> weight_shaped_blob(weight_shape);
			LOG(FATAL) << "Incorrect weight shape: expected shape "
				<< weight_shaped_blob.shape_string() << "; instead, shape was "
				<< this->blobs_[0]->shape_string();
		}
		if (bias_term_ && this->blobs_[1]->count() && bias_shape != this->blobs_[1]->shape()) {
			Blob<Dtype> bias_shaped_blob(bias_shape);
			LOG(FATAL) << "Incorrect bias shape: expected shape "
				<< bias_shaped_blob.shape_string() << "; instead, shape was "
				<< this->blobs_[1]->shape_string();
		}
		LOG(INFO) << "Skipping parameter initialization";
	}
	else*/ 
	{
		//if (bias_term_) {
		//	this->blobs_.resize(2);
		//}
		//else {
		//	this->blobs_.resize(1);
		//}
		// Initialize and fill the weights:
		// output channels x input channels per-group x kernel height x kernel width
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
			conv_param.weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());

#if 0	//for debug
		caffe_set(this->blobs_[0]->count(), Dtype(1), this->blobs_[0]->mutable_cpu_data());
		//const int filter_dim = kernel_shape_data[0] * kernel_shape_data[1] * kernel_shape_data[2];
		//int center_idx = filter_dim / 2;
		//caffe_set(this->blobs_[0]->count(), Dtype(0), this->blobs_[0]->mutable_cpu_data());
		//for (int row = 0; row < conv_out_channels_; row++)
		//{
		//	int idx = row * conv_in_channels_ * filter_dim;
		//	for (int j = 0; j < conv_in_channels_; j++)
		//	{
		//		this->blobs_[0]->mutable_cpu_data()[center_idx + j*filter_dim + idx] = 1.f;
		//	}
		//}
#endif

		// If necessary, initialize and fill the biases.
		if (bias_term_) {
			this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
				conv_param.bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());
		}
	}
	kernel_dim_ = this->blobs_[0]->count(1);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::forward_cpu_gemm(const float *bottom_hash, const unsigned char *bottom_offset,
	const PACKED_POSITION *bottom_posTag, int m_bar, int r_bar,
	int bottom_channels, int top_channels, int defined_voxel_num, int dense_res, float *out_col_buf)
{
	const int *kernel_shape = kernel_shape_.cpu_data();
	const ConvHashParameter &conv_param = this->layer_param_.conv_hash_param();
	conv_hash2col_cpu(bottom_hash, bottom_offset, bottom_posTag, kernel_shape,
		m_bar, r_bar, bottom_channels,
		defined_voxel_num, dense_res, (float*)col_buffer_.mutable_cpu_data());

#if DUMP_2_TXT	
	int rows = bottom_channels*kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
	int cols = defined_voxel_num;
	std::ofstream ofs("col_buf.txt");
	const float *buf_ptr = (const float*)col_buffer_.cpu_data();
	for (int col = 0; col < cols; col++)
	{
		const float *cur_row_ptr = buf_ptr;
		ofs << "col " << col << ":" << std::endl;
		for (int row = 0; row < rows; row++)
		{
			ofs << *cur_row_ptr << " ";
			cur_row_ptr += cols;
		}
		buf_ptr++;
		ofs << std::endl;
	}
	ofs.close();

	ofs.open("weight.txt");
	const float *weight_ptr = (const float*)this->blobs_[0]->cpu_data();
	for (int col = 0; col < rows; col++)
	{
		const float *cur_row_ptr = weight_ptr;
		ofs << "col " << col << ":" << std::endl;
		for (int row = 0; row < top_channels; row++)
		{
			ofs << *cur_row_ptr << " ";
			cur_row_ptr += rows;
		}
		ofs << std::endl;
	}
	ofs.close();
#endif

	const int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
	caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, top_channels, defined_voxel_num, bottom_channels*kernel_size,
		1.f, (const float*)this->blobs_[0]->cpu_data(), (const float*)col_buffer_.cpu_data(),
		0.f, out_col_buf);

#if DUMP_2_TXT 	//NOTE: should be modified. Now we convert to row-major
	ofs.open("outColBuf.txt");
	const float *outbuf_ptr = out_col_buf;
	for (int col = 0; col < cols; col++)
	{
		const float *cur_row_ptr = outbuf_ptr;
		ofs << "col " << col << ":" << std::endl;
		for (int row = 0; row < top_channels; row++)
		{
			ofs << *cur_row_ptr << " ";
			cur_row_ptr += cols;
		}
		outbuf_ptr++;
		ofs << std::endl;
	}
	ofs.close();
#endif

	//convert out col buf to top
	//conv_col2hash_cpu(bottom_hash, top_hash, m_bar, bottom_channels, top_channels, defined_voxel_num, out_col_buffer_.mutable_cpu_data());
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::forward_cpu_bias(float* out_col_buf, const float* bias, int defined_voxel_num) 
{
  caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, num_output_,
	  defined_voxel_num, 1, 1.f, bias, (const float*)bias_multiplier_.cpu_data(),
      1.f, out_col_buf);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::backward_cpu_gemm(const float *out_col_buf,
	int bottom_channels, int top_channels, int defined_voxel_num, float *col_buf)
{
	const int *kernel_shape = kernel_shape_.cpu_data();
	const int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];

	////convert the top dif to out col buffer
	//top_hash2col_cpu(top_hash_dif, top_posTag,
	//	m_bar, top_channels,
	//	defined_voxel_num, (float*)out_col_buffer_.mutable_cpu_data());

	//delta_x = WT * delta_y
    caffe_cpu_gemm<float>(CblasTrans, CblasNoTrans, bottom_channels*kernel_size,
        defined_voxel_num, top_channels,
        1.f, (const float*)this->blobs_[0]->cpu_data(), out_col_buf,
        0.f, col_buf);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::weight_cpu_gemm(const float *col_buf, const float* out_col_buf, float *weight_dif,
	int bottom_channels, int top_channels, int defined_voxel_num)
{
	const int *kernel_shape = kernel_shape_.cpu_data();
	const int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
    
	caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, top_channels,
		bottom_channels*kernel_size, defined_voxel_num,
		1.f, out_col_buf, col_buf,
		1.f, weight_dif);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::backward_cpu_bias(float* bias, const float* out_col_buf, int defined_voxel_num)
{
	caffe_cpu_gemv<float>(CblasNoTrans, num_output_, defined_voxel_num, 1.,
		out_col_buf, (const float*)bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvHashLayer<Dtype>::forward_gpu_gemm(const float *bottom_hash, const unsigned char *bottom_offset,
	const PACKED_POSITION *bottom_posTag, const int* bottom_validPos, int m_bar, int r_bar,
	int bottom_channels, int top_channels, int defined_voxel_num, int dense_res, float *out_col_buf) 
{
	const int *kernel_shape = kernel_shape_.cpu_data();
	const ConvHashParameter &conv_param = this->layer_param_.conv_hash_param();
	conv_hash2col_gpu(bottom_hash, bottom_offset, bottom_posTag, bottom_validPos, 
		kernel_shape, m_bar, r_bar, bottom_channels,
		defined_voxel_num, dense_res, (float*)col_buffer_.mutable_gpu_data());

	const int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
	caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, top_channels, defined_voxel_num, bottom_channels*kernel_size,
		1.f, (const float*)this->blobs_[0]->gpu_data(), (const float*)col_buffer_.gpu_data(),
		0.f, out_col_buf);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::forward_gpu_bias(float* out_col_buf, const float* bias, int defined_voxel_num) 
{
	caffe_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, num_output_,
		defined_voxel_num, 1, 1.f, bias, (const float*)bias_multiplier_.gpu_data(),
		1.f, out_col_buf);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::backward_gpu_gemm(const float *out_col_buf,
	int bottom_channels, int top_channels, int defined_voxel_num, float *col_buf)
{
	const int *kernel_shape = kernel_shape_.cpu_data();
	const int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];

	//delta_x = WT * delta_y
	caffe_gpu_gemm<float>(CblasTrans, CblasNoTrans, bottom_channels*kernel_size,
		defined_voxel_num, top_channels,
		1.f, (const float*)this->blobs_[0]->gpu_data(), out_col_buf,
		0.f, col_buf);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::weight_gpu_gemm(const float *col_buf, const float* out_col_buf, float *weight_dif,
	int bottom_channels, int top_channels, int defined_voxel_num)
{
	const int *kernel_shape = kernel_shape_.cpu_data();
	const int kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];

	caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans, top_channels,
		bottom_channels*kernel_size, defined_voxel_num,
		1.f, out_col_buf, col_buf,
		1.f, weight_dif);
}

template <typename Dtype>
void BaseConvHashLayer<Dtype>::backward_gpu_bias(float* bias, const float* out_col_buf, int defined_voxel_num)
{
	caffe_gpu_gemv<float>(CblasNoTrans, num_output_, defined_voxel_num, 1.,
		out_col_buf, (const float*)bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
int BaseConvHashLayer<Dtype>::writeDenseKernel_2_HF5(const char *filename) 
{
	if (blobs_.size()<1)
	{
		return 0;
	}
	writeDense_2_HF5((const float*)blobs_[0]->cpu_data(), 
		blobs_[0]->shape(0), blobs_[0]->shape(2), blobs_[0]->shape(1), filename);
	return 1;
}

template <typename Dtype>
int BaseConvHashLayer<Dtype>::writeBias_2_HF5(const char *filename)
{
	if (blobs_.size() < 2)
	{
		return 0;
	}
	writeDense_2_HF5((const float*)blobs_[1]->cpu_data(),
		1, 1, blobs_[1]->shape(0), filename);
	return 1;
}



template <typename Dtype>
int BaseConvHashLayer<Dtype>::writeDenseKernelDif_2_HF5(const char *filename)
{
	if (blobs_.size() < 1)
	{
		return 0;
	}
	writeDense_2_HF5((const float*)blobs_[0]->cpu_diff(),
		blobs_[0]->shape(0), blobs_[0]->shape(2), blobs_[0]->shape(1), filename);
	return 1;
}

template <typename Dtype>
int BaseConvHashLayer<Dtype>::writeBiasDif_2_HF5(const char *filename)
{
	if (blobs_.size() < 2)
	{
		return 0;
	}
	writeDense_2_HF5((const float*)blobs_[1]->cpu_diff(),
		1, 1, blobs_[1]->shape(0), filename);
	return 1;
}


#endif  // !CPU_ONLY



/********************************************************************************/

/******************************************************************************/
//NOTE: the matrix is row-major
int conv_hash2col_cpu(const float* hash_data, const unsigned char *offset_data, const PACKED_POSITION *position_tags,
	const int kernel_shape[3],	//D, H, W
	int m_bar, int r_bar, int channels, int defined_num,
	int dense_res, float* col_buff)
{
	//col is reception field: input_channels * KD * KH * KW; row: spatial domain
	const int m = m_bar * m_bar * m_bar;
	//const float *data_ptr = hash_data;
	float *col_ptr = col_buff;

	const int hx = kernel_shape[0] / 2;
	const int hy = kernel_shape[1] / 2;
	const int hz = kernel_shape[2] / 2;
	const int kernel_dim = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
	const int r2 = r_bar * r_bar;
	const int m2 = m_bar * m_bar;
	const int rows = kernel_dim * channels;
	const int cols = defined_num;
	//for pointer increment
	const int cross_channel_stride = cols * kernel_dim;


	//init vals to zero
	memset(col_buff, 0, sizeof(float)*cols * rows);

	int counter = 0;	//for debug
	for (int v = 0; v < m; v++)
	{
		//if the hash voxel is undefined, skip
		if (!ishashVoxelDefined(&position_tags[v]))
		{
			//data_ptr++;
			continue;
		}
		counter++;
		///////////////////////////////////////////

		//get the real voxel position from the position tag
		int cx, cy, cz;
		xyz_from_pack(position_tags[v], cx, cy, cz);	//get the voxel position mapped to this hash
														///////////////////////////////////////////////

														//loop over neighbors to fill the column
		int min_x = cx - hx;
		int min_y = cy - hy;
		int min_z = cz - hz;
		int mx, my, mz;
		float *cur_row = col_ptr;
		for (int nz = min_z; nz < min_z + kernel_shape[2]; nz++)
		{
			for (int ny = min_y; ny < min_y + kernel_shape[1]; ny++)
			{
				for (int nx = min_x; nx < min_x + kernel_shape[0]; nx++)
				{
					if (nx < 0 || ny < 0 || nz < 0 || nx >= dense_res || ny >= dense_res || nz >= dense_res)
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}
					//hash to get hash position
					Hash(nx, ny, nz, mx, my, mz,
						offset_data, m_bar, r_bar, r2);
					const int m_idx = NXYZ2I(mx, my, mz, m_bar, m2);

					if (!ishashVoxelDefined(&position_tags[m_idx]))	//the hash voxel is undefined
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}

					int stored_x, stored_y, stored_z;
					xyz_from_pack(position_tags[m_idx], stored_x, stored_y, stored_z);
					if (nx != stored_x || ny != stored_y || nz != stored_z)	//the neighbor is an undefined voxel
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}

					//fill the value at cur_row and corresponding channels
					const float *hash_ptr = &hash_data[m_idx];
					float *fill_row_ptr = cur_row;
					for (int c = 0; c < channels; c++)
					{
						*fill_row_ptr = *hash_ptr;
						fill_row_ptr += cross_channel_stride;
						hash_ptr += m;
					}
					cur_row += cols;
				}
			}
		}


		col_ptr++;
		//data_ptr++;
	}

	if (counter != defined_num)
	{
		printf("Fatal error: defined num not match!\n");
		exit(0);
	}

	//printf("col buffer size <%d, %d>\n", rows, counter);


	return 1;
}

//NOTE: the matrix is row major
int conv_col2hash_cpu(const PACKED_POSITION *pos_tags, float *out_hash_data,
	int m_bar, int out_channels, int defined_num, const float* col_buff)
{
	//col is reception field: input_channels * KD * KH * KW; row: spatial domain
	const int m = m_bar * m_bar * m_bar;
	float *out_data_ptr = out_hash_data;
	const float *col_ptr = col_buff;
	const int cols = defined_num;
	//to be safe, init out to zero
	memset(out_hash_data, 0, sizeof(float)*m*out_channels);

	for (int v = 0; v < m; v++)
	{
		//if the hash voxel is undefined, skip
		if (!ishashVoxelDefined(&pos_tags[v]))
		{
			out_data_ptr++;
			continue;
		}

		float *cur_out_ptr = out_data_ptr;
		const float *cur_row_ptr = col_ptr;
		for (int c = 0; c < out_channels; c++)
		{
			*cur_out_ptr = *cur_row_ptr;
			cur_out_ptr += m;
			cur_row_ptr += cols;
		}

		col_ptr++;
		out_data_ptr++;
	}
	return 1;
}


//used for BP, convert the top (dif) to col
int top_hash2col_cpu(const float *hash_data, const PACKED_POSITION *pos_tags,
	int m_bar, int out_channels, int defined_num, float* col_buff)
{
	//col is reception field: input_channels * KD * KH * KW; row: spatial domain
	const int m = m_bar * m_bar * m_bar;
	const float *data_ptr = hash_data;
	float *col_ptr = col_buff;
	const int cols = defined_num;
	//to be safe, init out to zero
	memset(col_buff, 0, sizeof(float)*defined_num*out_channels);

	for (int v = 0; v < m; v++)
	{
		//if the hash voxel is undefined, skip
		if (!ishashVoxelDefined(&pos_tags[v]))
		{
			data_ptr++;
			continue;
		}

		const float *cur_data_ptr = data_ptr;
		float *cur_row_ptr = col_ptr;
		for (int c = 0; c < out_channels; c++)
		{
			*cur_row_ptr = *cur_data_ptr;
			cur_data_ptr += m;
			cur_row_ptr += cols;
		}

		col_ptr++;
		data_ptr++;
	}
	return 1;
}


int bottom_col2hash_cpu(float* hash_data, const unsigned char *offset_data, const PACKED_POSITION *position_tags,
	const int kernel_shape[3],	//D, H, W
	int m_bar, int r_bar, int channels, int defined_num,
	int dense_res, const float* col_buff)
{
	//col is reception field: input_channels * KD * KH * KW; row: spatial domain
	const int m = m_bar * m_bar * m_bar;
	//const float *data_ptr = hash_data;
	const float *col_ptr = col_buff;

	const int hx = kernel_shape[0] / 2;
	const int hy = kernel_shape[1] / 2;
	const int hz = kernel_shape[2] / 2;
	const int kernel_dim = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
	const int r2 = r_bar * r_bar;
	const int m2 = m_bar * m_bar;
	const int rows = kernel_dim * channels;
	const int cols = defined_num;
	//for pointer increment
	const int cross_channel_stride = cols * kernel_dim;

	//init hash vals to zero
	memset(hash_data, 0, sizeof(float)*m * channels);

	int counter = 0;	//for debug
	for (int v = 0; v < m; v++)
	{
		//if the hash voxel is undefined, skip
		if (!ishashVoxelDefined(&position_tags[v]))
		{
			//data_ptr++;
			continue;
		}
		counter++;
		///////////////////////////////////////////

		//get the real voxel position from the position tag
		int cx, cy, cz;
		xyz_from_pack(position_tags[v], cx, cy, cz);	//get the voxel position mapped to this hash
														///////////////////////////////////////////////

														//loop over neighbors to fill the column
		int min_x = cx - hx;
		int min_y = cy - hy;
		int min_z = cz - hz;
		int mx, my, mz;
		const float *cur_row = col_ptr;
		for (int nz = min_z; nz < min_z + kernel_shape[2]; nz++)
		{
			for (int ny = min_y; ny < min_y + kernel_shape[1]; ny++)
			{
				for (int nx = min_x; nx < min_x + kernel_shape[0]; nx++)
				{
					if (nx < 0 || ny < 0 || nz < 0 || nx >= dense_res || ny >= dense_res || nz >= dense_res)
					{
						//just skip, as the values are inited as zeros
						cur_row += cols;
						continue;
					}
					//hash to get hash position
					Hash(nx, ny, nz, mx, my, mz,
						offset_data, m_bar, r_bar, r2);
					const int m_idx = NXYZ2I(mx, my, mz, m_bar, m2);

					if (!ishashVoxelDefined(&position_tags[m_idx]))	//the hash voxel is undefined
					{
						//just skip
						cur_row += cols;
						continue;
					}

					int stored_x, stored_y, stored_z;
					xyz_from_pack(position_tags[m_idx], stored_x, stored_y, stored_z);
					if (nx != stored_x || ny != stored_y || nz != stored_z)	//the neighbor is an undefined voxel
					{
						//just skip
						cur_row += cols;
						continue;
					}

					//accumulate the value at cur_row and corresponding channels to the hash data
					float *hash_ptr = &hash_data[m_idx];
					const float *fill_row_ptr = cur_row;
					for (int c = 0; c < channels; c++)
					{
						*hash_ptr += *fill_row_ptr;	//accumulate the value to the hash
						fill_row_ptr += cross_channel_stride;
						hash_ptr += m;
					}
					cur_row += cols;
				}
			}
		}


		col_ptr++;
		//data_ptr++;
	}

	if (counter != defined_num)
	{
		printf("Fatal error: defined num not match!\n");
		exit(0);
	}

	//printf("col buffer size <%d, %d>\n", rows, counter);


	return 1;
}

INSTANTIATE_CLASS(BaseConvHashLayer);



}  // namespace caffe
