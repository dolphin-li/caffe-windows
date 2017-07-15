#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_hash_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/MyMacro.h"
#include "caffe/util/HashData.h"
namespace caffe {

template <typename Dtype>
void ScaleHashLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


	//CHECK size
	if (bottom.size() != HASH_DATA_SIZE + HASH_STRUCTURE_SIZE)
	{
		printf("Fatal error: bottom size should be %d\n", HASH_DATA_SIZE + HASH_STRUCTURE_SIZE);
		exit(0);
	}
	if (top.size() != HASH_DATA_SIZE)
	{
		printf("Fatal error: top size should be %d\n", HASH_DATA_SIZE);
		exit(0);
	}


  const ScaleHashParameter& scale_hash_param = this->layer_param_.scale_hash_param();

  if (scale_hash_param.bias_term()) {
	  this->blobs_.resize(2);
  }
  else {
	  this->blobs_.resize(1);
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}



template <typename Dtype>
void ScaleHashLayer<Dtype>::init_self_blob(const vector<Blob<Dtype> *>& bottom)
{
	if (m_self_blob_init_flag)	//only init once
	{
		return;
	}
	m_self_blob_init_flag = true;

	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	CHECK_GT(channels_, 0);



	const ScaleHashParameter& param = this->layer_param_.scale_hash_param();
	
	// scale is a learned parameter; initialize it
	vector<int> scale_shape(1, channels_);
	this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
	FillerParameter filler_param(param.filler());
	if (!param.has_filler()) {
		// Default to unit (1) filler for identity operation.
		filler_param.set_type("constant");
		filler_param.set_value(1);
	}
	shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
	filler->Fill(this->blobs_[0].get());
	
	if (param.bias_term())
	{
		this->blobs_[1].reset(new Blob<Dtype>(scale_shape));
		FillerParameter filler_param(param.bias_filler());
		if (!param.has_bias_filler()) {
			// Default to unit (1) filler for identity operation.
			filler_param.set_type("constant");
			filler_param.set_value(0);
		}
		shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
		filler->Fill(this->blobs_[1].get());
	}
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}



template <typename Dtype>
void ScaleHashLayer<Dtype>::reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
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
}

template <typename Dtype>
void ScaleHashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

	// reshape temp to be channels * defNums
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	int total_defNum = 0;
	for (int i = 0; i < batch_num; i++)
		total_defNum += (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
	CHECK_GT(total_defNum, 0);

	//temp store valid hash data
	vector<int> sz2;
	sz2.push_back(channels_);
	sz2.push_back(total_defNum);
	temp_.Reshape(sz2);
	temp_bottom_.Reshape(sz2);

  sum_multiplier_.Reshape(vector<int>(1, total_defNum));
  if (sum_multiplier_.cpu_data()[total_defNum - 1] != Dtype(1)) {
    caffe_set(total_defNum, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ScaleHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  Dtype* scale_data = this->blobs_[0]->mutable_cpu_data();
  if (this->layer_param_.scale_hash_param().has_min_value()) 
  {
    for (int d = 0; d < channels_; d++) {
      scale_data[d] = std::max<Dtype>(scale_data[d], this->layer_param_.scale_hash_param().min_value());
    }
  }
  if (this->layer_param_.scale_hash_param().has_max_value()) {
    for (int d = 0; d < channels_; d++) {
      scale_data[d] = std::min<Dtype>(scale_data[d], this->layer_param_.scale_hash_param().max_value());
    }
  }


    //scale data
	float *tp_hash_ptr = (float*)top[HASH_DATA_BLOB]->mutable_cpu_data();
	const float *bt_hash_ptr = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
	const unsigned char *bt_offset_ptr = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *bt_posTag_ptr = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();

	//copy bottom to top
	memcpy(tp_hash_ptr,bt_hash_ptr,sizeof(float)*batch_hash_size_ * channels_);
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	for (int i = 0; i < batch_num; ++i)
	{
  		float* top_data = tp_hash_ptr;
  		const unsigned char* offset_data = bt_offset_ptr;
  		const PACKED_POSITION *pos_tags = bt_posTag_ptr;
  		const int m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
  		const int r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];
  		const int defNum = (int)bottom[DEFNUM_BLOB]->cpu_data()[i];
  
  		hash_mult_scalar(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, (const float*)this->blobs_[0]->cpu_data());
		if (this->layer_param_.scale_hash_param().bias_term())
		{

			hash_add_scalar(top_data, offset_data, pos_tags, m_bar, r_bar, channels_, defNum, (const float*)this->blobs_[1]->cpu_data());
		}
  
  		//to next hash
  		const int m = m_bar * m_bar * m_bar;
  		const int r = r_bar * r_bar * r_bar;
  
  		tp_hash_ptr += m * channels_;
  		bt_offset_ptr += r * 3;
  		bt_posTag_ptr += m;
	}
}

template <typename Dtype>
void ScaleHashLayer<Dtype>::backward_topDif2temp_cpu(const vector<Blob<Dtype>*>& bottom,
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
void ScaleHashLayer<Dtype>::bottom2tempBottom_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const float *hash = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	Dtype* tempBottom = temp_bottom_.mutable_cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = temp_bottom_.shape(1);
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

		Dtype *cur_tmp = tempBottom;

		for (int v = 0; v<m; v++)
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
		offsets += r * 3;
		posTags += m;

		tempBottom += def_num;
	}
}


template <typename Dtype>
void ScaleHashLayer<Dtype>::tempBottom2BottomDif_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	float *hash_dif = (float*)bottom[HASH_DATA_BLOB]->mutable_cpu_diff();
	const unsigned char *offsets = (const unsigned char*)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *posTags = (const PACKED_POSITION*)bottom[POSTAG_BLOB]->cpu_data();
	const Dtype* temp = temp_bottom_.cpu_data();
	const int batch_num = (int)bottom[M_BAR_BLOB]->shape(0);
	const int total_def_num = temp_bottom_.shape(1);
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
void ScaleHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	//convert top_dif to tmp
	backward_topDif2temp_cpu(bottom, top);
	const int total_def_num = temp_.shape(1);
	//bias bp
  if (this->layer_param_.scale_hash_param().bias_term() &&
      this->param_propagate_down_.back()) 
  {
	  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_, total_def_num, 1.,
		  temp_.cpu_data(), sum_multiplier_.cpu_data(), 0, this->blobs_[1]->mutable_cpu_diff());
  }
  //weight bp
  if (this->param_propagate_down_[0])
  {
	  //convert bottom to tmp_bottom
	  bottom2tempBottom_cpu(bottom, top);
	  const Dtype* top_diff = temp_.cpu_data();
	  const Dtype* bottom_data = temp_bottom_.cpu_data();
	  // Hack: store big eltwise product in bottom[0] diff, except in the special
	  // case where this layer itself does the eltwise product, in which case we
	  // can store it directly in the scale diff, and we're done.
	  // If we're computing in-place (and not doing eltwise computation), this
	  // hack doesn't work and we store the product in temp_.
	  Dtype* product = temp_bottom_.mutable_cpu_diff();
	  caffe_mul(temp_.count(), top_diff, bottom_data, product);

	  caffe_cpu_gemv(CblasNoTrans, channels_, total_def_num,
		  Dtype(1), product, sum_multiplier_.cpu_data(), Dtype(0),
		  this->blobs_[0]->mutable_cpu_diff());
  }
  else
  {
	  printf("Fatal error!\n");
  }
   
  //if (propagate_down[0])	//no use
  {
    const Dtype* top_diff = temp_.cpu_data();
    const Dtype* scale_data = this->blobs_[0]->cpu_data();

	//NOTE: now temp_bottom store the bottom diff
    Dtype* bottom_diff = temp_bottom_.mutable_cpu_data();
    
    for (int d = 0; d < channels_; ++d) 
	{
		const Dtype factor = scale_data[d];
		caffe_cpu_scale(total_def_num, factor, top_diff, bottom_diff);
		bottom_diff += total_def_num;
		top_diff += total_def_num;
    }
	tempBottom2BottomDif_cpu(bottom, top);
  }
}

template <typename Dtype>
void ScaleHashLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	Forward_cpu(bottom, top);
}


template <typename Dtype>
void ScaleHashLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	Backward_cpu(top, propagate_down, bottom);
}


#ifdef CPU_ONLY
STUB_GPU(ScaleHashLayer);
#endif

INSTANTIATE_CLASS(ScaleHashLayer);
REGISTER_LAYER_CLASS(ScaleHash);

}  // namespace caffe
