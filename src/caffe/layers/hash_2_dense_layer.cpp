#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hash_2_dense_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void Hash2DenseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	//CHECK size
	if (bottom.size() != HASH_DATA_SIZE + HASH_STRUCTURE_SIZE)	//data + input struct
	{
		printf("Fatal error: bottom size should be %d\n", HASH_DATA_SIZE + HASH_STRUCTURE_SIZE);
		exit(0);
	}
	if (top.size() != 1)
	{
		printf("Fatal error: top size should be 1\n");
		exit(0);
	}


	num_spatial_axes_ = 3;	//for 3D case
}



template <typename Dtype>
void Hash2DenseLayer<Dtype>::reshape_top(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	// Configure output channels and groups.
	channels_ = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	CHECK_GT(channels_, 0);

	const Blob<Dtype> *bottom_m_bar_blob = bottom[M_BAR_BLOB];
	if (!bottom_m_bar_blob->num_axes())
	{
		printf("*************Data not transferred. cannot reshape topHashData!\n**********");
		exit(0);
		return;
	}
	const int batch_num = bottom_m_bar_blob->shape(0);
	
	const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];

	std::vector<int> top_shape;
	top_shape.push_back(batch_num);
	top_shape.push_back(channels_);
	for (int i=0;i<num_spatial_axes_;i++)
	{
		top_shape.push_back(dense_res);
	}
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void Hash2DenseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	reshape_top(bottom, top);
}

template <typename Dtype>
void Hash2DenseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	const float *bt_hash = (const float*)bottom[HASH_DATA_BLOB]->cpu_data();
	const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
	const int channel_spatial_dim = channels_ * dense_res * dense_res * dense_res;

	float *dense_buf = (float*)top[0]->mutable_cpu_data();
	for (int i = 0; i < batch_num; ++i)
	{
		const float* cur_bt_hash = bt_hash;
		const unsigned char* cur_bt_offset = bt_offset;
		const PACKED_POSITION *cur_bt_postag = bt_posTag;
		const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];

		float *cur_dense_buf = dense_buf;

		hash_2_dense(cur_bt_hash, cur_bt_postag, cur_bt_offset, bt_m_bar,
			bt_r_bar, channels_, cur_dense_buf, dense_res);

#if 1
		//debug
		char buf[128];
		sprintf(buf, "H2D_top_%d.hf5", i);
		writeDense_2_HF5(cur_dense_buf, 1, dense_res, channels_, buf);
#endif

		//to next hash
		const int bt_m = bt_m_bar * bt_m_bar * bt_m_bar;
		const int bt_r = bt_r_bar * bt_r_bar * bt_r_bar;
		bt_hash += bt_m * channels_;
		bt_offset += bt_r * 3;
		bt_posTag += bt_m;
		//to next dense
		dense_buf += channel_spatial_dim;
	}
}


template <typename Dtype>
void Hash2DenseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	float *bt_hash = (float*)bottom[HASH_DATA_BLOB]->mutable_cpu_diff();
	const unsigned char*bt_offset = (const unsigned char *)bottom[OFFSET_BLOB]->cpu_data();
	const PACKED_POSITION *bt_posTag = (const PACKED_POSITION *)bottom[POSTAG_BLOB]->cpu_data();
	const int batch_num = bottom[M_BAR_BLOB]->shape(0);
	const int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
	const int channel_spatial_dim = channels_ * dense_res * dense_res * dense_res;

	const float *dense_buf = (const float*)top[0]->cpu_diff();
	for (int i = 0; i < batch_num; ++i)
	{
		float* cur_bt_hash = bt_hash;
		const unsigned char* cur_bt_offset = bt_offset;
		const PACKED_POSITION *cur_bt_postag = bt_posTag;
		const int bt_m_bar = (int)bottom[M_BAR_BLOB]->cpu_data()[i];
		const int bt_r_bar = (int)bottom[R_BAR_BLOB]->cpu_data()[i];

		const float *cur_dense_buf = dense_buf;

		dense_2_hash(cur_bt_hash, cur_bt_postag, cur_bt_offset, bt_m_bar,
			bt_r_bar, channels_, cur_dense_buf, dense_res);


#if 0
		float *ttt = new float[channels_*dense_res*dense_res*dense_res];
		hash_2_dense(cur_bt_hash, cur_bt_postag, cur_bt_offset, bt_m_bar,
			bt_r_bar, channels_, ttt, dense_res);
		for (int dd=0;dd<channels_*dense_res*dense_res*dense_res;dd++)
		{
			if (ttt[dd]!=cur_dense_buf[dd])
			{
				printf("Fatal error!\n");
			}
		}
#endif


		//to next hash
		const int bt_m = bt_m_bar * bt_m_bar * bt_m_bar;
		const int bt_r = bt_r_bar * bt_r_bar * bt_r_bar;
		bt_hash += bt_m * channels_;
		bt_offset += bt_r * 3;
		bt_posTag += bt_m;
		//to next dense
		dense_buf += channel_spatial_dim;
	}
}

#ifdef CPU_ONLY
STUB_GPU(Hash2DenseLayer);
#endif


template <typename Dtype>
void Hash2DenseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void Hash2DenseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

}

INSTANTIATE_CLASS(Hash2DenseLayer);

}  // namespace caffe
