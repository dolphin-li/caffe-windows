#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  cudnn::createTensorNdDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorNdDesc<Dtype>(&top_desc_);
  cudnn::createPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->num_spatial_axes_, this->kernel_shape_.cpu_data(), 
	  this->pad_shape_.cpu_data(), this->stride_shape_.cpu_data());
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(channel_axis_, 1);
  CHECK_EQ(top.size(), 1);	// cudnn conv layer does not support top_mask for MAX POOLING tracking
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->num_axes(), bottom[0]->shape().data());
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, top[0]->num_axes(), top[0]->shape().data());
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
}


template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	Forward_gpu(bottom, top);
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	Backward_gpu(top, propagate_down,bottom);
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif
