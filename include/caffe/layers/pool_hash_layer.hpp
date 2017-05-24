#ifndef CAFFE_POOL_HASH_LAYER_HPP_
#define CAFFE_POOL_HASH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/HashData.h"

namespace caffe {

  /**
  * @brief Pools the input image by taking the max, average, etc. within regions.
  *
  * TODO(dox): thorough documentation for Forward, Backward, and proto params.
  */

  template <typename Dtype>
  class PoolHashLayer : public Layer<Dtype> {
  public:
    explicit PoolHashLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "PoolHash"; }
    virtual inline int MinTopBlobs() const { return 1; }
    //// MAX POOL layers can output an extra top blob for the mask;
    //// others can only output the pooled inputs.
    //virtual inline int MaxTopBlobs() const {
    //  return (this->layer_param_.pooling_param().pool() ==
    //          PoolingParameter_PoolMethod_MAX) ? 2 : 1;
    //}
  protected:
	  void reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  void Forward_cpu_max(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  void forward_cpu_max(const float *bottom_hash, const unsigned char *bottom_offset,
		  const PACKED_POSITION *bottom_posTag, int bottom_m_bar, int bottom_r_bar,
		  float *top_hash, const unsigned char *top_offset,
		  const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar, 
		  int *mask, int channels, int dense_res);
	  void Forward_gpu_max(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  void forward_gpu_max(const float *bottom_hash, const unsigned char *bottom_offset,
		  const PACKED_POSITION *bottom_posTag, int bottom_m_bar, int bottom_r_bar,
		  float *top_hash, const unsigned char *top_offset,
		  const PACKED_POSITION *top_posTag, int top_m_bar, int top_r_bar,
		  int *mask, int channels, int dense_res);
	  void Backward_cpu_max(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  void backward_cpu_max(float *bottom_dif, int bottom_m_bar,
		  const float *top_dif, const PACKED_POSITION *top_posTag, int top_m_bar, 
		  const int *mask, int channels);
	  void Backward_gpu_max(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  void backward_gpu_max(float *bottom_dif, int bottom_m_bar,
		  const float *top_dif, const PACKED_POSITION *top_posTag, int top_m_bar,
		  const int *mask, int channels);
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<int> kernel_shape_;
	Blob<int> stride_shape_;
	Blob<int> pad_shape_;
	
	Blob<int> max_idx_;	//For max pooling: record the max input idx for each output valid voxel

	int num_spatial_axes_;	//3 in 3D case
	int channels_;
  };

}  // namespace caffe

//for debug
void bp_max_dense(const float *top_dif, const int *top_mask, float *bottom_dif, int top_res, int bottom_res, int channels);

#endif  // CAFFE_POOLING_LAYER_HPP_
