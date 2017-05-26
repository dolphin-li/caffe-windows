#ifndef CAFFE_HASH2DENSE_LAYER_HPP_
#define CAFFE_HASH2DENSE_LAYER_HPP_

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
  class Hash2DenseLayer : public Layer<Dtype> {
  public:
    explicit Hash2DenseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Hash2Dense"; }
    virtual inline int MinTopBlobs() const { return 1; }
    //// MAX POOL layers can output an extra top blob for the mask;
    //// others can only output the pooled inputs.
    //virtual inline int MaxTopBlobs() const {
    //  return (this->layer_param_.pooling_param().pool() ==
    //          PoolingParameter_PoolMethod_MAX) ? 2 : 1;
    //}
  protected:
	  void reshape_top(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int num_spatial_axes_;	//3 in 3D case
	int channels_;
  };

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
