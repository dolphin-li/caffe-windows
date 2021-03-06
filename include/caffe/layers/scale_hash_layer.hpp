#ifndef CAFFE_SCALE_HASH_LAYER_HPP_
#define CAFFE_SCALE_HASH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class ScaleHashLayer: public Layer<Dtype> {
 public:
  explicit ScaleHashLayer(const LayerParameter& param)
      : Layer<Dtype>(param) { m_self_blob_init_flag = false;}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScaleHash"; }
  // Scale
  virtual inline int MinBottomBlobs() const { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> temp_, temp_bottom_;
  int channels_;
  int batch_hash_size_;
  bool m_self_blob_init_flag;
public:
	void backward_topDif2temp_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	void bottom2tempBottom_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	void tempBottom2BottomDif_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
protected:
	void init_self_blob(const vector<Blob<Dtype>*>& bottom);
	void reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
};


}  // namespace caffe

#endif  // CAFFE_SCALE_LAYER_HPP_
