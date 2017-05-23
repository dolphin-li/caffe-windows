#ifndef CAFFE_BN_HASH_LAYER_HPP_
#define CAFFE_BN_HASH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Caffe, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BNHashLayer` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BNHashLayer : public Layer<Dtype> {
 public:
  explicit BNHashLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) { m_self_blob_init_flag = false; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BNHash"; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
protected:
	void init_self_blob(const vector<Blob<Dtype>*>& bottom);
	void reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	// convert data between hash and temp_;
	// data in temp_ are those valid voxels stored channel wise.
	void forward_hash2temp_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	void forward_temp2hash_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
protected:
  Blob<Dtype> mean_, variance_;
  bool use_global_stats_;
  Dtype moving_average_fraction_;
  Dtype eps_;
  bool disable_mean_;
  bool disable_variance_;
  int channels_;
  int batch_hash_size_;	//total m
  bool m_self_blob_init_flag;
  Blob<Dtype> temp_, temp2_, inv_sqrt_var_;
  Blob<Dtype> mean_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
