#ifndef CAFFE_HASH_DATA_LAYER_HPP_
#define CAFFE_HASH_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/HashData.h"
namespace caffe {

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HashDataLayer : public Layer<Dtype> {
 public:
  explicit HashDataLayer(const LayerParameter& param)
	  : Layer<Dtype>(param), offset_() { channels_ = 0; }
  virtual ~HashDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "HashData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
protected:	//for hash data
	//transfer a batch hases to blobs
	//void hashes_2_blobs(const std::vector<HashData> &hases, const std::vector<unsigned int> &batch_perm,
	//	const std::vector<Blob<Dtype>*>& top);
	void HierHashes_2_blobs(const std::vector<CHierarchyHash *> &vpHierHashes, const std::vector<unsigned int> &batch_perm,
			const std::vector<Blob<Dtype>*>& top);
public:
	//for debug
	void save_blobs_to_hashFiles(const std::vector<Blob<Dtype>*>& top, const char *main_body);
 protected:
  void Next();
  bool Skip();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
protected:
  void LoadHashLabelFileData(const char* filename);
  bool Load_labels(FILE *fp);


  std::vector<std::string> hash_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  unsigned int current_row_;
  Blob<Dtype>  label_blob_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;
  uint64_t offset_;

  //record the input channels 
  unsigned int channels_;
  //hash
public:
	void destroyHierHashes();
	int loadHierHashes(FILE *fp);
  //std::vector<HashData> m_hashes;
  std::vector<CHierarchyHash *> m_vpHierHashes;
  std::vector<unsigned int> m_batch_perm;	//batch permutation
};

}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_LAYER_HPP_
