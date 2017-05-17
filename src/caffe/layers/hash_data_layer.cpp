#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "stdint.h"

#include "caffe/layers/hash_data_layer.hpp"
#include "caffe/util/MyMacro.h"


namespace caffe {

template <typename Dtype>
HashDataLayer<Dtype>::~HashDataLayer<Dtype>() { destroyHashes(m_hashes); }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HashDataLayer<Dtype>::LoadHashLabelFileData(const char* filename) 
{
  //NOTE: we dont destroy the memory to avoid the memory cost; We only enlarge the memory if needed
  //destroyHashes(m_hashes);
  DLOG(INFO) << "Loading HashLabel file: " << filename;

  FILE *fp = fopen(filename,"rb");
  if (!fp)
  {
	  printf("Error: failed to load hash labels from %s\n", filename);
	  exit(0);
  }
  if (!loadHashes(m_hashes, fp))
  {
	  printf("Error: failed to load hashes from %s\n",filename);
	  fclose(fp);
	  exit(0);
  }
  if (!Load_labels(fp))
  {
	  printf("Error: failed to load labels from %s\n", filename);
	  fclose(fp);
	  exit(0);
  }

  fclose(fp);

  CHECK_EQ((int)m_hashes.size(), label_blobs_[0]->shape(0)) << "hash num should == label num.";

  // Default to identity permutation.
  if (data_permutation_.size()!=m_hashes.size())
  {
	  data_permutation_.resize(m_hashes.size());
  }
  for (int i = 0; i < (int)m_hashes.size(); i++)
	  data_permutation_[i] = i;
 

  // Shuffle if needed.
  if (this->layer_param_.hash_data_param().shuffle()) {
	  std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
	  DLOG(INFO) << "Successfully loaded " << m_hashes.size()
		  << " hashes (shuffled)";
  }
  else {
	  DLOG(INFO) << "Successfully loaded " << m_hashes.size() << " hashes";
  }
}

template <typename Dtype>
bool HashDataLayer<Dtype>::Load_labels(FILE *fp)
{
	int hash_n;
	fread(&hash_n, sizeof(int), 1, fp);
	int label_size;
	fread(&label_size, sizeof(int), 1, fp);

	if (label_size < 1 || hash_n<1)
	{
		printf("Error: invalid hash num %d or label size %d\n", hash_n, label_size);
		return false;
	}

	std::vector<int> label_shape;
	label_shape.push_back(hash_n);
	label_shape.push_back(label_size);
	label_blobs_[0]->Reshape(label_shape);
	fread(label_blobs_[0]->mutable_cpu_data(),sizeof(Dtype),hash_n*label_size,fp);

	return true;
}

template <typename Dtype>
void HashDataLayer<Dtype>::hashes_2_blobs(const std::vector<HashData> &vHashes,
	const std::vector<unsigned int> &batch_perm, const std::vector<Blob<Dtype>*>& top_blobs)
{
	//reshape blob
	if (top_blobs.size() < HASH_MIN_BLOB_NUM + 1)
	{
		printf("Error: hash data layer: top size should be larger than %d\n", HASH_MIN_BLOB_NUM);
		exit(0);
	}


	//blobs: hash data, offset data, position tags, m_bars, r_bars, defNums, channel_num

	//channel
	std::vector<int> channel_shape(1, 1);
	top_blobs[CHANNEL_BLOB]->Reshape(channel_shape);
	const int channels = vHashes[0].m_channels;
	top_blobs[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)channels;

	//m_bar, r_bar, define_num
	std::vector<int> structure_shape(1, batch_perm.size());
	top_blobs[M_BAR_BLOB]->Reshape(structure_shape);
	top_blobs[R_BAR_BLOB]->Reshape(structure_shape);
	top_blobs[DEFNUM_BLOB]->Reshape(structure_shape);
	
	//gather the total size of m, and record each m && r
	int batch_hash_size = 0;
	int batch_offset_size = 0;
	int m_bar, r_bar, defNum;
	for (int i = 0; i < (int)batch_perm.size(); i++)
	{
		const int idx = batch_perm[i];
		m_bar = vHashes[idx].m_mBar;
		r_bar = vHashes[idx].m_rBar;
		defNum = vHashes[idx].m_defNum;

		top_blobs[M_BAR_BLOB]->mutable_cpu_data()[i] = (Dtype)m_bar;
		top_blobs[R_BAR_BLOB]->mutable_cpu_data()[i] = (Dtype)r_bar;
		top_blobs[DEFNUM_BLOB]->mutable_cpu_data()[i] = (Dtype)defNum;

		batch_hash_size += m_bar * m_bar * m_bar;
		batch_offset_size += r_bar * r_bar * r_bar;
	}

	printf("Batch hash size is %d\n", batch_hash_size);
	printf("Batch offset size is %d\n", batch_offset_size);

	//hash data
	const int hashdata_size_f = (batch_hash_size * channels * sizeof(float)) / sizeof(Dtype) + 1;
	std::vector<int> hash_data_shape(1, hashdata_size_f);
	top_blobs[HASH_DATA_BLOB]->Reshape(hash_data_shape);
	//offset data, we have to convert 3*r char to float
	const int offset_size_f = (batch_offset_size * 3 * sizeof(unsigned char)) / sizeof(Dtype) + 1;
	std::vector<int> offset_shape(1, offset_size_f);
	top_blobs[OFFSET_BLOB]->Reshape(offset_shape);
	//position tag, all convert PACKED_POSITION to float
	const int posTag_size_f = (batch_hash_size * sizeof(PACKED_POSITION)) / sizeof(Dtype) + 1;
	std::vector<int> posTag_shape(1, posTag_size_f);
	top_blobs[POSTAG_BLOB]->Reshape(posTag_shape);

	float *batch_hash_ptr = (float*)top_blobs[HASH_DATA_BLOB]->mutable_cpu_data();
	unsigned char *batch_offset_ptr = (unsigned char *)top_blobs[OFFSET_BLOB]->mutable_cpu_data();
	PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)top_blobs[POSTAG_BLOB]->mutable_cpu_data();
	int m, r;	//m_bar*m_bar*m_bar
	for (int i = 0; i < (int)batch_perm.size(); i++)
	{
		const int idx = batch_perm[i];
		m_bar = vHashes[idx].m_mBar;
		r_bar = vHashes[idx].m_rBar;
		m = m_bar*m_bar*m_bar;
		r = r_bar*r_bar*r_bar;

		memcpy(batch_hash_ptr, vHashes[idx].m_hash_data, sizeof(float)*m*channels);
		memcpy(batch_offset_ptr, vHashes[idx].m_offset_data, sizeof(unsigned char)*r * 3);
		memcpy(batch_posTag_ptr, vHashes[idx].m_position_tag, sizeof(PACKED_POSITION)*m);

		batch_hash_ptr += m*channels;
		batch_offset_ptr += r * 3;
		batch_posTag_ptr += m;
	}
}

template <typename Dtype>
void HashDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hash_data_param().source();
  LOG(INFO) << "Loading list of HashLabel filenames from: " << source;
  hash_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
		hash_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hash_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HashLabel files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HashLabel filename listed in "
    << source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hash_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HashLabel file and initialize the line counter.
  LoadHashLabelFileData(hash_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  //init batch permutation
  const int batch_num = this->layer_param_.hash_data_param().batch_size();
  m_batch_perm.resize(batch_num);
  for (int i=0;i<batch_num;i++)
  {
	  m_batch_perm[i] = data_permutation_[i];
  }
  //NOTE: we have to convert hashes to blobs here. As the following layer need the shape information of its bottom
  hashes_2_blobs(m_hashes, m_batch_perm, top);
  //reshape the top label, as the other blobs' shape will change every batch
 
  vector<int> top_label_shape;
  top_label_shape.resize(label_blobs_[0]->num_axes());
  top_label_shape[0] = batch_num;
  for (int j = 1; j < top_label_shape.size(); ++j) 
  {
	  top_label_shape[j] = label_blobs_[0]->shape(j);
  }
  top[LABEL_BLOB]->Reshape(top_label_shape);
}

template <typename Dtype>
bool HashDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void HashDataLayer<Dtype>::Next() 
{
  if (++current_row_ == (int)m_hashes.size()) {
    if (num_files_ > 1) {
      ++current_file_;
      if (current_file_ == num_files_) {
        current_file_ = 0;
        if (this->layer_param_.hash_data_param().shuffle()) {
          std::random_shuffle(file_permutation_.begin(),
                              file_permutation_.end());
        }
        DLOG(INFO) << "Looping around to first file.";
      }
      LoadHashLabelFileData(
        hash_filenames_[file_permutation_[current_file_]].c_str());
    }
    current_row_ = 0;
    if (this->layer_param_.hash_data_param().shuffle())
      std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
  }
  offset_++;
}

template <typename Dtype>
void HashDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hash_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i) {
    while (Skip()) {
      Next();
    }

	//NOTE: currently we simply record current_row_ as m_batch_perm[i].
	//It has a problem: when loading a new file during the for-loop, 
	//previously recorded m_batch_germ[i] will now refer to the new hashes.
	//Thus some tail files in the last file will be thrown away
  
	m_batch_perm[i] = data_permutation_[current_row_];
	
	//int data_dim = top[j]->count() / top[j]->shape(0);
	//caffe_copy(data_dim,
	//	&hash_blobs_[j]->cpu_data()[data_permutation_[current_row_]
	//	* data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    
    Next();
  }

  hashes_2_blobs(m_hashes, m_batch_perm, top);
  //send labels
  for (int i = 0; i < batch_size; ++i)
  {
	  int data_dim = top[LABEL_BLOB]->count() / top[LABEL_BLOB]->shape(0);
	  caffe_copy(data_dim,
		  &label_blobs_[0]->cpu_data()[m_batch_perm[i]
		  * data_dim], &top[LABEL_BLOB]->mutable_cpu_data()[i * data_dim]);
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HashDataLayer, Forward);
#endif

INSTANTIATE_CLASS(HashDataLayer);
REGISTER_LAYER_CLASS(HashData);

}  // namespace caffe
