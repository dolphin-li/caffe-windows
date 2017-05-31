#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "stdint.h"

#include "caffe/layers/hash_data_layer.hpp"
#include "caffe/util/MyMacro.h"


namespace caffe {

template <typename Dtype>
HashDataLayer<Dtype>::~HashDataLayer<Dtype>() 
{
	destroyHierHashes();
}

template <typename Dtype>
void HashDataLayer<Dtype>::destroyHierHashes() 
{
	for (int i=0;i<(int)m_vpHierHashes.size();i++)
	{
		delete m_vpHierHashes[i];
	}
	m_vpHierHashes.resize(0);
}

template <typename Dtype>
int HashDataLayer<Dtype>::loadHierHashes(FILE *fp)
{
	int n;
	fread(&n, sizeof(int), 1, fp);

	if (n<=0)
	{
		printf("Fatal error: hier hash num <= 0!\n");
		exit(0);
	}

	//actually when load batches, n will always to the same
	int ori_n = (int)m_vpHierHashes.size();
	if (ori_n<n)
	{
		for (int i=ori_n;i<n;i++)
		{
			m_vpHierHashes.push_back(new CHierarchyHash());
		}
	}
	if (ori_n>n)
	{
		for (int i = ori_n-1; i >= n; i--)
		{
			delete m_vpHierHashes[i];
			m_vpHierHashes.pop_back();
		}
	}

	for (int i = 0; i < n; i++)
	{
		m_vpHierHashes[i]->load(fp);
	}
	return true;
}


template <typename Dtype>
bool HashDataLayer<Dtype>::Load_labels(FILE *fp)
{
	int hash_n;
	fread(&hash_n, sizeof(int), 1, fp);
	int label_size;
	fread(&label_size, sizeof(int), 1, fp);

	if (label_size < 1 || hash_n < 1)
	{
		printf("Error: invalid hash num %d or label size %d\n", hash_n, label_size);
		return false;
	}

	std::vector<int> label_shape;
	label_shape.push_back(hash_n);
	label_shape.push_back(label_size);
	label_blob_.Reshape(label_shape);
	fread(label_blob_.mutable_cpu_data(), sizeof(Dtype), hash_n*label_size, fp);

	return true;
}


// Load data and label from Hash filename into the class property blobs.
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
  if (!loadHierHashes(fp))
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

  CHECK_EQ((int)m_vpHierHashes.size(), label_blob_.shape(0)) << "hash num should == label num.";

  // Default to identity permutation.
  if (data_permutation_.size()!=m_vpHierHashes.size())
  {
	  data_permutation_.resize(m_vpHierHashes.size());
  }
  for (int i = 0; i < (int)m_vpHierHashes.size(); i++)
	  data_permutation_[i] = i;
 

  // Shuffle if needed.
  if (this->layer_param_.hash_data_param().shuffle()) {
	  std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
	  DLOG(INFO) << "Successfully loaded " << m_vpHierHashes.size()
		  << " hashes (shuffled)";
  }
  else {
	  DLOG(INFO) << "Successfully loaded " << m_vpHierHashes.size() << " hashes";
  }
}

//template <typename Dtype>
//void HashDataLayer<Dtype>::hashes_2_blobs(const std::vector<HashData> &vHashes,
//	const std::vector<unsigned int> &batch_perm, const std::vector<Blob<Dtype>*>& top_blobs)
//{
//	//reshape blob
//	if (top_blobs.size() < HASH_MIN_BLOB_NUM + 1)
//	{
//		printf("Error: hash data layer: top size should be larger than %d\n", HASH_MIN_BLOB_NUM);
//		exit(0);
//	}
//
//
//	//blobs: hash data, offset data, position tags, m_bars, r_bars, defNums, channel_num
//
//	//channel
//	std::vector<int> channel_shape(1, 1);
//	top_blobs[CHANNEL_BLOB]->Reshape(channel_shape);
//	const int channels = vHashes[0].m_channels;
//	top_blobs[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)channels;
//
//	//m_bar, r_bar, define_num
//	std::vector<int> structure_shape(1, batch_perm.size());
//	top_blobs[M_BAR_BLOB]->Reshape(structure_shape);
//	top_blobs[R_BAR_BLOB]->Reshape(structure_shape);
//	top_blobs[DEFNUM_BLOB]->Reshape(structure_shape);
//	
//	//gather the total size of m, and record each m && r
//	int batch_hash_size = 0;
//	int batch_offset_size = 0;
//	int m_bar, r_bar, defNum;
//	for (int i = 0; i < (int)batch_perm.size(); i++)
//	{
//		const int idx = batch_perm[i];
//		m_bar = vHashes[idx].m_mBar;
//		r_bar = vHashes[idx].m_rBar;
//		defNum = vHashes[idx].m_defNum;
//
//		top_blobs[M_BAR_BLOB]->mutable_cpu_data()[i] = (Dtype)m_bar;
//		top_blobs[R_BAR_BLOB]->mutable_cpu_data()[i] = (Dtype)r_bar;
//		top_blobs[DEFNUM_BLOB]->mutable_cpu_data()[i] = (Dtype)defNum;
//
//		batch_hash_size += m_bar * m_bar * m_bar;
//		batch_offset_size += r_bar * r_bar * r_bar;
//	}
//
//	printf("Batch hash size is %d\n", batch_hash_size);
//	printf("Batch offset size is %d\n", batch_offset_size);
//
//	//hash data
//	const int hashdata_size_f = (batch_hash_size * channels * sizeof(float)) / sizeof(Dtype) + 1;
//	std::vector<int> hash_data_shape(1, hashdata_size_f);
//	top_blobs[HASH_DATA_BLOB]->Reshape(hash_data_shape);
//	//offset data, we have to convert 3*r char to float
//	const int offset_size_f = (batch_offset_size * 3 * sizeof(unsigned char)) / sizeof(Dtype) + 1;
//	std::vector<int> offset_shape(1, offset_size_f);
//	top_blobs[OFFSET_BLOB]->Reshape(offset_shape);
//	//position tag, all convert PACKED_POSITION to float
//	const int posTag_size_f = (batch_hash_size * sizeof(PACKED_POSITION)) / sizeof(Dtype) + 1;
//	std::vector<int> posTag_shape(1, posTag_size_f);
//	top_blobs[POSTAG_BLOB]->Reshape(posTag_shape);
//
//	float *batch_hash_ptr = (float*)top_blobs[HASH_DATA_BLOB]->mutable_cpu_data();
//	unsigned char *batch_offset_ptr = (unsigned char *)top_blobs[OFFSET_BLOB]->mutable_cpu_data();
//	PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)top_blobs[POSTAG_BLOB]->mutable_cpu_data();
//	int m, r;	//m_bar*m_bar*m_bar
//	for (int i = 0; i < (int)batch_perm.size(); i++)
//	{
//		const int idx = batch_perm[i];
//		m_bar = vHashes[idx].m_mBar;
//		r_bar = vHashes[idx].m_rBar;
//		m = m_bar*m_bar*m_bar;
//		r = r_bar*r_bar*r_bar;
//
//		memcpy(batch_hash_ptr, vHashes[idx].m_hash_data, sizeof(float)*m*channels);
//		memcpy(batch_offset_ptr, vHashes[idx].m_offset_data, sizeof(unsigned char)*r * 3);
//		memcpy(batch_posTag_ptr, vHashes[idx].m_position_tag, sizeof(PACKED_POSITION)*m);
//
//		batch_hash_ptr += m*channels;
//		batch_offset_ptr += r * 3;
//		batch_posTag_ptr += m;
//	}
//}


template <typename Dtype>
void HashDataLayer<Dtype>::HierHashes_2_blobs(const std::vector<CHierarchyHash *> &vpHierHashes,
	const std::vector<unsigned int> &batch_perm, const std::vector<Blob<Dtype>*>& top_blobs)
{
	if (!vpHierHashes.size())
	{
		printf("Fatal error: HierHashes_2_blobs failed! no hierHashes!\n");
		exit(0);
	}
	const int struct_num = (int)vpHierHashes[0]->m_vpStructs.size();
	if (!struct_num)
	{
		printf("Fatal error: HierHashes_2_blobs failed! no structures!\n");
		exit(0);
	}
	const int channels = vpHierHashes[0]->m_channels;
	const int bottom_dense_res = vpHierHashes[0]->m_dense_res;
	//record the channels
	channels_ = channels;

	if (channels<1)
	{
		printf("Fatal error: HierHashes_2_blobs failed! no channels!\n");
		exit(0);
	}

	//reshape blob
	if (top_blobs.size() != HASH_DATA_SIZE + struct_num * HASH_STRUCTURE_SIZE + 1)	//data + n*struct + label
		//structure info for multi layers + one hash data + one label
	{
		printf("Error: hash data layer: top blobs should be have %d structus and 1 hash data!\n",struct_num);
		exit(0);
	}

	//*****NOTE: the blob order is <data, <structures>, label>***********/

	//blobs: hash data, < offset data, position tags, m_bars, r_bars, defNums >, label
	std::vector<int> batch_shape(1, batch_perm.size());
	//reshape m_bar, r_bar, define_num
	for (int i=0;i<struct_num;i++)
	{
		top_blobs[M_BAR_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[R_BAR_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[DEFNUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
	}
	
	std::vector<int> scalar_shape(1, 1);	//the blob only has one scalar value
	for (int i = 0; i < struct_num; i++)	
	{
		top_blobs[DENSE_RES_BLOB]->Reshape(scalar_shape);
		top_blobs[CHANNEL_BLOB]->Reshape(scalar_shape);
	}
	top_blobs[DENSE_RES_BLOB ]->mutable_cpu_data()[0] = (Dtype)(bottom_dense_res);
	top_blobs[CHANNEL_BLOB]->mutable_cpu_data()[0] = (Dtype)channels_;
	
	//fill the value of m_bar, r_bar, define_num, and gather the total size for batched offset and pos_tag, 
	//gather the total size of m, and record each m && r, offset, postag, and bottom_hash_data
	for (int si = 0; si < struct_num; si++)
	{
		Blob<Dtype>* offset_blob = top_blobs[OFFSET_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* postag_blob = top_blobs[POSTAG_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* mBar_blob = top_blobs[M_BAR_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* rBar_blob = top_blobs[R_BAR_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* defNum_blob = top_blobs[DEFNUM_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* validPos_blob = top_blobs[VALID_POS_BLOB + si * HASH_STRUCTURE_SIZE];

		int batch_m = 0;
		int batch_r = 0;
		int m_bar, r_bar, defNum;
		int m, r;	//m_bar*m_bar*m_bar, r_bar*r_bar*r_bar
		for (int j=0 ; j<(int)batch_perm.size();j++)
		{
			const int idx = batch_perm[j];
			m_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_mBar;
			r_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_rBar;
			defNum = m_vpHierHashes[idx]->m_vpStructs[si]->m_defNum;

			mBar_blob->mutable_cpu_data()[j] = (Dtype)m_bar;
			rBar_blob->mutable_cpu_data()[j] = (Dtype)r_bar;
			defNum_blob->mutable_cpu_data()[j] = (Dtype)defNum;

			batch_m += m_bar * m_bar * m_bar;
			batch_r += r_bar * r_bar * r_bar;
		}

		//fill the offset and posTag
		const int offset_size_f = (batch_r * 3 * sizeof(unsigned char)) / sizeof(Dtype) + 1;
		std::vector<int> offset_shape(1, offset_size_f);
		offset_blob->Reshape(offset_shape);
		//position tag, all convert PACKED_POSITION to float
		const int posTag_size_f = (batch_m * sizeof(PACKED_POSITION)) / sizeof(Dtype) + 1;
		std::vector<int> posTag_shape(1, posTag_size_f);
		postag_blob->Reshape(posTag_shape);
		//valid position tag, should convert to int, we store it the same size as posTag, 
		// but its actual size should be defNum
		const int validPos_size_f = (batch_m * sizeof(int)) / sizeof(Dtype) + 1;
		std::vector<int> validPos_shape(1, validPos_size_f);
		validPos_blob->Reshape(validPos_shape);

		unsigned char *batch_offset_ptr = (unsigned char *)offset_blob->mutable_cpu_data();
		PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)postag_blob->mutable_cpu_data();
		int* batch_validPos_ptr = (int*)validPos_blob->mutable_cpu_data();
		
		for (int j = 0; j < (int)batch_perm.size(); j++)
		{
			const int idx = batch_perm[j];
			m_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_mBar;
			r_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_rBar;
			m = m_bar*m_bar*m_bar;
			r = r_bar*r_bar*r_bar;

			memcpy(batch_offset_ptr, m_vpHierHashes[idx]->m_vpStructs[si]->m_offset_data, sizeof(unsigned char)*r * 3);
			memcpy(batch_posTag_ptr, m_vpHierHashes[idx]->m_vpStructs[si]->m_position_tag, sizeof(PACKED_POSITION)*m);
			getValidPoses(batch_posTag_ptr, batch_validPos_ptr, m);
			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
			batch_validPos_ptr += m;
		}


		//fill the bottom hash data
		if (si == 0)	//bottom hash
		{
			Blob<Dtype>* hash_data_blob = top_blobs[HASH_DATA_BLOB];
			const int hashdata_size_f = (batch_m * channels * sizeof(float)) / sizeof(Dtype) + 1;
			std::vector<int> hash_data_shape(1, hashdata_size_f);
			hash_data_blob->Reshape(hash_data_shape);

			float *batch_hash_ptr = (float*)hash_data_blob->mutable_cpu_data();
			for (int j = 0; j < (int)batch_perm.size(); j++)
			{
				const int idx = batch_perm[j];
				m_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_mBar;
				m = m_bar*m_bar*m_bar;

				memcpy(batch_hash_ptr, m_vpHierHashes[idx]->m_hash_data, sizeof(float)*m*channels);
				batch_hash_ptr += m*channels;
			}
		}
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
  
  //HierHashes_2_blobs(m_vpHierHashes, m_batch_perm, top);
  //reshape the top label, as the other blobs' shape will change every batch
 
  vector<int> top_label_shape;
  top_label_shape.resize(label_blob_.num_axes());
  top_label_shape[0] = batch_num;
  for (int j = 1; j < top_label_shape.size(); ++j) 
  {
	  top_label_shape[j] = label_blob_.shape(j);
  }
  printf("Label shape : \n");
  for (int i=0;i<(int)top_label_shape.size();i++)
  {
	  printf("%d ", top_label_shape[i]);
  }
  printf("\n");
  
  const int structure_num = m_vpHierHashes[0]->m_vpStructs.size();
  int label_blob_idx = HASH_DATA_SIZE + HASH_STRUCTURE_SIZE * structure_num;
  top[label_blob_idx]->Reshape(top_label_shape);
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
  if (++current_row_ == (int)m_vpHierHashes.size()) {
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

  HierHashes_2_blobs(m_vpHierHashes, m_batch_perm, top);

#if DUMP_2_TXT//for debug
  save_blobs_to_hashFiles(top, "test_hash_data_layer");
#endif

  //send labels
  const int structure_num = m_vpHierHashes[0]->m_vpStructs.size();
  int label_blob_idx = HASH_DATA_SIZE + HASH_STRUCTURE_SIZE * structure_num;
  for (int i = 0; i < batch_size; ++i)
  {
	  int data_dim = top[label_blob_idx]->count() / top[label_blob_idx]->shape(0);
	  caffe_copy(data_dim,
		  &label_blob_.cpu_data()[m_batch_perm[i]
		  * data_dim], &top[label_blob_idx]->mutable_cpu_data()[i * data_dim]);
  }
}

template <typename Dtype>
void HashDataLayer<Dtype>::save_blobs_to_hashFiles(const std::vector<Blob<Dtype>*>& top_blobs, const char *main_body)
{
	int top_num = (int)top_blobs.size();
	const int structure_num = m_vpHierHashes[0]->m_vpStructs.size();
	const int dense_res = m_vpHierHashes[0]->m_dense_res;
	if (top_num!=HASH_DATA_SIZE + structure_num * HASH_STRUCTURE_SIZE + 1)
	{
		printf("Hash data layer error: top num not expected!\n");
		exit(0);
	}
	char buf[128];

	//save other structures
	for (int si = 1; si < structure_num; si++)
	{
		Blob<Dtype>* offset_blob = top_blobs[OFFSET_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* postag_blob = top_blobs[POSTAG_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* mBar_blob = top_blobs[M_BAR_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* rBar_blob = top_blobs[R_BAR_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* defNum_blob = top_blobs[DEFNUM_BLOB + si * HASH_STRUCTURE_SIZE];

		int m, r;	//m_bar*m_bar*m_bar, r_bar*r_bar*r_bar
		unsigned char *batch_offset_ptr = (unsigned char *)offset_blob->cpu_data();
		PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)postag_blob->cpu_data();

		for (int j = 0; j < (int)m_batch_perm.size(); j++)
		{
			const int idx = m_batch_perm[j];

			sprintf(buf, "%s_%d_%d.st", main_body, idx, si);
			

			HashData one_hash;
			one_hash.m_mBar = (int)mBar_blob->cpu_data()[j];
			one_hash.m_rBar = (int)rBar_blob->cpu_data()[j];
			one_hash.m_defNum = (int)defNum_blob->cpu_data()[j];
			one_hash.m_offset_data = batch_offset_ptr;
			one_hash.m_position_tag = batch_posTag_ptr;
			one_hash.m_channels = channels_;
			one_hash.m_dense_res = dense_res;
			one_hash.m_hash_data = NULL;
			saveHashStruct(one_hash, buf);

			m = one_hash.m_mBar*one_hash.m_mBar*one_hash.m_mBar;
			r = one_hash.m_rBar*one_hash.m_rBar*one_hash.m_rBar;
			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
		}
	}

	//save bottom hash data and struct
	int si = 0;
	{
		Blob<Dtype>* offset_blob = top_blobs[OFFSET_BLOB];
		Blob<Dtype>* postag_blob = top_blobs[POSTAG_BLOB];
		Blob<Dtype>* mBar_blob = top_blobs[M_BAR_BLOB];
		Blob<Dtype>* rBar_blob = top_blobs[R_BAR_BLOB];
		Blob<Dtype>* defNum_blob = top_blobs[DEFNUM_BLOB];
		Blob<Dtype>* hashdata_blob = top_blobs[HASH_DATA_BLOB];

		int m, r;	//m_bar*m_bar*m_bar, r_bar*r_bar*r_bar
		unsigned char *batch_offset_ptr = (unsigned char *)offset_blob->cpu_data();
		PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)postag_blob->cpu_data();
		float *batch_hash_ptr = (float*)hashdata_blob->cpu_data();
		for (int j = 0; j < (int)m_batch_perm.size(); j++)
		{
			const int idx = m_batch_perm[j];

			sprintf(buf, "%s_%d_%d.psh", main_body, idx, si);


			HashData one_hash;
			one_hash.m_mBar = (int)mBar_blob->cpu_data()[j];
			one_hash.m_rBar = (int)rBar_blob->cpu_data()[j];
			one_hash.m_defNum = (int)defNum_blob->cpu_data()[j];
			one_hash.m_offset_data = batch_offset_ptr;
			one_hash.m_position_tag = batch_posTag_ptr;
			one_hash.m_channels = channels_;
			one_hash.m_dense_res = dense_res;
			one_hash.m_hash_data = batch_hash_ptr;
			saveHash(one_hash, buf);

			m = one_hash.m_mBar*one_hash.m_mBar*one_hash.m_mBar;
			r = one_hash.m_rBar*one_hash.m_rBar*one_hash.m_rBar;
			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
			batch_hash_ptr += m*channels_;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HashDataLayer, Forward);
#endif

template <typename Dtype>
void HashDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(HashDataLayer);
REGISTER_LAYER_CLASS(HashData);

}  // namespace caffe
