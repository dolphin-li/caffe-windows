#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "stdint.h"

#include "caffe/layers/hash_data_layer.hpp"
#include "caffe/util/MyMacro.h"
#include <boost/thread.hpp>

#define MULT_THREAD 1
#define USE_BATCH_HASH 1
namespace caffe {

	template <typename Dtype>
	HashDataLayer<Dtype>::HashDataLayer(const LayerParameter& param)
		: Layer<Dtype>(param), offset_(), prefetch_(param.data_param().prefetch()),
		prefetch_free_(), prefetch_full_(), prefetch_current_(NULL) 
	{
		channels_ = 0; 
#if MULT_THREAD
		for (int i = 0; i < prefetch_.size(); ++i) 
		{
			prefetch_[i].reset(new GeneralBatch<Dtype>());
			prefetch_free_.push(prefetch_[i].get());
		}
#endif
	}

template <typename Dtype>
HashDataLayer<Dtype>::~HashDataLayer<Dtype>() 
{
#if MULT_THREAD
	this->StopInternalThread();
#endif
	destroyHierHashes();
	destroyBatch();
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
void HashDataLayer<Dtype>::destroyBatch()
{
	for (int i = 0; i<(int)m_curBatchHash.size(); i++)
	{
		delete m_curBatchHash[i];
	}
	m_curBatchHash.resize(0);
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
	std::vector<int> batch_shape_1(1, batch_perm.size()+1);
	//reshape m_bar, r_bar, define_num
	for (int i=0;i<struct_num;i++)
	{
		top_blobs[M_BAR_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[R_BAR_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[DEFNUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[DEFNUM_SUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape_1);
		top_blobs[M_SUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape_1);
		top_blobs[R_SUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape_1);
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
		Blob<Dtype>* volIdx_blob = top_blobs[VOLUME_IDX_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* defNumSum_blob = top_blobs[DEFNUM_SUM_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* mSum_blob = top_blobs[M_SUM_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* rSum_blob = top_blobs[R_SUM_BLOB + si * HASH_STRUCTURE_SIZE];

		int batch_m = 0;
		int batch_r = 0;
		int m_bar, r_bar, defNum;
		int m, r;	//m_bar*m_bar*m_bar, r_bar*r_bar*r_bar
		int totalDefNum = 0;
		for (int j=0 ; j<(int)batch_perm.size();j++)
		{
			const int idx = batch_perm[j];
			m_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_mBar;
			r_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_rBar;
			defNum = m_vpHierHashes[idx]->m_vpStructs[si]->m_defNum;

			mBar_blob->mutable_cpu_data()[j] = (Dtype)m_bar;
			rBar_blob->mutable_cpu_data()[j] = (Dtype)r_bar;
			defNum_blob->mutable_cpu_data()[j] = (Dtype)defNum;

			if (!defNum)
			{
				printf("Fatal error: defNum zero! cur row: %d, idx: %d\n",current_row_,idx);
				
				m_vpHierHashes[idx]->save("wrong_hierHash.hst");
				exit(0);
			}

			defNumSum_blob->mutable_cpu_data()[j] = (Dtype)totalDefNum;
			mSum_blob->mutable_cpu_data()[j] = (Dtype)batch_m;
			rSum_blob->mutable_cpu_data()[j] = (Dtype)batch_r;

			batch_m += m_bar * m_bar * m_bar;
			batch_r += r_bar * r_bar * r_bar;
			totalDefNum += defNum;
		}
		defNumSum_blob->mutable_cpu_data()[batch_perm.size()] = (Dtype)totalDefNum;
		mSum_blob->mutable_cpu_data()[batch_perm.size()] = (Dtype)batch_m;
		rSum_blob->mutable_cpu_data()[batch_perm.size()] = (Dtype)batch_r;

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
		const int validPos_size_f = (totalDefNum * sizeof(int)) / sizeof(Dtype) + 1;
		std::vector<int> validPos_shape(1, validPos_size_f);
		validPos_blob->Reshape(validPos_shape);
		//volume index of each valid voxel
		const int volIdx_size_f = (totalDefNum * sizeof(VolumeIndexType)) / sizeof(Dtype) + 1;
		std::vector<int> volIdx_shape(1, volIdx_size_f);
		volIdx_blob->Reshape(volIdx_shape);

		unsigned char *batch_offset_ptr = (unsigned char *)offset_blob->mutable_cpu_data();
		PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)postag_blob->mutable_cpu_data();
		int* batch_validPos_ptr = (int*)validPos_blob->mutable_cpu_data();
		VolumeIndexType* batch_volIdx_ptr = (VolumeIndexType*)volIdx_blob->mutable_cpu_data();
		
		for (int j = 0; j < (int)batch_perm.size(); j++)
		{
			const int idx = batch_perm[j];
			m_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_mBar;
			r_bar = m_vpHierHashes[idx]->m_vpStructs[si]->m_rBar;
			m = m_bar*m_bar*m_bar;
			r = r_bar*r_bar*r_bar;
			defNum = m_vpHierHashes[idx]->m_vpStructs[si]->m_defNum;

			memcpy(batch_offset_ptr, m_vpHierHashes[idx]->m_vpStructs[si]->m_offset_data, sizeof(unsigned char)*r * 3);
			memcpy(batch_posTag_ptr, m_vpHierHashes[idx]->m_vpStructs[si]->m_position_tag, sizeof(PACKED_POSITION)*m);
			getValidPoses(batch_posTag_ptr, batch_validPos_ptr, m);
			for (int tmp_i = 0; tmp_i < defNum; tmp_i++)
				batch_volIdx_ptr[tmp_i] = (VolumeIndexType)j;

			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
			batch_validPos_ptr += defNum;
			batch_volIdx_ptr += defNum;
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
void HashDataLayer<Dtype>::BatchHierHashes_2_blobs(const std::vector<CHierarchyHash *> &batch_hashes, const std::vector<Blob<Dtype>*>& top_blobs)
{
	const int batch_num = this->layer_param_.hash_data_param().batch_size();
	if (!batch_hashes.size()|| (int)batch_hashes.size()!=batch_num)
	{
		printf("Fatal error: BatchHierHashes_2_blobs failed! no hierHashes or batch size not match!\n");
		exit(0);
	}

	const int struct_num = (int)batch_hashes[0]->m_vpStructs.size();
	if (!struct_num)
	{
		printf("Fatal error: BatchHierHashes_2_blobs failed! no structures!\n");
		exit(0);
	}
	const int channels = batch_hashes[0]->m_channels;
	const int bottom_dense_res = batch_hashes[0]->m_dense_res;
	//record the channels
	channels_ = channels;

	if (channels<1)
	{
		printf("Fatal error: BatchHierHashes_2_blobs failed! no channels!\n");
		exit(0);
	}

	//reshape blob
	if (top_blobs.size() != HASH_DATA_SIZE + struct_num * HASH_STRUCTURE_SIZE + 1)	//data + n*struct + label
																					//structure info for multi layers + one hash data + one label
	{
		printf("Error: hash data layer: top blobs should be have %d structus and 1 hash data!\n", struct_num);
		exit(0);
	}

	//*****NOTE: the blob order is <data, <structures>, label>***********/

	//blobs: hash data, < offset data, position tags, m_bars, r_bars, defNums >, label
	std::vector<int> batch_shape(1, batch_num);
	std::vector<int> batch_shape_1(1, batch_num + 1);
	//reshape m_bar, r_bar, define_num
	for (int i = 0; i<struct_num; i++)
	{
		top_blobs[M_BAR_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[R_BAR_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[DEFNUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape);
		top_blobs[DEFNUM_SUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape_1);
		top_blobs[M_SUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape_1);
		top_blobs[R_SUM_BLOB + i * HASH_STRUCTURE_SIZE]->Reshape(batch_shape_1);
	}

	std::vector<int> scalar_shape(1, 1);	//the blob only has one scalar value
	for (int i = 0; i < struct_num; i++)
	{
		top_blobs[DENSE_RES_BLOB]->Reshape(scalar_shape);
		top_blobs[CHANNEL_BLOB]->Reshape(scalar_shape);
	}
	top_blobs[DENSE_RES_BLOB]->mutable_cpu_data()[0] = (Dtype)(bottom_dense_res);
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
		Blob<Dtype>* volIdx_blob = top_blobs[VOLUME_IDX_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* defNumSum_blob = top_blobs[DEFNUM_SUM_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* mSum_blob = top_blobs[M_SUM_BLOB + si * HASH_STRUCTURE_SIZE];
		Blob<Dtype>* rSum_blob = top_blobs[R_SUM_BLOB + si * HASH_STRUCTURE_SIZE];

		int batch_m = 0;
		int batch_r = 0;
		int m_bar, r_bar, defNum;
		int m, r;	//m_bar*m_bar*m_bar, r_bar*r_bar*r_bar
		int totalDefNum = 0;
		for (int idx = 0; idx<batch_num; idx++)
		{
			m_bar = batch_hashes[idx]->m_vpStructs[si]->m_mBar;
			r_bar = batch_hashes[idx]->m_vpStructs[si]->m_rBar;
			defNum = batch_hashes[idx]->m_vpStructs[si]->m_defNum;

			mBar_blob->mutable_cpu_data()[idx] = (Dtype)m_bar;
			rBar_blob->mutable_cpu_data()[idx] = (Dtype)r_bar;
			defNum_blob->mutable_cpu_data()[idx] = (Dtype)defNum;

			if (!defNum)
			{
				printf("Fatal error: defNum zero! cur row: %d, idx: %d\n", current_row_, idx);
				batch_hashes[idx]->save("wrong_hierHash.hst");
				exit(0);
			}

			defNumSum_blob->mutable_cpu_data()[idx] = (Dtype)totalDefNum;
			mSum_blob->mutable_cpu_data()[idx] = (Dtype)batch_m;
			rSum_blob->mutable_cpu_data()[idx] = (Dtype)batch_r;

			batch_m += m_bar * m_bar * m_bar;
			batch_r += r_bar * r_bar * r_bar;
			totalDefNum += defNum;
		}
		defNumSum_blob->mutable_cpu_data()[batch_num] = (Dtype)totalDefNum;
		mSum_blob->mutable_cpu_data()[batch_num] = (Dtype)batch_m;
		rSum_blob->mutable_cpu_data()[batch_num] = (Dtype)batch_r;

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
		const int validPos_size_f = (totalDefNum * sizeof(int)) / sizeof(Dtype) + 1;
		std::vector<int> validPos_shape(1, validPos_size_f);
		validPos_blob->Reshape(validPos_shape);
		//volume index of each valid voxel
		const int volIdx_size_f = (totalDefNum * sizeof(VolumeIndexType)) / sizeof(Dtype) + 1;
		std::vector<int> volIdx_shape(1, volIdx_size_f);
		volIdx_blob->Reshape(volIdx_shape);

		unsigned char *batch_offset_ptr = (unsigned char *)offset_blob->mutable_cpu_data();
		PACKED_POSITION *batch_posTag_ptr = (PACKED_POSITION *)postag_blob->mutable_cpu_data();
		int* batch_validPos_ptr = (int*)validPos_blob->mutable_cpu_data();
		VolumeIndexType* batch_volIdx_ptr = (VolumeIndexType*)volIdx_blob->mutable_cpu_data();

		for (int idx = 0; idx < batch_num; idx++)
		{
			m_bar = batch_hashes[idx]->m_vpStructs[si]->m_mBar;
			r_bar = batch_hashes[idx]->m_vpStructs[si]->m_rBar;
			m = m_bar*m_bar*m_bar;
			r = r_bar*r_bar*r_bar;
			defNum = batch_hashes[idx]->m_vpStructs[si]->m_defNum;

			memcpy(batch_offset_ptr, batch_hashes[idx]->m_vpStructs[si]->m_offset_data, sizeof(unsigned char)*r * 3);
			memcpy(batch_posTag_ptr, batch_hashes[idx]->m_vpStructs[si]->m_position_tag, sizeof(PACKED_POSITION)*m);
			getValidPoses(batch_posTag_ptr, batch_validPos_ptr, m);
			for (int tmp_i = 0; tmp_i < defNum; tmp_i++)
				batch_volIdx_ptr[tmp_i] = (VolumeIndexType)idx;

			batch_offset_ptr += r * 3;
			batch_posTag_ptr += m;
			batch_validPos_ptr += defNum;
			batch_volIdx_ptr += defNum;
		}


		//fill the bottom hash data
		if (si == 0)	//bottom hash
		{
			Blob<Dtype>* hash_data_blob = top_blobs[HASH_DATA_BLOB];
			const int hashdata_size_f = (batch_m * channels * sizeof(float)) / sizeof(Dtype) + 1;
			std::vector<int> hash_data_shape(1, hashdata_size_f);
			hash_data_blob->Reshape(hash_data_shape);

			float *batch_hash_ptr = (float*)hash_data_blob->mutable_cpu_data();
			for (int idx = 0; idx < batch_num; idx++)
			{
				m_bar = batch_hashes[idx]->m_vpStructs[si]->m_mBar;
				m = m_bar*m_bar*m_bar;

				memcpy(batch_hash_ptr, batch_hashes[idx]->m_hash_data, sizeof(float)*m*channels);
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
#if 1 //added by tianjia, to solve the problem that the last few data may be lost when file switch
  destroyBatch();
  m_curBatchHash.resize(batch_num);
  for (int i = 0; i < batch_num; i++)
  {
	  m_curBatchHash[i] = new CHierarchyHash();
	  *m_curBatchHash[i] = *m_vpHierHashes[m_batch_perm[i]];
  }
#endif

  
  //send the first batch to tops in the setup stage, in order to allocate essential memories
#if USE_BATCH_HASH
  BatchHierHashes_2_blobs(m_curBatchHash, top);
#else//old, may lose last few data when file switches
  HierHashes_2_blobs(m_vpHierHashes, m_batch_perm, top);
#endif
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


#if MULT_THREAD
  /******added for multi thread******/
  const int top_blob_num = (int)top.size();
 for (int i=0;i<(int)prefetch_.size();i++)
 {
	 prefetch_[i]->init(top_blob_num);
	 for (int j = 0; j < top_blob_num; j++)
	 {
		 prefetch_[i]->blobs_[j]->ReshapeLike(*top[j]);
	 }
	 //prefetch_[i]->blobs_[label_blob_idx]->Reshape(top_label_shape);
 }
 // Before starting the prefetch thread, we make cpu_data and gpu_data
 // calls so that the prefetch thread does not accidentally make simultaneous
 // cudaMalloc calls when the main thread is running. In some GPUs this
 // seems to cause failures if we do not so.
 for (int i = 0; i < prefetch_.size(); ++i) 
 {
	 for (int j = 0; j < top_blob_num; j++)
	 {
		 prefetch_[i]->blobs_[j]->mutable_cpu_data();
	 }
 }
#ifndef CPU_ONLY
 if (Caffe::mode() == Caffe::GPU) {
	 for (int i = 0; i < prefetch_.size(); ++i)
	 {
		 for (int j = 0; j < top_blob_num; j++)
		 {
			 prefetch_[i]->blobs_[j]->mutable_gpu_data();
		 }
	 }
 }
#endif
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
#endif
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

#if MULT_THREAD
template <typename Dtype>
void HashDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if (prefetch_current_) {
		prefetch_free_.push(prefetch_current_);
	}
	prefetch_current_ = prefetch_full_.pop("Waiting for data");
	
	// Reshape to loaded data.
	for (int i=0;i<(int)top.size();i++)
	{
		top[i]->ReshapeLike(*prefetch_current_->blobs_[i]);
		top[i]->set_cpu_data(prefetch_current_->blobs_[i]->mutable_cpu_data());
	}
}
#else
/********************************OLD: single thread***********************************/

#if USE_BATCH_HASH
template <typename Dtype>
void HashDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	const int batch_size = this->layer_param_.hash_data_param().batch_size();
	const int structure_num = m_vpHierHashes[0]->m_vpStructs.size();
	const int label_blob_idx = HASH_DATA_SIZE + HASH_STRUCTURE_SIZE * structure_num;
	const int label_dim = top[label_blob_idx]->count() / top[label_blob_idx]->shape(0);
	for (int i = 0; i < batch_size; ++i) {
		while (Skip()) {
			Next();
		}

		//record batch hash, will be send to top at last
		*m_curBatchHash[i] = *m_vpHierHashes[data_permutation_[current_row_]];
		//send labels
		
		caffe_copy(label_dim,
			&label_blob_.cpu_data()[data_permutation_[current_row_]
			* label_dim], &top[label_blob_idx]->mutable_cpu_data()[i * label_dim]);

		Next();
	}

	BatchHierHashes_2_blobs(m_curBatchHash, top);
}
#else
template <typename Dtype>
void HashDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hash_data_param().batch_size();
  //current_row_ = 4096;
  //LOG(INFO) << "current file: " << current_file_ << " " << file_permutation_[current_file_]
	 // << "current_row_: " << current_row_;
  for (int i = 0; i < batch_size; ++i) {
    while (Skip()) {
      Next();
    }

	//NOTE: currently we simply record current_row_ as m_batch_perm[i].
	//It has a problem: when loading a new file during the for-loop, 
	//previously recorded m_batch_germ[i] will now refer to the new hashes.
	//Thus some tail files in the last file will be thrown away
  
#if 1	//NOTE: will cause bug here, when file switches, previously recorded m_batch_perm[i] is old index, which might > current size
	m_batch_perm[i] = data_permutation_[current_row_];
#endif
	//int data_dim = top[j]->count() / top[j]->shape(0);
	//caffe_copy(data_dim,
	//	&hash_blobs_[j]->cpu_data()[data_permutation_[current_row_]
	//	* data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    
    Next();
  }

#if 1	//fix the bug
  int hierhash_num = (int)(int)m_vpHierHashes.size();
  for (int i = 0; i < batch_size; ++i) 
  {
	if (m_batch_perm[i]>= hierhash_num)
	{
		m_batch_perm[i] = m_batch_perm[i] % hierhash_num;
	}
  }
#endif


  HierHashes_2_blobs(m_vpHierHashes, m_batch_perm, top);

#if 0//for debug
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
#endif
#endif

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



template <typename Dtype>
void HashDataLayer<Dtype>::InternalThreadEntry() 
{
	try {
		while (!must_stop()) {
			if (prefetch_free_.size())
			{
				GeneralBatch<Dtype>* batch = prefetch_free_.pop();
				fetch_batch(batch);
				prefetch_full_.push(batch);
			}
		}
	}
	catch (boost::thread_interrupted&) {
		// Interrupted exception is expected on shutdown
	}
}

template <typename Dtype>
void HashDataLayer<Dtype>::fetch_batch(GeneralBatch<Dtype>* batch)
{

	std::vector<Blob<Dtype>*> vpBlobs(batch->blobs_.size());
	for (int i = 0; i<(int)vpBlobs.size(); i++)
	{
		vpBlobs[i] = batch->blobs_[i].get();
	}


	const int batch_size = this->layer_param_.hash_data_param().batch_size();
	const int structure_num = m_vpHierHashes[0]->m_vpStructs.size();
	const int label_blob_idx = HASH_DATA_SIZE + HASH_STRUCTURE_SIZE * structure_num;
	const int label_dim = batch->blobs_[label_blob_idx]->count() / batch->blobs_[label_blob_idx]->shape(0);
	for (int i = 0; i < batch_size; ++i) 
	{
		while (Skip()) {
			Next();
		}

		//record batch hash, will be send to top at last
		*m_curBatchHash[i] = *m_vpHierHashes[data_permutation_[current_row_]];
		//send labels
		
		caffe_copy(label_dim,
			&label_blob_.cpu_data()[data_permutation_[current_row_]
			* label_dim], &batch->blobs_[label_blob_idx]->mutable_cpu_data()[i * label_dim]);

		Next();
	}

	BatchHierHashes_2_blobs(m_curBatchHash, vpBlobs);
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
