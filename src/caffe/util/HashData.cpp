#include "caffe/util/HashData.h"
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "caffe/util/MyMacro.h"

#include "hdf5.h"
#include "hdf5_hl.h"

#pragma comment(lib, "hdf5_hl.lib")
#pragma comment(lib, "hdf5.lib")

//NOTE: the data is organized as channel * depth * height * weight
//int getDefinedVoxelNum(const float *hash_data, int m_bar, int channels)
//{
//	const int m = m_bar * m_bar * m_bar;
//	int defined_num = 0;
//	const float *data_ptr = hash_data;
//	for (int v = 0; v < m; v++)
//	{
//		if (ishashVoxelDefined(data_ptr, channels, m))
//		{
//			defined_num++;
//		}
//		data_ptr++;
//	}
//	return defined_num;
//}
int getDefinedVoxelNum(const PACKED_POSITION *pos_tags, int m)
{
	int defined_num = 0;
	const PACKED_POSITION *pos_tag_ptr = pos_tags;
	for (int v = 0; v < m; v++)
	{
		if (ishashVoxelDefined(pos_tag_ptr))
		{
			defined_num++;
		}
		pos_tag_ptr++;
	}
	return defined_num;
}

bool loadHash(HashData &one_hash, FILE *fp)
{
	const int ori_channels = one_hash.m_channels;
	const int ori_mBar = one_hash.m_mBar;
	const int ori_rBar = one_hash.m_rBar;
	const int ori_m = ori_mBar * ori_mBar * ori_mBar;
	const int ori_r = ori_rBar * ori_rBar * ori_rBar;

	int num_channels;
	fread(&num_channels, sizeof(int), 1, fp);
	one_hash.m_channels = num_channels;
	fread(&one_hash.m_mBar, sizeof(int), 1, fp);	//m, for hash table and position hash table
	fread(&one_hash.m_rBar, sizeof(int), 1, fp);	//r, for offset table

	//hash data
	const int m = one_hash.m_mBar*one_hash.m_mBar*one_hash.m_mBar;
	if (m * num_channels > ori_m * ori_channels)	//need larger memory
	{
		SAFE_VDELETE(one_hash.m_hash_data);
		one_hash.m_hash_data = new float[m*num_channels];
	}
	fread(one_hash.m_hash_data, sizeof(float), m*num_channels, fp);
	//offset data
	const int r = one_hash.m_rBar * one_hash.m_rBar * one_hash.m_rBar;
	if (r > ori_r)
	{
		SAFE_VDELETE(one_hash.m_offset_data);
		one_hash.m_offset_data = new unsigned char[r * 3];
	}
	fread(one_hash.m_offset_data, sizeof(unsigned char), r * 3, fp);
	//position hash
	if (m > ori_m)
	{
		SAFE_VDELETE(one_hash.m_position_tag);
		one_hash.m_position_tag = new PACKED_POSITION[m];
	}
	fread(one_hash.m_position_tag, sizeof(PACKED_POSITION), m, fp);

	//calc defined num
	one_hash.m_defNum = getDefinedVoxelNum(one_hash.m_position_tag, m);

	return true;
}

bool loadHash(HashData &one_hash, const char *filename)
{
	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		printf("Error: failed to save PSH to %s\n", filename);
		return false;
	}
	bool flag = loadHash(one_hash,fp);
	fclose(fp);
	return flag;
}


bool loadBatchHashData(BatchHashData &batch_data, const char *filelist)
{
	std::ifstream ifs(filelist);
	if (ifs.fail())
	{
		printf("Error: failed to load batch Hash from list %s\n", filelist);
		ifs.close();
		return false;
	}
	std::string line, name;
	std::vector<std::string> hash_names;
	while (!ifs.eof())
	{
		std::getline(ifs, line);
		if (ifs.fail())
		{
			break;
		}
		if (line.empty())
			continue;
		std::istringstream sin(line);
		sin >> name;
		hash_names.push_back(name);
	}
	ifs.close();


	//load hashes
	std::vector<HashData> vHashData(hash_names.size());
	for (int i = 0; i < (int)hash_names.size();i++)
	{
		if (!loadHash(vHashData[i], hash_names[i].c_str()))
		{
			printf("Fatal error: load batch hash failed!\n");
			exit(0);
		}
	}

	mergeHash_2_Batch(vHashData, batch_data);

	return true;
}


void initBatchHash(BatchHashData &batchHash)
{
	batchHash.m_hash_data = NULL;
	batchHash.m_position_tag = NULL;
	batchHash.m_offset_data = NULL;

	batchHash.m_mBars.resize(0);
	batchHash.m_rBars.resize(0);
	batchHash.m_defNums.resize(0);
	batchHash.m_channels = 0;
}

void initHash(HashData &one_hash)
{
	one_hash.m_hash_data = NULL;
	one_hash.m_position_tag = NULL;
	one_hash.m_offset_data = NULL;
	one_hash.m_mBar = 0;
	one_hash.m_rBar = 0;
	one_hash.m_defNum = 0;
	one_hash.m_channels = 0;
}

void mergeHash_2_Batch(std::vector<HashData> &vHashes, BatchHashData &out_batch)
{
	out_batch.m_channels = vHashes[0].m_channels;

	out_batch.m_mBars.resize(vHashes.size());
	out_batch.m_rBars.resize(vHashes.size());
	out_batch.m_defNums.resize(vHashes.size());
	//gather the total size of m, and record each m && r
	int batch_hash_size = 0;
	int batch_offset_size = 0;
	for (int i = 0; i < (int)vHashes.size(); i++)
	{
		int m_bar = vHashes[i].m_mBar;
		int r_bar = vHashes[i].m_rBar;

		out_batch.m_mBars[i] = m_bar;
		out_batch.m_rBars[i] = r_bar;
		out_batch.m_defNums[i] = vHashes[i].m_defNum;

		batch_hash_size += m_bar * m_bar * m_bar;
		batch_offset_size += r_bar * r_bar * r_bar;
	}
	

	printf("Batch hash size is %d\n",batch_hash_size);
	printf("Batch offset size is %d\n",batch_offset_size);
	out_batch.m_hash_data = new float[batch_hash_size * out_batch.m_channels];
	out_batch.m_offset_data = new unsigned char[batch_offset_size * 3];
	out_batch.m_position_tag = new PACKED_POSITION[batch_hash_size];

	float *batch_hash_ptr = out_batch.m_hash_data;
	unsigned char *batch_offset_ptr = out_batch.m_offset_data;
	PACKED_POSITION *batch_posTag_ptr = out_batch.m_position_tag;
	for (int i = 0; i < (int)vHashes.size(); i++)
	{
		int m_bar = vHashes[i].m_mBar;
		int r_bar = vHashes[i].m_rBar;
		int m = m_bar*m_bar*m_bar;
		int r = r_bar*r_bar*r_bar;

		memcpy(batch_hash_ptr,vHashes[i].m_hash_data,sizeof(float)*m*out_batch.m_channels);
		memcpy(batch_offset_ptr, vHashes[i].m_offset_data, sizeof(unsigned char)*r*3);
		memcpy(batch_posTag_ptr, vHashes[i].m_position_tag, sizeof(PACKED_POSITION)*m);

		batch_hash_ptr += m*out_batch.m_channels;
		batch_offset_ptr += r * 3;
		batch_posTag_ptr += m;
	}
}

void share_hash_structure(BatchHashData &src, BatchHashData &tgt)
{
	tgt.m_defNums.assign(src.m_defNums.begin(), src.m_defNums.end());
	tgt.m_mBars.assign(src.m_mBars.begin(), src.m_mBars.end());
	tgt.m_rBars.assign(src.m_rBars.begin(), src.m_rBars.end());
	tgt.m_offset_data = src.m_offset_data;
	tgt.m_position_tag = src.m_position_tag;
}

void reshape_batchHash(BatchHashData &batch_hash, int channels)
{
	SAFE_VDELETE(batch_hash.m_hash_data);
	batch_hash.m_channels = channels;
	int batch_hash_size = 0;
	for (int i = 0; i < (int)batch_hash.m_mBars.size(); i++)
	{
		int m_bar = batch_hash.m_mBars[i];
		batch_hash_size += m_bar * m_bar * m_bar;
	}
	
	batch_hash.m_hash_data = new float[batch_hash_size * channels];
}


void hash_2_dense(const float *hash_data, const PACKED_POSITION *position_tags, const unsigned char *m_offset_data,
	int m_bar, int r_bar, int channels,
	float *dense_data, int res)
{
	int res3 = res*res*res;
	memset(dense_data, 0, sizeof(float)*res3 * channels);

	int m = m_bar * m_bar * m_bar;
	const float *hash_ptr = hash_data;
	for (int i = 0; i < m; i++)
	{
		if (!ishashVoxelDefined(&position_tags[i]))
		{
			hash_ptr++;
			continue;
		}
		int x, y, z;
		xyz_from_pack(position_tags[i], x, y, z);
		int ni = NXYZ2I(x, y, z, res, res*res);
		float *cur_dense_ptr = dense_data + ni;
		const float *cur_hash_ptr = hash_ptr;
		for (int c = 0; c < channels; c++)
		{
			*cur_dense_ptr = *cur_hash_ptr;
			cur_dense_ptr+=res3;
			cur_hash_ptr += m;
		}
		hash_ptr++;
	}
}




bool loadHashes(std::vector<HashData> &hashes, const char *filename)
{
	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		printf("Error: failed to load mulitple hashes from %s\n", filename);
		return false;
	}
	bool flag = loadHashes(hashes, fp);
	fclose(fp);
	return flag;
}


bool loadHashes(std::vector<HashData> &hashes, FILE *fp)
{
	//!!!NOTE: we dont destroy the memory here. 
	//When loadHash(), we only destroy and new when the size is bigger than exisiting one
	//for (int i = 0; i < (int)hashes.size(); i++)
	//{
	//	destroyHash(hashes[i]);
	//}
	int n;
	fread(&n, sizeof(int), 1, fp);


	//actually when load batches, n will always to the same
	if (hashes.size() && (int)hashes.size()!=n || !hashes.size())
	{
		for (int i = 0; i < (int)hashes.size(); i++)
		{
			destroyHash(hashes[i]);
		}
		hashes.resize(n);
		for (int i=0;i<n;i++)
		{
			initHash(hashes[i]);
		}
	}

	for (int i = 0; i < n; i++)
	{
		loadHash(hashes[i], fp);
	}
	return true;
}


void destroyHash(HashData &one_hash)
{
	SAFE_VDELETE(one_hash.m_hash_data);
	SAFE_VDELETE(one_hash.m_position_tag);
	SAFE_VDELETE(one_hash.m_offset_data);
	one_hash.m_mBar = 0;
	one_hash.m_rBar = 0;
	one_hash.m_defNum = 0;
	one_hash.m_channels = 0;
}

void destroyHashes(std::vector<HashData> &hashes)
{
	for (int i = 0; i < hashes.size(); i++)
	{
		destroyHash(hashes[i]);
	}
	hashes.resize(0);
}

void destroyBatchHash(BatchHashData &batch_hash)
{
	SAFE_VDELETE(batch_hash.m_hash_data);
	SAFE_VDELETE(batch_hash.m_position_tag);
	SAFE_VDELETE(batch_hash.m_offset_data);
	batch_hash.m_mBars.resize(0);
	batch_hash.m_rBars.resize(0);
	batch_hash.m_defNums.resize(0);
	batch_hash.m_channels = 0;
}

//NOTE: in caffe, the memory of batchhash is managed by blobs
void blobs_2_batchHash(const std::vector<caffe::Blob<float>*>& blobs, BatchHashData &batch_hash)
{
	int batch_num = blobs[M_BAR_BLOB]->shape(0);
	if (!batch_num)
	{
		return;
	}
	initBatchHash(batch_hash);

	batch_hash.m_hash_data = blobs[HASH_DATA_BLOB]->mutable_cpu_data();
	batch_hash.m_position_tag = (PACKED_POSITION*)blobs[POSTAG_BLOB]->mutable_cpu_data();
	batch_hash.m_offset_data = (unsigned char *)blobs[OFFSET_BLOB]->mutable_cpu_data();	

	if ((int)batch_hash.m_mBars.size()!= batch_num ||
		(int)batch_hash.m_rBars.size() != batch_num || 
		(int)batch_hash.m_defNums.size() != batch_num)
	{
		batch_hash.m_mBars.resize(batch_num);
		batch_hash.m_rBars.resize(batch_num);
		batch_hash.m_defNums.resize(batch_num);
	}
	for (int i=0;i<batch_num;i++)
	{
		batch_hash.m_mBars[i] = (int)blobs[M_BAR_BLOB]->mutable_cpu_data()[i];
		batch_hash.m_rBars[i] = (int)blobs[R_BAR_BLOB]->mutable_cpu_data()[i];
		batch_hash.m_defNums[i] = (int)blobs[DEFNUM_BLOB]->mutable_cpu_data()[i];
	}
	
	
	int m_channels;
}