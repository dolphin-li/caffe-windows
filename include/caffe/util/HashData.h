#ifndef _HASH_DATA_H
#define _HASH_DATA_H

#include <vector>
#include "caffe/blob.hpp"
#include "MyMacro.h"
//used to pack xyz position 
typedef struct tag_PACKED_POSITION
{
	unsigned char _val[6];	//10 for X, 32 for Y, 54 for Z
}PACKED_POSITION;

inline void pack_xyz(PACKED_POSITION &packed_val, int x, int y, int z)
{
	//int &t0 = *(int*)(&packed_val._val[0]);
	//int &t1 = *(int*)(&packed_val._val[1]);
	//int &t2 = *(int*)(&packed_val._val[2]);
	//int &t3 = *(int*)(&packed_val._val[3]);
	//int &t4 = *(int*)(&packed_val._val[4]);
	//int &t5 = *(int*)(&packed_val._val[5]);
	//pack x y

	//NOTE!!!! (int*)(&val[0]] will occupy [0,1,2,3]. !!!!BUT!!!! 0 is the lowest position in the int!!!!
	//!!!THAT is, the int is ACTUALLY organized as 3210!!!!
	int &packed_xy = *(int*)(&packed_val._val[0]);
	packed_xy = ((packed_xy & 0xFFFF0000) | (x & 0x0000FFFF));
	packed_xy = ((packed_xy & 0x0000FFFF) | ((y & 0x0000FFFF) << 16));

	//pack z
	int &packed_z = *(int*)(&packed_val._val[2]);
	packed_z = (packed_z & 0x0000FFFF) | ((z & 0x0000FFFF) << 16);

}

inline void xyz_from_pack(const PACKED_POSITION &packed_val, int &x, int &y, int &z)
{
	const int &packed_xy = *(const int*)(&packed_val._val[0]);
	x = (packed_xy & 0x0000FFFF);
	y = (packed_xy & 0xFFFF0000) >> 16;

	const int &packed_z = *(const int*)(&packed_val._val[2]);
	z = (packed_z & 0xFFFF0000) >> 16;
}

class HashData
{
public:
	HashData()
	{
		m_hash_data = NULL;
		m_position_tag = NULL;
		m_offset_data = NULL;
		m_mBar = 0;
		m_rBar = 0;
		m_defNum = 0;
		m_channels = 0;
		m_dense_res = 0;
	}
public:
	float *m_hash_data;
	PACKED_POSITION *m_position_tag;
	unsigned char *m_offset_data;
	int m_mBar;
	int m_rBar;
	int m_defNum;
	int m_channels;
	int m_dense_res;
};

class BatchHashData
{
public:
	BatchHashData()
	{
		m_hash_data = NULL;
		m_position_tag = NULL;
		m_offset_data = NULL;
		m_channels = 0;
	}
public:
	float *m_hash_data;		//hash data; in batch mode, it is Packed as (M1, M2, ..., Mn)
	PACKED_POSITION *m_position_tag;	//position tag; in batch mode, it is packed as (M1, M2, ..., Mn)
	unsigned char *m_offset_data;	//offset table; in batch mode, it is Packed as (R1, R2, ..., Rn)

	//unsigned char *m_hash_table_p;	//position hash table; in batch mode, it is packed as (P1,P2,...,Pn)
	std::vector<int> m_mBars;			//different m for each hash (m*m*m to store each hash data)
	std::vector<int> m_rBars;			//different r for each hash (r*r*r to store offset data)
	std::vector<int> m_defNums;			//defined voxel num in each hash
	int m_channels;
};

//int getDefinedVoxelNum(const float *hash_data, int m_bar, int channels);
int getDefinedVoxelNum(const PACKED_POSITION *pos_tags, int m);
void getValidPoses(const PACKED_POSITION *pos_tags, int* valid_poses, int m);
//inline bool ishashVoxelDefined(const float *hash_voxel_ptr, int channels, int m)
//{
//	for (int c = 0; c < channels; c++)
//	{
//		if (*hash_voxel_ptr)
//		{
//			return true;
//			break;
//		}
//		hash_voxel_ptr+=m;
//	}
//	return false;
//}
inline bool ishashVoxelDefined(const PACKED_POSITION *pos_tag_ptr)
{
	int x, y, z;
	xyz_from_pack(*pos_tag_ptr, x, y, z);
	return !(x== INVALID_POSTAG || y== INVALID_POSTAG || z== INVALID_POSTAG);
}

bool loadHash(HashData &hash_data, const char *filename);
bool loadHash(HashData &hash_data, FILE *fp);
bool saveHash(const HashData &hash_data, const char *filename);
bool saveHash(const HashData &hash_data, FILE *fp);
bool saveHashStruct(const HashData &hash_data, const char *filename);
bool saveHashStruct(const HashData &hash_data, FILE *fp);
bool loadBatchHashData(BatchHashData &batch_data, const char *filelist);

bool loadHashes(std::vector<HashData> &hashes, const char *filename);
bool loadHashes(std::vector<HashData> &hashes, FILE *fp);

void mergeHash_2_Batch(std::vector<HashData> &vHashes, BatchHashData &out_batch);
void initBatchHash(BatchHashData &batchHash);
void initHash(HashData &one_hash);
void destroyHash(HashData &hash_data);
void destroyHashes(std::vector<HashData> &hashes);
void destroyBatchHash(BatchHashData &batch_hash);


void share_hash_structure(BatchHashData &src, BatchHashData &tgt);
void reshape_batchHash(BatchHashData &batch_hash,int channels);

void hash_2_dense(const float *hash_data, const PACKED_POSITION *position_tags, const unsigned char *m_offset_data,
	int m_bar, int r_bar, int channels,
	float *dense_data, int res);
void topMask_2_dense(const int *top_mask, const PACKED_POSITION *top_posTags, const unsigned char *top_offset,
	int top_m_bar, int top_r_bar, int channels, int top_res,
	const PACKED_POSITION *bottom_posTags, int bottom_res,
	int *dense_idx);
void dense_2_hash(float *hash_data, const PACKED_POSITION *position_tags, const unsigned char *m_offset_data,
	int m_bar, int r_bar, int channels,
	const float *dense_data, int res);

_inline int NXYZ2I(int nx, int ny, int nz, int n, int n2)
{
	return nz*n2 + ny*n + nx;
};

_inline void Hash(int nx, int ny, int nz, int& mx, int& my, int& mz,
	const unsigned char *offset_data, int m_bar, int r_bar, int r2)
{
	int rx = nx%r_bar;
	int ry = ny%r_bar;
	int rz = nz%r_bar;

	const unsigned char *offset = &offset_data[NXYZ2I(rx, ry, rz, r_bar, r2) * 3];

	mx = (nx + offset[0]) % m_bar;
	my = (ny + offset[1]) % m_bar;
	mz = (nz + offset[2]) % m_bar;
}

void blobs_2_batchHash(const std::vector<caffe::Blob<float>*>& blobs, BatchHashData &batch_hash, int dif_flag = 0);


#ifdef __CUDACC__
__device__ __host__ inline void xyz_from_pack_g(PACKED_POSITION packed_val, int &x, int &y, int &z)
{
	const int packed_xy = *(const int*)(packed_val._val);
	x = (packed_xy & 0x0000FFFF);
	y = (packed_xy & 0xFFFF0000) >> 16;

	const int packed_z = *(const int*)(packed_val._val + 2);
	z = (packed_z & 0xFFFF0000) >> 16;
}

__device__ __host__ inline bool ishashVoxelDefined_g(PACKED_POSITION pos_tag)
{
	int x, y, z;
	xyz_from_pack_g(pos_tag, x, y, z);
	return !(x == INVALID_POSTAG || y == INVALID_POSTAG || z == INVALID_POSTAG);
}

__device__ __host__ inline int NXYZ2I_g(int nx, int ny, int nz, int n, int n2)
{
	return nz*n2 + ny*n + nx;
};

__device__ __host__ inline void Hash_g(int nx, int ny, int nz, int& mx, int& my, int& mz,
	const unsigned char *offset_data, int m_bar, int r_bar, int r2)
{
	int rx = nx%r_bar;
	int ry = ny%r_bar;
	int rz = nz%r_bar;

	const unsigned char *offset = offset_data + NXYZ2I_g(rx, ry, rz, r_bar, r2) * 3;

	mx = (nx + offset[0]) % m_bar;
	my = (ny + offset[1]) % m_bar;
	mz = (nz + offset[2]) % m_bar;
}
#endif

/************************************************************************************************/
/*******************************For hierachy hashes**********************************/
/*****************************Different layers will have different hash structures********************/
/*****************************Different layers may share the same hash structures**************/

//hash struct info, may be shared by different layers
class CHashStructInfo	
{
public:
	CHashStructInfo();
	~CHashStructInfo();
	void destroy();
	int load(FILE *fp);
	int save(FILE *fp) const;
public:
	PACKED_POSITION *m_position_tag;
	unsigned char *m_offset_data;
	int m_mBar;
	int m_rBar;
	int m_defNum;
};

//preloaded hierhash
class CHierarchyHash
{
public:
	CHierarchyHash();
	~CHierarchyHash();
	void destroy();
	void destroyStructs();
	void initStructs(int n);
	int load(FILE *fp);
	int save(FILE *fp) const;
public:
	float *m_hash_data;		//bottom hash data
	int m_dense_res;	//bottom dense res
	int m_channels;		//bottom channel
	std::vector<CHashStructInfo*> m_vpStructs;
};


int writeDense_2_HF5(const float *dense_data, int n, int res, int channels, const char *filename);

int writeBatchHash_2_denseFiles(const BatchHashData &batch, int res, const char *prefix);

int writeDense_2_Grid(const float *dense_data, int res, int channels, const char *filename);
int writeDense_2_Grid(const int *dense_data, int res, int channels, const char *filename);
/************************************UTILS********************************/
void calc_sum(const float *hash, const unsigned char *offset,
	const PACKED_POSITION *posTag, int m_bar, int r_bar,
	int channels, int def_num, float weight, float *out_weighted_sum);
void hash_add_scalar(float *hash, const unsigned char *offset,
	const PACKED_POSITION *posTag, int m_bar, int r_bar,
	int channels, int def_num, const float *to_adds);	
void hash_subtract_scalar(float *hash, const unsigned char *offset,
	const PACKED_POSITION *posTag, int m_bar, int r_bar,
	int channels, int def_num, const float *to_substracts);
void hash_mult_scalar(float *hash, const unsigned char *offset,
	const PACKED_POSITION *posTag, int m_bar, int r_bar,
	int channels, int def_num, const float *to_mults);
//used for calc variance
void calc_square_sum(const float *hash, const unsigned char *offset,
	const PACKED_POSITION *posTag, int m_bar, int r_bar,
	int channels, int def_num, float weight, float *out_weighted_sum);


#endif