#ifndef _STJ_MACROS_H
#define _STJ_MACROS_H

#undef SAFE_DELETE
#define SAFE_DELETE(ptr) \
if (ptr) { delete ptr; ptr = NULL; }

#undef SAFE_VDELETE
#define SAFE_VDELETE(ptr) \
if (ptr) { delete[]ptr; ptr = NULL; }

#define CLAMP(x, min, max) ((x)<(min) ? (min) :((x)>(max) ? (max) : (x)))

#define STANDARD_RES 256

#define INVALID_POSTAG 9999


#define HASH_STRUCTURE_SIZE 10	//offset, postag, m_bar, r_bar, def_num, valid_pos
#define HASH_DATA_SIZE 3	//layer output blob num: data, channel, dense_res
//data blob idx 
#define HASH_DATA_BLOB 0	
#define DENSE_RES_BLOB 1	
#define CHANNEL_BLOB 2
//structure blob idx
#define OFFSET_BLOB 3
#define POSTAG_BLOB 4
#define M_BAR_BLOB 5
#define R_BAR_BLOB 6
#define DEFNUM_BLOB 7
#define VALID_POS_BLOB 8
#define VOLUME_IDX_BLOB 9
#define DEFNUM_SUM_BLOB 10 // def_num_sum[i] = sum(defnum(0:i-1));
#define M_SUM_BLOB 11	// m_sum[i] = sum(m(0:i-1)), where m=m_bar*m_bar*m_bar
#define R_SUM_BLOB 12	// r_sum[i] = sum(r(0:i-1)), where r=r_bar*r_bar*r_bar

#define DUMP_2_TXT 0
#endif