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


#define HASH_STRUCTURE_SIZE 6	//offset, postag, m_bar, r_bar, def_num, valid_pos
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

#define DUMP_2_TXT 0
#endif