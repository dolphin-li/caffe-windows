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

//blobs to store hash
#define HASH_STRUCTURE_SIZE 5	//offset, postag, m_bar, r_bar, def_num
//blob idx 
#define HASH_DATA_BLOB 0
#define OFFSET_BLOB 1
#define POSTAG_BLOB 2
#define M_BAR_BLOB 3
#define R_BAR_BLOB 4
#define DEFNUM_BLOB 5
//#define CHANNEL_BLOB 6
#endif