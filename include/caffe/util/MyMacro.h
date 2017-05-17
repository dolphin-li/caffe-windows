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

//blobs to store hash
#define HASH_MIN_BLOB_NUM 7
//blob idx 
#define HASH_DATA_BLOB 0
#define OFFSET_BLOB 1
#define POSTAG_BLOB 2
#define M_BAR_BLOB 3
#define R_BAR_BLOB 4
#define DEFNUM_BLOB 5
#define CHANNEL_BLOB 6
//for data layer
#define LABEL_BLOB 7
//for layer whose top shape is different from bottom
#define TOP_OFFSET_BLOB 7
#define TOP_POSTAG_BLOB 8
#define TOP_M_BAR_BLOB 9
#define TOP_R_BAR_BLOB 10
#define TOP_DEFNUM_BLOB 11
#define TOP_CHANNEL_BLOB 12
#endif