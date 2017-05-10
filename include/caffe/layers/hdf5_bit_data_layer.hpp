#ifndef CAFFE_HDF5_BIT_DATA_LAYER_HPP_
#define CAFFE_HDF5_BIT_DATA_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/hdf5_data_layer.hpp"

namespace caffe {

	/**
	 * @brief Provides data to the Net from HDF5 files.
	 *
	 * NOTE: int8 is treated BIT-WISE in this class
	 *	that is, for an input HDF5 data array of (N, C, ...) with type int8
	 *		we treat each int8 as 8 bool values and create a (N*8, C, ...) blob
	 *  thus for float-value class instances, 32 times CPU memory are used.
	 *		if the input HDF5 array size is 1Mb, the CPU memory used by this class is 32Mb
	 */
	template <typename Dtype>
	class HDF5BitDataLayer : public HDF5DataLayer<Dtype> {
	public:
		explicit HDF5BitDataLayer(const LayerParameter& param)
			: HDF5DataLayer<Dtype>(param){}
		virtual ~HDF5BitDataLayer();
		virtual inline const char* type() const { return "HDF5BitData"; }
		static inline Dtype bool_to_value(bool b) { return b ? Dtype(-1) : Dtype(1); }
	protected:
		virtual void LoadHDF5FileData(const char* filename);
	};

}  // namespace caffe

#endif  // CAFFE_HDF5_DATA_LAYER_HPP_
