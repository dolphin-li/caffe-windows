/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include "stdint.h"
#include "caffe/layers/hdf5_bit_data_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

	template <typename Dtype>
	HDF5BitDataLayer<Dtype>::~HDF5BitDataLayer<Dtype>() { }

	template <typename Dtype>
	static void convert_bool_to_dtype(int B_cnt, const char* B, Dtype* A)
	{
		for (int i = 0; i < B_cnt; i++)
		{
			int apos = i * 8;
			const char data = B[i];
			for (int k = 0; k < 8; k++, apos++)
				A[apos] = HDF5BitDataLayer<Dtype>::bool_to_value((data >> k) & 0x1);
		}
	}

	template <typename Dtype>
	void hdf5_load_nd_dataset_char_as_bit_helper(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<Dtype>* blob, bool& is_bit_data)
	{
		// Verify that the dataset exists.
		CHECK(H5LTfind_dataset(file_id, dataset_name_)) << "Failed to find HDF5 dataset " << dataset_name_;
		// Verify that the number of dimensions is in the accepted range.
		int ndims = 0;
		herr_t status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
		CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
		CHECK_GE(ndims, min_dim);
		CHECK_LE(ndims, max_dim);

		// Verify that the data format is what we expect: float or double.
		std::vector<hsize_t> dims(ndims);
		H5T_class_t class_;
		size_t class_bytes_ = 0;
		status = H5LTget_dataset_info(file_id, dataset_name_, dims.data(), &class_, &class_bytes_);
		CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
		switch (class_) {
		case H5T_FLOAT:
		{LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_FLOAT"; }
			break;
		case H5T_INTEGER:
		{LOG_FIRST_N(INFO, 1) << "Datatype class: H5T_INTEGER"; }
			break;
		case H5T_TIME:
		{LOG(FATAL) << "Unsupported datatype class: H5T_TIME"; }
		case H5T_STRING:
		{LOG(FATAL) << "Unsupported datatype class: H5T_BITFIELD"; }
		case H5T_BITFIELD:
		{LOG(FATAL) << "Unsupported datatype class: H5T_BITFIELD"; }
		case H5T_OPAQUE:
		{LOG(FATAL) << "Unsupported datatype class: H5T_OPAQUE"; }
		case H5T_COMPOUND:
		{LOG(FATAL) << "Unsupported datatype class: H5T_COMPOUND"; }
		case H5T_REFERENCE:
		{LOG(FATAL) << "Unsupported datatype class: H5T_REFERENCE"; }
		case H5T_ENUM:
		{LOG(FATAL) << "Unsupported datatype class: H5T_ENUM"; }
		case H5T_VLEN:
		{LOG(FATAL) << "Unsupported datatype class: H5T_VLEN"; }
		case H5T_ARRAY:
		{LOG(FATAL) << "Unsupported datatype class: H5T_ARRAY"; }
		default:
		{LOG(FATAL) << "Datatype class unknown"; }
		}

		// get dimensions, here we treat int8 as bit data.. a hack since HDF5 does not natively support bool bit.
		vector<int> blob_dims(dims.size());
		for (int i = 0; i < dims.size(); ++i)
			blob_dims[i] = dims[i];
		is_bit_data = (class_ == H5T_INTEGER && class_bytes_ == 1);
		if (is_bit_data)
			blob_dims[0] *= 8;
		blob->Reshape(blob_dims);
	}

	template <typename Dtype>
	void hdf5_load_nd_dataset_char_as_bit(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<Dtype>* blob);
	template <>
	void hdf5_load_nd_dataset_char_as_bit(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<float>* blob)
	{
		bool is_bit_data = false;
		hdf5_load_nd_dataset_char_as_bit_helper<float>(file_id, dataset_name_, min_dim, max_dim, blob, is_bit_data);

		// if bit data, we convert it to value
		if (is_bit_data)
		{
			std::vector<char> tmp(blob->count() / 8);
			CHECK_EQ(blob->shape(0) % 8, 0) << "HDF5 bit data, num must be *times 8: " << dataset_name_;
			herr_t status = H5LTread_dataset_char(file_id, dataset_name_, tmp.data());
			CHECK_GE(status, 0) << "Failed to read bit dataset" << dataset_name_;
			convert_bool_to_dtype(tmp.size(), tmp.data(), blob->mutable_cpu_data());
		}
		// else, we treat as general data
		else
		{
			herr_t status = H5LTread_dataset_float(file_id, dataset_name_, blob->mutable_cpu_data());
			CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
		}
	}
	template <>
	void hdf5_load_nd_dataset_char_as_bit(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<double>* blob)
	{
		bool is_bit_data = false;
		hdf5_load_nd_dataset_char_as_bit_helper<double>(file_id, dataset_name_, min_dim, max_dim, blob, is_bit_data);

		// if not bit data, we simply read it
		if (is_bit_data)
		{
			std::vector<char> tmp(blob->count() / 8);
			CHECK_EQ(blob->shape(0) % 8, 0) << "HDF5 bit data, num must be *times 8: " << dataset_name_;
			herr_t status = H5LTread_dataset_char(file_id, dataset_name_, tmp.data());
			CHECK_GE(status, 0) << "Failed to read bit dataset" << dataset_name_;
			convert_bool_to_dtype(tmp.size(), tmp.data(), blob->mutable_cpu_data());
		}
		// else, we treat as bit data
		else
		{
			herr_t status = H5LTread_dataset_double(file_id, dataset_name_, blob->mutable_cpu_data());
			CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
		}
	}

	// Load data and label from HDF5 filename into the class property blobs.
	template <typename Dtype>
	void HDF5BitDataLayer<Dtype>::LoadHDF5FileData(const char* filename) 
	{
		LOG(INFO) << "###############0: " << filename;
		DLOG(INFO) << "Loading HDF5 file: " << filename;
		hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file_id < 0)
			LOG(FATAL) << "Failed opening HDF5 file: " << filename;

		int top_size = this->layer_param_.top_size();
		hdf_blobs_.resize(top_size);

		const int MIN_DATA_DIM = 1;
		const int MAX_DATA_DIM = INT_MAX;

		for (int i_top = 0; i_top < top_size; ++i_top)
		{
			hdf_blobs_[i_top] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
			// Allow reshape here, as we are loading data not params
			hdf5_load_nd_dataset_char_as_bit(file_id, this->layer_param_.top(i_top).c_str(),
				MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i_top].get());
		} // end for i_top

		herr_t status = H5Fclose(file_id);
		CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

		// MinTopBlobs==1 guarantees at least one top blob
		CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
		for (int i = 1; i < top_size; ++i)
			CHECK_EQ(hdf_blobs_[i]->shape(0), hdf_blobs_[0]->shape(0));
		
		// Default to identity permutation.
		data_permutation_.clear();
		data_permutation_.resize(hdf_blobs_[0]->shape(0));
		for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
			data_permutation_[i] = i;

		// Shuffle if needed.
		if (this->layer_param_.hdf5_data_param().shuffle()) {
			std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
			DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows (shuffled)";
		}
		else
			DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows";
	}

	INSTANTIATE_CLASS(HDF5BitDataLayer);
	REGISTER_LAYER_CLASS(HDF5BitData);
}  // namespace caffe
