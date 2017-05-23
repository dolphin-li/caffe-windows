#ifndef CAFFE_BASE_CONV_HASH_LAYER_HPP_
#define CAFFE_BASE_CONV_HASH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/HashData.h"
namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvHashLayer : public Layer<Dtype> {
 public:
  explicit BaseConvHashLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) { m_weight_bias_inited_flag = false; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
	 void forward_cpu_gemm(const float *bottom_hash, const unsigned char *bottom_offset,
		 const PACKED_POSITION *bottom_posTag, int m_bar, int r_bar,
		 int bottom_channels, int top_channels, int defined_voxel_num, int dense_res, float *out_col_buf);
  void forward_cpu_bias(float* out_col_buf, const float* bias, int defined_voxel_num);
  //void backward_cpu_gemm(const float *top_hash_dif, const unsigned char *top_offset,
	 // const PACKED_POSITION *top_posTag, int m_bar, int r_bar,
	 // int bottom_channels, int top_channels, int defined_voxel_num, int dense_res, float *col_buf);
  void backward_cpu_gemm(const float *out_col_buf, 
	  int bottom_channels, int top_channels, int defined_voxel_num, float *col_buf);
  void weight_cpu_gemm(const float *col_buf, const float* out_col_buf, float *weight_dif, 
	  int bottom_channels, int top_channels, int defined_voxel_num);
  void backward_cpu_bias(float* bias, const float* out_col_buf,int defined_voxel_num);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const float *bottom_hash, const unsigned char *bottom_offset,
	  const PACKED_POSITION *bottom_posTag, const int* valid_positions, int m_bar, int r_bar,
	  int bottom_channels, int top_channels, int defined_voxel_num, int dense_res, float *out_col_buf);
  void forward_gpu_bias(float* out_col_buf, const float* bias, int defined_voxel_num);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  void reshape_topHashData(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  void reshape_colBuf(const vector<Blob<Dtype>*>& bottom);
  //NOTE: unlike conv_layer, the input channels can be directly got 
  void reshape_weight_bias(const vector<Blob<Dtype>*>& bottom);	
  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;

  int num_spatial_axes_;	//3 in 3D case
  int channels_;
  int num_output_;
  bool bias_term_;
  
  int conv_out_channels_;
  int conv_in_channels_;
  int kernel_dim_;

  bool m_weight_bias_inited_flag;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> out_col_buffer_;
  Blob<Dtype> bias_multiplier_;

public://for debug
	int writeDenseKernel_2_HF5(const char *filename);
	int writeBias_2_HF5(const char *filename);
};


//NOTE: all matrix is row major
int conv_hash2col_cpu(const float* hash_data, const unsigned char *offset_data, const PACKED_POSITION *position_tags,
	const int kernel_shape[3],	//D, H, W
	int m_bar, int r_bar, int channels, int defined_num,
	int dense_res, float* col_buff);

int conv_col2hash_cpu(const PACKED_POSITION *pos_tags, float *out_hash_data,
	int m_bar, int out_channels, int defined_num, const float* col_buff);

int conv_hash2col_gpu(const float* hash_data, const unsigned char *offset_data, const PACKED_POSITION *position_tags,
	const int* valid_positions, const int kernel_shape[3],	//D, H, W
	int m_bar, int r_bar, int channels, int defined_num,
	int dense_res, float* col_buff);

int conv_col2hash_gpu(const PACKED_POSITION *pos_tags, const int* valid_positions, float *out_hash_data,
	int m_bar, int out_channels, int defined_num, const float* col_buff);

//added by tianjia, for BP
//convert the top dif has to col
int top_hash2col_cpu(const float *hash_data, const PACKED_POSITION *pos_tags,
	int m_bar, int out_channels, int defined_num, float* col_buff);
//conver the col to bottom dif
int bottom_col2hash_cpu(float* hash_data, const unsigned char *offset_data, const PACKED_POSITION *position_tags,
	const int kernel_shape[3],	//D, H, W
	int m_bar, int r_bar, int channels, int defined_num,
	int dense_res, const float* col_buff);

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
