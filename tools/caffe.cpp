#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/signal_handler.h"
//******added by tianjia
#include <windows.h>
#include "caffe/util/MyMacro.h"
#include "caffe/layers/conv_hash_layer.hpp"
#include "caffe/layers/pool_hash_layer.hpp"
//**************************//
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  return caffe::SolverAction::NONE;
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.has_solver_mode()
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  LOG(INFO) << "Starting Optimization";
  if (gpus.size() > 1) {
#ifdef USE_NCCL
    caffe::NCCL<float> nccl(solver);
    nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
    LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

/**************************For testing hash**************************/
void test_hash_data_layer_forward(std::vector<Blob<float> *> &top)
{
	std::vector<Blob<float> *> bottom; //no use
	const int struct_num = 3;
	int top_blob_num = HASH_DATA_SIZE + HASH_STRUCTURE_SIZE * struct_num + 1;
	top.resize(top_blob_num);
	for (int i = 0; i < top_blob_num; i++)
	{
		top[i] = new Blob<float>();
	}

	caffe::HashDataParameter *hash_data_param = new caffe::HashDataParameter;
	const int batch_size = 10;
	hash_data_param->set_source("HashLayerList.txt");
	hash_data_param->set_batch_size(batch_size);
	hash_data_param->set_shuffle(true);

	caffe::LayerParameter hash_layer_param;
	hash_layer_param.set_type("HashData");
	hash_layer_param.set_allocated_hash_data_param(hash_data_param);

	shared_ptr<Layer<float> > hash_data_layer =
		caffe::LayerRegistry<float>::CreateLayer(hash_layer_param);

	//setup 
	hash_data_layer->SetUp(bottom, top);
	//forward
	hash_data_layer->Forward(bottom, top);
}

void test_hash_conv_layer_forward(const std::vector<Blob<float> *> &bottom, std::vector<Blob<float> *> &top,
	int num_output, int kernel_size)
{
	const int top_blob_num = HASH_DATA_SIZE;
	top.resize(top_blob_num);	//only data updated, structure will share the same
	for (int i = 0; i < top_blob_num; i++)
	{
		top[i] = new Blob<float>();
	}

	caffe::ConvHashParameter *conv_hash_param = new caffe::ConvHashParameter;
	conv_hash_param->set_num_output(num_output);
	conv_hash_param->set_bias_term(true);

	int channels = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	for (int i=0;i<channels;i++)
	{
		conv_hash_param->add_kernel_size(kernel_size);
	}
	

	caffe::FillerParameter *w_filler = new caffe::FillerParameter;
	w_filler->set_type("gaussian");
	w_filler->set_std(1);
	//w_filler->set_type("constant");
	//w_filler->set_value(1);
	conv_hash_param->set_allocated_weight_filler(w_filler);

	caffe::FillerParameter *b_filler = new caffe::FillerParameter;
	b_filler->set_type("gaussian");
	b_filler->set_std(1);
	conv_hash_param->set_allocated_bias_filler(b_filler);
	

	caffe::LayerParameter conv_hash_layer_param;
	conv_hash_layer_param.set_type("ConvHash");
	conv_hash_layer_param.set_allocated_conv_hash_param(conv_hash_param);

	shared_ptr<Layer<float> > hash_conv_layer(new caffe::ConvHashLayer<float>(conv_hash_layer_param));
	//setup 
	hash_conv_layer->SetUp(bottom, top);
	//forward
	hash_conv_layer->Forward(bottom, top);

	//save dense to HDF5 for evaluation
	BatchHashData bottom_batch;
	blobs_2_batchHash(bottom, bottom_batch);
	bottom_batch.m_channels = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	int dense_res = (int)bottom[DENSE_RES_BLOB]->cpu_data()[0];
	writeBatchHash_2_denseFiles(bottom_batch, dense_res, "bottom");
	
	std::vector<Blob<float> *> structed_top(HASH_DATA_SIZE + HASH_STRUCTURE_SIZE);
	structed_top[HASH_DATA_BLOB] = top[HASH_DATA_BLOB];
	structed_top[CHANNEL_BLOB] = top[CHANNEL_BLOB];
	structed_top[DENSE_RES_BLOB] = top[DENSE_RES_BLOB];
	structed_top[OFFSET_BLOB] = bottom[OFFSET_BLOB];
	structed_top[POSTAG_BLOB] = bottom[POSTAG_BLOB];
	structed_top[M_BAR_BLOB] = bottom[M_BAR_BLOB];
	structed_top[R_BAR_BLOB] = bottom[R_BAR_BLOB];
	structed_top[DEFNUM_BLOB] = bottom[DEFNUM_BLOB];
	BatchHashData top_batch;
	blobs_2_batchHash(structed_top, top_batch);
	top_batch.m_channels = num_output;
	writeBatchHash_2_denseFiles(top_batch, dense_res, "top");
}

void test_pool_layer_forward(const std::vector<Blob<float> *> &bottom, std::vector<Blob<float> *> &top,
	int stride)
{
	const int top_blob_num = HASH_DATA_SIZE;
	top.resize(top_blob_num);	//only data updated, structure is already given
	for (int i = 0; i < top_blob_num; i++)
	{
		top[i] = new Blob<float>();
	}

	caffe::PoolHashParameter *pool_hash_param = new caffe::PoolHashParameter;

	int channels = (int)bottom[CHANNEL_BLOB]->cpu_data()[0];
	for (int i = 0; i < channels; i++)
	{
		pool_hash_param->add_stride(stride);
	}
	pool_hash_param->set_pool(caffe::PoolHashParameter_PoolMethod_MAX);

	caffe::LayerParameter pool_hash_layer_param;
	pool_hash_layer_param.set_type("PoolvHash");
	pool_hash_layer_param.set_allocated_pool_hash_param(pool_hash_param);

	shared_ptr<Layer<float> > pool_conv_layer(new caffe::PoolHashLayer<float>(pool_hash_layer_param));
	//setup 
	pool_conv_layer->SetUp(bottom, top);
	//forward
	pool_conv_layer->Forward(bottom, top);
}

void test_hash()
{
	printf("Testing hash data layer...\n");
	const char *root_dir = "D:\\Projects\\TestHCNN\\testData";
	SetCurrentDirectoryA(root_dir);
	
	char Buffer[128];
	DWORD dwRet;
	dwRet = GetCurrentDirectoryA(128, Buffer);
	printf("Cur dir %s\n",Buffer);


	/******************Data layer**************************/
	std::vector<Blob<float> *> data_top;
	test_hash_data_layer_forward(data_top);

	/*****************Conv layer***************************/
	const int num_output = 10;
	const int kernel_size = 3;
	std::vector<Blob<float> *> conv_top;
	std::vector<Blob<float>*> conv_bottom(HASH_DATA_SIZE + HASH_STRUCTURE_SIZE);
	conv_bottom[HASH_DATA_BLOB] = data_top[HASH_DATA_BLOB];
	conv_bottom[CHANNEL_BLOB] = data_top[CHANNEL_BLOB];
	conv_bottom[DENSE_RES_BLOB] = data_top[DENSE_RES_BLOB];
	conv_bottom[OFFSET_BLOB] = data_top[OFFSET_BLOB];
	conv_bottom[POSTAG_BLOB] = data_top[POSTAG_BLOB];
	conv_bottom[M_BAR_BLOB] = data_top[M_BAR_BLOB];
	conv_bottom[R_BAR_BLOB] = data_top[R_BAR_BLOB];
	conv_bottom[DEFNUM_BLOB] = data_top[DEFNUM_BLOB];

	test_hash_conv_layer_forward(conv_bottom, conv_top, num_output, kernel_size);
	
	/****************Pooling layer*****************************/
	const int stride = 2;
	std::vector<Blob<float> *> pool_top;
	std::vector<Blob<float>*> pool_bottom(HASH_DATA_SIZE + HASH_STRUCTURE_SIZE*2);
	pool_bottom[HASH_DATA_BLOB] = conv_top[HASH_DATA_BLOB];
	pool_bottom[CHANNEL_BLOB] = conv_top[CHANNEL_BLOB];
	pool_bottom[DENSE_RES_BLOB] = conv_top[DENSE_RES_BLOB];
	pool_bottom[OFFSET_BLOB] = data_top[OFFSET_BLOB];	//pool bottom struct
	pool_bottom[POSTAG_BLOB] = data_top[POSTAG_BLOB];
	pool_bottom[M_BAR_BLOB] = data_top[M_BAR_BLOB];
	pool_bottom[R_BAR_BLOB] = data_top[R_BAR_BLOB];
	pool_bottom[DEFNUM_BLOB] = data_top[DEFNUM_BLOB];
	//pool top struct
	pool_bottom[OFFSET_BLOB + HASH_STRUCTURE_SIZE] = data_top[OFFSET_BLOB + HASH_STRUCTURE_SIZE];	//pool bottom struct
	pool_bottom[POSTAG_BLOB + HASH_STRUCTURE_SIZE] = data_top[POSTAG_BLOB + HASH_STRUCTURE_SIZE];
	pool_bottom[M_BAR_BLOB + HASH_STRUCTURE_SIZE] = data_top[M_BAR_BLOB + HASH_STRUCTURE_SIZE];
	pool_bottom[R_BAR_BLOB + HASH_STRUCTURE_SIZE] = data_top[R_BAR_BLOB + HASH_STRUCTURE_SIZE];
	pool_bottom[DEFNUM_BLOB + HASH_STRUCTURE_SIZE] = data_top[DEFNUM_BLOB + HASH_STRUCTURE_SIZE];

	test_pool_layer_forward(pool_bottom, pool_top, 2);
	//do not handle memory, just for testing...
	printf("\n");
}


int main(int argc, char** argv) {
	srand(time(NULL));
	/*************For test*****************/
	test_hash();
	return 0;
	/**************************************/

  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
