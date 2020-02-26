#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#ifdef USE_OPENCV
using std::ofstream;
using caffe::Caffe;

using namespace caffe;

namespace fs = boost::filesystem;

#define INTERVAL_NUM 2048	//Set interval num of distribution

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  int count = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDeviceCount(&count));
#else
  NO_GPU;
#endif
  for (int i = 0; i < count; ++i) {
    gpus->push_back(i);
  }
}

int main(int argc, char** argv){
	::google::InitGoogleLogging(argv[0]);

	gflags::SetUsageMessage("Compute Scale Factors"
        "Usage:\n"
        "    compute_qunatization_factor PROTO_FILE INPUT_CAFFEMODEL\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);


	// Check args
	if (argc != 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/drain_layer_weights");
		return 1;
	}

	// Caffe setting
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() == 0) {
		LOG(INFO) << "Use CPU.";
		std::cout<<"Use CPU.\n";
		Caffe::set_mode(Caffe::CPU);
	} else {
    	ostringstream s;
    	for (int i = 0; i < gpus.size(); ++i) {
    	  s << (i ? ", " : "") << gpus[i];
    	}
    	LOG(INFO) << "Using GPUs " << s.str();
		std::cout<<"Use GPUs.\n";
#ifndef CPU_ONLY
    	cudaDeviceProp device_prop;
    	for (int i = 0; i < gpus.size(); ++i) {
    	  cudaGetDeviceProperties(&device_prop, gpus[i]);
    	  LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    	}
#endif
    	Caffe::SetDevice(gpus[0]);
    	Caffe::set_mode(Caffe::GPU);
	}


	// Read Net Model from Proto Files: ref. https://gist.github.com/onauparc/dd80907401b26b602885
	std::cout<<"read Net Model..\n";
	Net<float> caffe_net(argv[1], caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(argv[2]);


	int max_col = 0;
	printf("%d\n", caffe_net.layers().size());
	for(int i=0;i<caffe_net.layers().size();i++) {
		Layer<float>* layer = caffe_net.layers()[i].get();
		if(strcmp(layer->type(),"Convolution")!=0) continue;
//		printf("convolution layer\n");
		if(max_col < layer->blobs()[0].get()->count() )
			max_col = layer->blobs()[0].get()->count();
	}
	printf("max: %d\n", max_col);

	for(int j=0; j<max_col; j++) {
		for(int i=0;i<caffe_net.layers().size();i++) {
			Layer<float>* layer = caffe_net.layers()[i].get();
			if(strcmp(layer->type(),"Convolution")!=0) continue;
			if(j >= layer->blobs()[0].get()->count()) {
//				printf("");
			}
			else {
				printf("%lf", layer->blobs()[0].get()->cpu_data()[j]);
			}
			printf(",");
		}
		printf("\n");
	}

	return 0;
}
#else
int main(int argc, char** argv(){
	LOG(FATAL) <<"This tool requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV

