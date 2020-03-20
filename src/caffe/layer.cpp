#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);

  //INT8 Edited
  vector<Blob<char>*> bottom_int8;
  vector<Blob<int>*> top_int;
  if (int8_inference_) {
	for (int id = 0; id < bottom.size(); id++) {
  		Blob<char>* bottom_int8_ = new Blob<char>(bottom[id]->shape());
  		bottom_int8.push_back(bottom_int8_);
	}
	for (int id = 0; id < top.size(); id++) {
  		Blob<int>* top_int_ = new Blob<int>(top[id]->shape());
  		top_int.push_back(top_int_);
	}
  }

  switch (Caffe::mode()) {
  case Caffe::CPU:
	//INT8 Edited
  	if (int8_inference_) {
		// in this file
		Quantization_int8(bottom, bottom_int8);
		// src/caffe/layers/conv_layer.cpp .. for convolution layer
		Forward_int8_cpu(bottom_int8, top_int);
		// in this file
		Dequantization_int8(top_int, top);
		//Forward_bias_int8_cpu(top, top);
	}
	else {
		Forward_cpu(bottom, top);
	}
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
	//INT8 Edited
	if (int8_inference_) {
		Quantization_int8_gpu(bottom, bottom_int8);
		Forward_int8_gpu(bottom_int8, top_int);
		Dequantization_int8_gpu(top_int, top);
		Forward_bias_int8_gpu(top, top);
	}
	else {
    	Forward_gpu(bottom, top);
	}
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  
//	std::cout<<"out blob\n";
//	std::cout<<(top[0]->cpu_data()[0])<<"\n";
//	std::cout<<(top[0]->cpu_data()[1])<<"\n";
/*
	char filename[200];
	sprintf(filename, "./results/%s/%s_output", (!int8_inference_)?("output_fp"):((int8_symmetric_)?("output_sym"):("output_asym")), layer_param_.name().c_str());
	FILE* f = fopen(filename, "w");
		{
		int tensor[4] = { top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width() };
		int c = tensor[0]*tensor[1]*tensor[2]*tensor[3];
		fprintf(f, "%d %d %d %d\n", tensor[0], tensor[1], tensor[2], tensor[3]);
		for(int i=0;i<c;i++) {
			fprintf(f, "%lf\n", top[0]->cpu_data()[i]);
			//fwrite(top[0], sizeof(int), c, f);
		}
		fprintf(f,"\n");
	}
	printf("end\n");
	int aa;
	scanf("%d", &aa);
*/
  //INT8 Edited
  if (int8_inference_) {
	for (int id = 0; id < bottom.size(); id++) {
  		delete bottom_int8[id];
	}
	for (int id = 0; id < top.size(); id++) {
  		delete top_int[id];
	}
  }
  return loss;
}


//INT8 Edited
// TODO: use ACL
// Quantization bottom blobs in CPU
template <typename Dtype>
void Layer<Dtype>::Quantization_int8(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<char>*>& bottom_int8) {
	for (int n = 0; n < bottom.size(); n++) {
		const Dtype* bottom_data = bottom[n]->cpu_data();
		if(int8_symmetric_) {
			signed char* int8_data = (signed char*)(bottom_int8[n]->mutable_cpu_data());
			const int count = bottom[n]->count();
			float res;
			for (int i = 0; i < count; ++i) {
				res = (bottom_data[i] * activation_scale_factor_);
				if (res>=0.f) res += 0.5;
				else res -= 0.5;

				if (res>127)
					int8_data[i] = 127;
				else if (res<-128)
					int8_data[i] = -128;
				else
					int8_data[i] = (signed char)res;
			}
		}
		else {
			unsigned char* int8_data = (unsigned char*)(bottom_int8[n]->mutable_cpu_data());
			const int count = bottom[n]->count();
			float res;
			for (int i = 0; i < count; ++i) {
				res = (bottom_data[i] * activation_scale_factor_) + activation_zero_point_;
				if (res>=0.f) res += 0.5;
				else res -= 0.5;

				if (res>255)
					int8_data[i] = 255;
				else if (res<0)
					int8_data[i] = 0;
				else
					int8_data[i] = (unsigned char)res;
			}
		}
	}
}
// Dequantization top blobs in CPU
template <typename Dtype>
void Layer<Dtype>::Dequantization_int8(const vector<Blob<int>*>& top_int,
		  const vector<Blob<Dtype>*>& top) {
	for (int n = 0; n < top_int.size(); n++) {
		const int* int_data = top_int[n]->cpu_data();
		Dtype* top_data = top[n]->mutable_cpu_data();

		const int nums = top_int[n]->shape()[0];
		const int channels = top_int[n]->shape()[1];
		const int size = top_int[n]->shape()[2] * top_int[n]->shape()[3];
		CHECK_EQ(channels, weight_scale_factor_.size()) << "Dequant: Filter channel number error";
		for (int num = 0; num < nums; num++) {
			for (int c = 0; c < channels; c++) {
				for (int i = 0; i < size; i++) {
						float factor = 1.0 / (activation_scale_factor_ * weight_scale_factor_[c]);
						if(weight_scale_factor_[c] == 0.0) factor = 0;
						*top_data = (Dtype)(*int_data) * factor;
						top_data++;
						int_data++;
				}
			}
		}
	}
}

// Quantization bottom blobs in GPU
template <typename Dtype>
void Layer<Dtype>::Quantization_int8_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<char>*>& bottom_int8) {
	// incompletion
	Quantization_int8(bottom, bottom_int8);
}
// Dequantization top blobs in GPU
template <typename Dtype>
void Layer<Dtype>::Dequantization_int8_gpu(const vector<Blob<int>*>& top_int,
		  const vector<Blob<Dtype>*>& top) {
	Dequantization_int8(top_int, top);
}



INSTANTIATE_CLASS(Layer);

}  // namespace caffe
