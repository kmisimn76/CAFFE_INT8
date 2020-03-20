#include <vector>
#include <string.h>

#include "caffe/layers/conv_layer.hpp"
#include <unistd.h>
#include <sys/time.h>

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

//INT8 Edited
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_int8_cpu(const vector<Blob<char>*>& bottom,
		const vector<Blob<int>*>& top) {
	const char* weight = this->int8_blobs_[0]->cpu_data();
	if(bottom.size()!=1) {
		printf("warning ! : bottom size: %ld != 1\n", bottom.size());
		sleep(10);
	}
	if (this->int8_symmetric_) { //symmetric
		for (int i = 0; i < bottom.size(); i++) {
			const char* bottom_data = bottom[i]->cpu_data();
			int* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; n++) {
				// src/caffe/layers/base_conv_layer.cpp
				this->forward_cpu_gemm_int8_conv(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
				  const int* bias = this->int_blobs_[0]->cpu_data();
				  this->forward_cpu_int_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}
	else { //asymmetric
		for (int i = 0; i < bottom.size(); i++) {
			const char* bottom_data = bottom[i]->cpu_data();
			int* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; n++) {
				// src/caffe/layers/base_conv_layer.cpp
				this->forward_cpu_asymm_gemm_int8_conv(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
				  const int* bias = this->int_blobs_[0]->cpu_data();
				  this->forward_cpu_int_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}
#define PRINT_INT8_LOG
#ifdef PRINT_INT8_LOG
/*	char filename[100];

	FILE* csv = fopen("num.csv", "r");
	int nu = -1;
	char np[100];
	while(!feof(csv))
	{
		int i;
		fscanf(csv, "%d, ", &i);
		fscanf(csv, "%s", np);
		if(strstr(this->layer_param_.name().c_str(), np)!=0) {
			nu = i;
			break;
		}
	}*/
/*
	sprintf(filename, "./results/retinaface/weight/%d_retiina_int8_weight", nu);
	FILE* f = fopen(filename, "ab");
	{
		int tensor[4] = { this->int8_blobs_[0]->num(), this->int8_blobs_[0]->channels(), this->int8_blobs_[0]->height(), this->int8_blobs_[0]->width() };
		int c = tensor[0]*tensor[1]*tensor[2]*tensor[3];
		//fwrite(tensor, sizeof(int), 4, f);
		fwrite(weight, sizeof(char), c, f);
	}
	fclose(f);

	sprintf(filename, "./results/retinaface/input/%d_int8_input", nu);
	f = fopen(filename, "ab");
	{
		int tensor[4] = { bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width() };
		int c = tensor[0]*tensor[1]*tensor[2]*tensor[3];
		//fwrite(tensor, sizeof(int), 4, f);
		fwrite(bottom[0], sizeof(char), c, f);
	}
	fclose(f);

	sprintf(filename, "./results/retinaface/output/%d_int8_output", nu);
	f = fopen(filename, "ab");
	{
		int tensor[4] = { top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width() };
		int c = tensor[0]*tensor[1]*tensor[2]*tensor[3];
		//fwrite(tensor, sizeof(int), 4, f);
		fwrite(top[0], sizeof(int), c, f);
	}
	fclose(f);*/
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_bias_int8_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	//no bias computation
  /*for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
	  for (int k = 0; k < bottom[i]->count(0); k++) {
		  top_data[k] = bottom_data[k];
	  }
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
