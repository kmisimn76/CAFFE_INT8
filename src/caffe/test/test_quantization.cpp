#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class QuantizationTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  QuantizationTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
	  	blob_bottom_2_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()),
   		blob_top_2_(new Blob<Dtype>())
		{

//	blob_top_ = new Blob<Dtype>();
//	blob_top_2_ = new Blob<Dtype>();
	num_output = 3;

    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_min(-1000);
    filler_param.set_max(1000);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    //filler.Fill(this->blob_top_);
    //filler.Fill(this->blob_top_2_);
	Blob<Dtype> test(2,1,num_output,num_output);
	filler_param.set_min(0);
	UniformFiller<Dtype> filler2(filler_param);
	filler2.Fill(&test);	
	activation_scale_factor_ = 0.8;
	for(int i=0;i<num_output;i++)
		weight_scale_factor_.push_back(test.data_at(1,0,i,i));
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top_2_);
  }
  virtual ~QuantizationTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  float activation_scale_factor_;
  int num_output;
  vector<float> weight_scale_factor_;
};

TYPED_TEST_CASE(QuantizationTest, TestDtypesAndDevices);

TYPED_TEST(QuantizationTest, TestQuantization) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_int8_inference(true);
  layer_param.set_activation_scale_factor(this->activation_scale_factor_);

  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);

  vector<Blob<char>*> bottom_int8;
	for (int id = 0; id < this->blob_bottom_vec_.size(); id++) {
  		Blob<char>* bottom_int8_ = new Blob<char>(this->blob_bottom_->shape());
  		bottom_int8.push_back(bottom_int8_);
	}
  
  vector<Blob<Dtype>*> top_int;
	for (int id = 0; id < this->blob_top_vec_.size(); id++) {
  		Blob<Dtype>* top_int_ = new Blob<Dtype>(this->blob_top_->shape());
  		top_int.push_back(top_int_);
	}

  shared_ptr<Layer<Dtype> > layer(new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Quantization_int8(this->blob_bottom_vec_, bottom_int8);


  for(int id = 0; id < this->blob_bottom_vec_.size(); id++){
  	const char* data = bottom_int8[id]->cpu_data();
  	const int count = bottom_int8[id]->count();
  	const Dtype* in_data_a = this->blob_bottom_vec_[id]->cpu_data();
  	const Dtype in_data_b = this->activation_scale_factor_;
  	for (int i = 0; i < count; ++i) {
	  Dtype o = in_data_a[i] * in_data_b;
	  char c = (o>127)?(127):((o<-128)?(-128):((int)o));
  	  EXPECT_NEAR(data[i], c, 1e-5);
	  std::cout<<(int)data[i]<<" "<<(int)c<<"\n";
  	}
  }
}

TYPED_TEST(QuantizationTest, TestDequantization) {
  typedef typename TypeParam::Dtype Dtype;
  /*this->blob_bottom_vec_.push_back(this->blob_bottom_eltwise_);
  LayerParameter layer_param;
  layer_param.mutable_scale_param()->set_axis(0);
  shared_ptr<ScaleLayer<Dtype> > layer(new ScaleLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_bottom_->shape(), this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_eltwise_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-5);
  }*/
}

}  // namespace caffe
