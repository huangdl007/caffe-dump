#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DepthsLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::RASEO2_SharePhoneNumbers(bottom, top);
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
			<< "Inputs must have the same dimension.";
		diff_ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void DepthsLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			int count = bottom[0]->count();
			Dtype* bottom0_log = new Dtype(count);
			Dtype* bootom1_log = new Dtype(count);

			//compute log for Y and Y*
			for (int i = 0; i < count; i++)
			{
				bottom0_log[i] = log(bottom[0]->cpu_data()[i]);
				bootom1_log[i] = log(bottom[0]->cpu_data()[i]);
			}

			//compute logY - logY*
			caffe_sub(
				count,
				bottom0_log,
				bootom1_log,
				diff_.mutable_cpu_data());

			//part1 of Loss
			Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
			//part2 of Loss
			Dtype log_sum = Dtype(0);
			Dtype* diff_data = diff_.mutable_cpu_data();
			for (int i = 0; i < count; i++) {
				log_sum += diff_data[i];
			}

			double gamma = 0.5;
			Dtype loss = dot / bottom[0]->num() - gamma * log_sum * log_sum / bottom[0]->num() / bottom[0]->num();

			top[0]->mutable_cpu_data()[0] = loss;

	}

	template <typename Dtype>
	void DepthsLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to label inputs.";
			}
			if (propagate_down[0]) {
				int count = bottom[0]->count();
				int num = bottom[0]->num();

				Dtype log_sum = Dtype(0);
				Dtype* diff_data = diff_.mutable_cpu_data();
				for (int i = 0; i < count; i++) {
					log_sum += diff_data[i];
				}

				Dtype* bottom_diff = bottom[0].mutable_cpu_diff();
				Dtype* bottom_data = bottom[0].mutable_cpu_data();
				for (int i = 0; i < count; i++) {
					bottom_diff[i] = Dtype(2) * diff_data[i] / num / bottom_data[i] - Dtype(2) * gamma * log_sum / num / num / diff_data[i];
				}
				//ta da
			}
	}

} // namespace caffe