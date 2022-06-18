#include <torch/script.h> 
#include <iostream>
#include <memory>
#include<vector>
#include<string>
#include <ATen/ATen.h>
#include<time.h>
using namespace torch;

using namespace std;
vector<torch::jit::IValue> get_input_sample(torch::Tensor tensor){
	vector<torch::jit::IValue> tensor_list;
	tensor_list.push_back(tensor);
	
	return tensor_list;
}




int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: torch_resnet_jit <traced_Resnet_50_model.pt>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
   time_t start, end;
    start = time(NULL);
  //module.to(at::kCUDA);
  auto tensor_inputs=get_input_sample(torch::ones({1,3,224,224}));
  at::Tensor output_preds= module.forward(tensor_inputs).toTensor();
  std::cout<<"Outputs"<<output_preds.slice(/*dim=*/1, /*start=*/0, /*end=*/5)<<endl;
  end = time(NULL);
  std::cout<<"Time taken "<<difftime(end, start)<<endl;
  

  std::cout << "ok\n";
}