#include <torch/torch.h>
#include <iostream>
// Define a new Module.
using namespace torch;

struct smallNet : torch ::nn::Module  {
	
	smallNet(){
		
		ffn1=register_module("ffn1",nn::Linear(784,64));
		ffn2=register_module("ffn2",nn::Linear(64,32));
		ffn3=register_module("ffn3",nn::Linear(32,10));
   	//fc4=register_module("fc4",nn::Linear(16,10));
		
		
	}
	
	Tensor feedforward(Tensor x){
		
		x= relu(ffn1->forward(x.reshape({x.size(0),784})));
		x=dropout(x,/*p=*/0.5,/*train=*/is_training());
		x=relu(ffn2->forward(x));
    //x=dropout(x,/*p=*/0.5,/*train=*/is_training());
    //x=relu(ffn3->forward(x));
		x=log_softmax(ffn3->forward(x),/*dim=*/1);
		return x;
	}
	
	
	nn::Linear ffn1{nullptr}, ffn2{nullptr}, ffn3{nullptr};
	
};


struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
  // Create a new Net.
  auto net = std::make_shared<smallNet>();

  // Create a multi-threaded data loader for the MNIST dataset.
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/64);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  //torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
      torch::optim::Adam optimizer(net->parameters(),/*lr=*/0.01);
  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      
      torch::Tensor prediction = net->feedforward(batch.data);
      //std::cout<<"Tensor in forward pass"<<prediction<<std::endl;
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(net, "net.pt");
      }
    }
  }
}