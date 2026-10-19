# ht = tanh(Wx*Xt + Wh*ht-1 + b)

# Wx        weight of current input
# Xt        current input 
# Wh        weight of previus hidden state
# ht-1      Previus hidden state 
# b         bias 
# tanh      Activation function (-1 to 1)
# ht        New hidden state 

import  numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def MarvellousRNNPredictions():
    print("Calculations of RNN") 

    # Food was not good 
    inputs = [1,2,5,3] 

    hidden_state = 0 

    # RNN parameters 
    Wx = 0.5
    Wh = 0.8
    bias = 0.1

    for time_step, x in enumerate(inputs):
        previous_hidden_state = hidden_state

        weighted_input = Wx * x 
        weighted_memory = Wh * previous_hidden_state

        total = weighted_input + weighted_memory + bias 

        hidden_state = np.tanh(total) 

        print("TimeStep : ", time_step+1) 
        print("Input : ", x) 
        print("Hidden State : ", hidden_state) 
        print("-" * 30) 

    print("Final Hidden state : ", hidden_state) 


def main():
    MarvellousRNNPredictions() 


if __name__ == "__main__":
    main() 