import torch

# class PRNetwork(torch.nn.Module):
#     '''
#         Phoneme Recognition Network composed of 3 layers of LSTM and a linear layer for classification.
    
#     '''
#     def __init__(self, input_size = 23, hidden_size = 512, output_size = 43, isTrain = False):

#         super(PRNetwork, self).__init__()
#         self.hidden = None 
#         self.isTrain = isTrain
#         self.lstm = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 3, bidirectional= False, batch_first=True)
#         self.classification = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, output_size),
#         )

#     def reset_hidden(self):
#         self.hidden = None

#     def forward(self, x):
#         x, _, = self.lstm(x)
#         if self.isTrain:
#             x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
#         logits = self.classification(x)
#         return logits

#     def streaming_forward(self, input_frame):
#         lstm_out, self.hidden = self.lstm(input_frame, self.hidden)
#         logits = self.classification(lstm_out)
#         return logits
    
class PRNetwork_large(torch.nn.Module):
    def __init__(self, input_size = 23*5, hidden_size = 512, output_size = 43):
        '''
            Phoneme Recognition Network (large) with feature stack input, hiddensize = 512.        
        '''
        super(PRNetwork_large, self).__init__()
        self.hidden = None 
        self.affine = torch.nn.Linear(input_size, hidden_size)
        self.act1 = torch.nn.Tanh()
        self.lstm = torch.nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers = 3, bidirectional= False, batch_first=True)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, output_size),
        )
    def reset_hidden(self):
        self.hidden = None
    def forward(self, x):
        x = self.affine(x)
        x = self.act1(x)
        x, _ = self.lstm(x)
        logits = self.classification(x)
        return logits
    def streaming_forward(self, x):
        x = self.affine(x)
        x = self.act1(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        logits = self.classification(lstm_out)
        return logits
    

class PRNetwork_lite(torch.nn.Module):
    def __init__(self, input_size = 23*5, hidden_size = 128, output_size = 43):
        '''
            Phoneme Recognition Network (lite) with feature stack input, hiddensize = 128.         
        '''
        super(PRNetwork_lite, self).__init__()
        self.hidden = None 
        self.affine = torch.nn.Linear(input_size, hidden_size)
        self.act1 = torch.nn.Tanh()
        self.lstm = torch.nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers = 3, bidirectional= False, batch_first=True)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, output_size),
        )
    def reset_hidden(self):
        self.hidden = None
    def forward(self, x):
        x = self.affine(x)
        x = self.act1(x)
        x, _ = self.lstm(x)
        logits = self.classification(x)
        return logits
    def streaming_forward(self, x):
        x = self.affine(x)
        x = self.act1(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        logits = self.classification(lstm_out)
        return logits


            
    
