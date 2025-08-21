import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import sLSTM, mLSTM, SEBlock

class xLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_segments, hidden_dim=512, num_layers=2, dropout=0.5, lstm_type='mLSTM'):
        super(xLSTM, self).__init__()
        self.slstm = sLSTM(input_size, hidden_dim, dropout=dropout)
        self.mlstm = mLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.se_block = SEBlock(input_size)
        self.conv = nn.Conv1d(input_size, hidden_dim, kernel_size=1)
        
        self.attn_linear = nn.Linear(input_size, input_size)
        self.attn_softmax = nn.Softmax(dim=-1)
        
        self.fc = nn.Linear(input_size, output_size)
        self.fc_output = nn.Linear(output_size, 1)
        self.num_segments = num_segments

    def forward(self, x):
        x_slstm = self.slstm(x)
        x_mlstm, attn_weights = self.mlstm(x_slstm)

        # try with and without this
        gate = torch.sigmoid(self.attn_linear(x_slstm + x_mlstm))
        x_combined = gate * x_slstm + (1 - gate) * x_mlstm
        # x_combined = (x_slstm + x_mlstm) / 2
        
        x_se = self.se_block(x_combined.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.fc(x_se)
        output = self.fc_output(output)
        output = output.view(output.size(0), -1)
        
        return output, attn_weights


if __name__ == '__main__':
    pass