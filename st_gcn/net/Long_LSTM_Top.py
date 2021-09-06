import torch.nn as nn
import torch

win_size = [154, 150, 145, 103, 99, 77]
stride = [145, 145, 145, 98, 98, 74]
start_time = [1, 5, 10, 1, 5, 1]

# win_size = [198, 170, 150, 100, 70, 40]
# stride = [100, 100, 100, 50, 45, 25]
# start_time = [1, 1, 1, 1, 1, 1]


class Long_LSTM_Top(torch.nn.Module):
    def __init__(self, is_training, input_size, hidden_size, class_size):
        super().__init__()
        self.num_steps = 300
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_size = class_size

        num_LSTMs_0 = len(range(start_time[0], self.num_steps - win_size[0] + start_time[0], stride[0]))
        num_LSTMs_1 = len(range(start_time[1], self.num_steps - win_size[1] + start_time[1], stride[1]))
        num_LSTMs_2 = len(range(start_time[2], self.num_steps - win_size[2] + start_time[2], stride[2]))

        self.mL0 = LSTM_Long_0(input_size, hidden_size, class_size)
        #self._initial_state = self.mL0.initial_state
        self.fc1 = nn.Linear((num_LSTMs_0 + num_LSTMs_1*0 + num_LSTMs_2*0) * self.hidden_size, self.class_size)

    def forward(self, x):


        x = self.mL0(x)

        #output_depthconcat_long_0 = self.mL0.get_depth_concat_output()
        logits = self.fc1(x)

        return logits

class LSTM_Long_0(torch.nn.Module):
    def __init__(self, input_size, hidden_size, class_size):

        super().__init__()

        #self.batch_size = batch_size = config.batch_size
        #self.num_steps = num_steps = config.num_steps  #76

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_size = class_size
        self.num_steps = 300
        #input shape(batch_size, seq_length, input_size)
        #output shape(batch_size, seq_length, output_size)
        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        #inputs = input_data
        #self._initial_state = torch.zeros(1, self.batch_size, self.hidden_size)

    def forward(self, inputs):

        ''' #파이토치언어 변경필요
        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        '''
        #inputs shape = (batch_size, sequence_length, channel) = (12, 300, 256)
        batch_size, sequence_length, channel = inputs.size()

        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device='cuda:0')
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device='cuda:0')
        outputs_0 = []
        state_0 = state_0 = []

        win_size_0 = win_size[0]  # 75 frames
        stride_0 = stride[0]  # df 1
        start_time_0 = start_time[0]  # 1
        num_LSTMs_0 = len(range(start_time_0, self.num_steps - win_size_0 + start_time_0, stride_0))

        #print(range(start_time_0, self.num_steps - win_size_0 + start_time_0, stride_0))
        #print("num_LSTMs_0: ", num_LSTMs_0)

        for time_step in range(start_time_0, self.num_steps):
            for win_step in range(num_LSTMs_0):
                if time_step == start_time_0:
                    outputs_0.append([])
                    state_0.append([])
                    state_0[win_step] = (h0, c0) #(h0,c0)

                if time_step < start_time_0 + win_step * stride_0:
                    cell_output = torch.zeros(batch_size, channel).to(device='cuda:0')
                    outputs_0[win_step].append(cell_output)

                elif time_step >= start_time_0 + win_step * stride_0 and time_step < start_time_0 + win_step * stride_0 + win_size_0:

                    #shape = (batch, channel)
                    cell_output =(inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :])
                    outputs_0[win_step].append(cell_output)
                    #(cell_output, state_0[win_step]) = self.lstm(lstm_input, state_0[win_step])
                    #self.outputs_0[win_step].append(cell_output)
                else:
                    cell_output = torch.zeros(batch_size, channel).to(device='cuda:0')
                    outputs_0[win_step].append(cell_output)

        #shape of self.output_0[win_step] = [time_step, batch, hidden] = [200, 34, 100]
        output_0 = []
        for win_step in range(num_LSTMs_0):
            outputs_0_temp = torch.stack(outputs_0[win_step], dim=0).to(device='cuda:0')

            output_0.append([])
            #output shape : [batch, seqence_length, hidden]
            output_0[win_step], _ = self.lstm(outputs_0_temp.permute(1,0,2).contiguous(), state_0[win_step])
            temp_list = [output_0[win_step].permute(1,0,2).contiguous()]
            output_0[win_step] = torch.cat(temp_list, dim=1).view(-1, self.hidden_size)
            #output_0[win_step] = tf.reshape(tf.concat(self.outputs_0[win_step], 1), [-1, size])

        temp_output_0 = []
        for win_step in range(num_LSTMs_0):
            temp_output_0.append([])
            temp_output_0[win_step] = output_0[win_step].view(batch_size, self.num_steps - start_time_0, self.hidden_size)
            #temp_output_0[win_step] = tf.reshape(output_0[win_step], [batch_size, num_steps - start_time_0, size])
            if win_step == 0:
                input_0 = temp_output_0[win_step]
            else:
                input_0 = torch.cat([input_0, temp_output_0[win_step]], dim=1)
                #input_0 = tf.concat([input_0, temp_output_0[win_step]], 1)
        input_0 = input_0.view(batch_size, num_LSTMs_0, self.num_steps - start_time_0, self.hidden_size)
        #input_0 = tf.reshape(input_0, [batch_size, num_LSTMs_0, num_steps - start_time_0, size])

        concat_output_real_0 = input_0.sum(dim=2)
        #concat_output_real_0 = tf.reduce_sum(input_0, 2)
        #shape is [12, 100]
        out_concat_output_real_0 = concat_output_real_0.view(batch_size, num_LSTMs_0*self.hidden_size)
        #self.out_concat_output_real_0 = tf.reshape(concat_output_real_0, [batch_size, num_LSTMs_0 * size])

        return out_concat_output_real_0

    # def get_depth_concat_output(self):
    #     return self.out_concat_output_real_0

'''
class LSTM_Long_1(torch.nn.module):
    def __init__(self, input_size, hidden_size):
        suepr().__init__()

    def forward(self, x):




class LSTM_Long_2(torch.nn.module):
    def __init__(self, input_size, hidden_size):
        suepr().__init__()

    def forward(self, x):

'''