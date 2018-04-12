import tensorflow as tf

class Param():
    def __init__(self):
        self.batch_size = 64
        self.conv_size = [[48, 5],[64, 5],[96, 3]]
        self.keep_prob = 0.3
        self.cell_size = 128
        self.cell_num = 3
        self.class_num = 0
        self.learning_rate = 1e-3

class Model():
    def __init__(self, param):

        self.if_train = tf.placeholder(tf.bool)
        self.model_variables = []

        def _add_conv_layers(inks, lengths):
            convolved = inks
            _, _, in_channels = inks.get_shape()
            in_channels = in_channels.value
            for i in range(len(param.conv_size)):
                out_channels = param.conv_size[i][0]
                kernel_size = param.conv_size[i][1]
                filters = tf.Variable(tf.random_normal((kernel_size, in_channels, out_channels)))
                convolved = tf.nn.conv1d(convolved, filters=filters, stride=1, padding='SAME', data_format='NWC')

                bias = tf.Variable(tf.zeros(out_channels))
                convolved = tf.nn.bias_add(convolved, bias)

                self.model_variables.append(filters)
                self.model_variables.append(bias)

                if i < (len(param.conv_size) - 1):
                    convolved = tf.layers.dropout(convolved, rate=param.keep_prob, training=self.if_train)
            
                in_channels = out_channels

            return convolved, lengths

        def _add_rnn_layers(convolved, lengths):
            cell = tf.nn.rnn_cell.BasicLSTMCell

            cells_fw = [cell(param.cell_size) for _ in range(param.cell_num)]
            cells_bw = [cell(param.cell_size) for _ in range(param.cell_num)]

            #todo use tensorboard to see bidirectional rnn
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=convolved, sequence_length=lengths, dtype=tf.float32)

            for cell in cells_fw:
                self.model_variables.extend(cell.variables)
            for cell in cells_bw:
                self.model_variables.extend(cell.variables)
            
            mask = tf.tile(tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2), [1, 1, tf.shape(outputs)[2]])
            zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))

            outputs = tf.reduce_sum(zero_outside, axis=1)

            return outputs
        
        self.input_inks = tf.placeholder(tf.float32, (None, None, 3))
        self.input_labels = tf.placeholder(tf.int64)
        self.input_lengths = tf.placeholder(tf.int64)
        inks, labels, lengths = self.input_inks, self.input_labels, self.input_lengths

        convolved, lengths = _add_conv_layers(inks, lengths)
        rnn_outputs = _add_rnn_layers(convolved, lengths)
        _, size = rnn_outputs.get_shape()
        out_w = tf.Variable(tf.random_normal((size.value, param.class_num)))
        out_b = tf.Variable(tf.zeros(param.class_num))

        self.model_variables.append(out_w)
        self.model_variables.append(out_b)

        output = tf.matmul(rnn_outputs, out_w) + out_b
        self.inks = inks
        self.convolved = convolved

        self.predictions = tf.argmax(output, axis=1)

        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))
        optimizer = tf.train.AdamOptimizer(param.learning_rate)
        self.train_op = optimizer.minimize(self.cross_entropy)
        self.optimizer_variables = optimizer.variables()
        correct_pred = tf.equal(self.predictions, tf.squeeze(labels))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
