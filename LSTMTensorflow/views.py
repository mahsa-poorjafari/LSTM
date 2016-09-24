from django.shortcuts import render
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from django.http import JsonResponse
import numpy as np
import json
from channels import Channel
from channels import Group
import LSTM.settings as s
from channels.sessions import channel_session
from django.views import View
import LSTMTensorflow.consumers

def index(request):
    # user = request.META['USER']
    return render(request, 'index.html')


def RNN(x, weights, biases, n_input, n_steps, n_hidden):
    print '------------we are in RNN NOW! ------------'
    # print 'x:', x
    # print 'weights:', weights
    # print 'biases:', biases
    # print 'n_input:', n_input
    # print 'n_steps:', n_steps
    # print 'n_hidden:', n_hidden

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps ---- Permutes the dimensions according to perm.
    # tf.transpose(a, perm=None, name='transpose')
    # a: A Tensor.
    # perm: A permutation of the dimensions of a.
    # name: A name for the operation (optional).
    x = tf.transpose(x, [1, 0, 2])

    # Reshaping to (n_steps*batch_size, n_input)
    # Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
    # tf.reshape(tensor, shape, name=None)
    # tensor: A Tensor
    # shape: A Tensor of type int32. Defines the shape of the output tensor. -1 is inferred to be n_input:
    x = tf.reshape(x, [-1, n_input])

    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # Splits a tensor into num_split tensors along one dimension.
    # tf.split(split_dim, num_split, value, name='split')
    # split_dim: A 0-D int32 Tensor. The dimension along which to split. Must be in the range [0, rank(value)).
    # num_split: A Python integer. The number of ways to split.
    # value: The Tensor to split.
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # Multiplies matrix a by matrix b, producing a * b.
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def mnist_data_set(request):
    print '------------we are in mnist_data_set NOW! ------------'
    learning_rate = 0
    training_iters= 0
    batch_size= 0
    display_step= 0
    n_input= 0
    n_steps= 0
    n_hidden= 0
    n_classes= 0
    traning_array = np.chararray([30])
    # if request.is_ajax():
    # if request.method == 'POST':
        #print 'Raw Data: "%s"' % request.body
    #print 'Raw Data: "%s"' % request

    #received_json_data = json.loads(request.body)
    # print '------------Parameters------------'
    # learning_rate = float(str(received_json_data[1]['value']))  # for POST form method
    # print 'learning_rate: ', learning_rate
    # training_iters = long(str(received_json_data[2]['value']))  # for POST form method
    # print 'training_iters: ', training_iters
    # batch_size = int(str(received_json_data[3]['value']))  # for POST form method
    # print 'batch_size: ', batch_size
    # display_step = int(str(received_json_data[4]['value']))  # for POST form method
    # print 'display_step: ', display_step
    # n_input = int(str(received_json_data[5]['value']))  # for POST form method
    # print 'n_input: ', n_input
    # n_steps = int(str(received_json_data[6]['value']))  # for POST form method
    # print 'n_steps: ', n_steps
    # n_hidden = int(str(received_json_data[7]['value']))  # for POST form method
    # print 'n_hidden: ', n_hidden
    # n_classes = int(str(received_json_data[8]['value']))  # for POST form method
    # print 'n_classes: ', n_classes

    # received_json_data = json.loads(request.content['text'])
    #received_json_data = json.dumps(request)
    received_json_data = request
    #received_json_data = json.loads(received_json_data)
    print 'received_json_data:', received_json_data
    # print '------------Parameters------------'
    learning_rate = float(received_json_data['learning_rate'])  # for POST form method
    #print 'learning_rate: ', learning_rate
    training_iters = long(received_json_data['training_iters'])  # for POST form method
    #print 'training_iters: ', training_iters
    batch_size = int(received_json_data['batch_size'] )  # for POST form method
    #print 'batch_size: ', batch_size
    display_step = int(received_json_data['display_step'])  # for POST form method
    #print 'display_step: ', display_step
    n_input = int(received_json_data['n_input'])  # for POST form method
    #print 'n_input: ', n_input
    n_steps = int(received_json_data['n_steps'])  # for POST form method
    #print 'n_steps: ', n_steps
    n_hidden = int(received_json_data['n_hidden'])  # for POST form method
    #print 'n_hidden: ', n_hidden
    n_classes = int(received_json_data['n_classes'])  # for POST form method
    #print 'n_classes: ', n_classes

    if learning_rate != 0 and training_iters!= 0 and batch_size!= 0 and display_step!= 0:

        try:
            # tf Graph input ----------- Feeding the Graph
            x = tf.placeholder("float", [None, n_steps, n_input])
            y = tf.placeholder("float", [None, n_classes])
            # Define weights --------When you train a model, you use variables to hold and update parameters
            # Create a variables.
            weights = {
                'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([n_classes]))
            }
        except TypeError, err:
            print 'It returned None instead:', err

        pred = RNN(x, weights, biases, n_input, n_steps, n_hidden)
        #print '=================Pred=============='
        #print pred


        # Define loss and optimizer

        #tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
        # Computes softmax cross entropy between logits and labels.
        # logits: Unscaled log probabilities.
        # labels: Each row labels[i] must be a valid probability distribution.
        # Reduces input_tensor along the dimensions given in reduction_indices.
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        # Optimizer that implements the Adam algorithm.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model

        # Returns the truth value of (x == y) element-wise.
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()


        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

                    traning_array.append = str(step * batch_size)
                    traning_array.append= "{:.6f}".format(loss)
                    traning_array.append = "{:.5f}".format(acc)

                    print "Step --> " + str(step) + ", Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)
                    data = {

                        'Step': str(step), "Iter": str(step * batch_size), "Minibatch_Loss": "{:.6f}".format(loss),
                        "Training_Accuracy": "{:.5f}".format(acc)
                    }
                    # Group("training-%s" % label).send({'text': json.dumps(message)})
                    # Channel("steps").reply_channel.send({
                    #     "message": data,
                    # })
                    #Channel('ws_message').send({'message': data})
                    Group("train").send({
                        'text': json.dumps(data),
                    })
                    # LSTMTensorflow.consumers.ws_message(message)
                    print ("----------------Message has been sent from mnist_data_set now!-------------")
                    #message.reply_channel.send({"text": data})
                step += 1
            print "Optimization Finished!"

            # Calculate accuracy for 128 mnist test images
            test_len = 128
            test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
            test_label = mnist.test.labels[:test_len]
            print "Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label})

    return render(request, 'index.html')

    # if pred != 0:
    #     data = {
    #                  'msg' : "Training is Done!"}
    #     return JsonResponse(data, safe=False)
    # else:
    #     return render(request, 'index.html')
    # print "------showclassifications--------"
    #     # print 'showclassifications: ', json.dumps(showclassifications)
    #
    #     if request.is_ajax():
    #         print "------------------request.is_ajax-------------"
    #         # template = 'index.html'
    #         # preds_seri = json.dumps(preds, cls=NumpyAwareJSONEncoder)
    #
    #         # dbn_seri = json.dumps(dbn, separators=(','))
    #         # showclassifications_br = string.replace(showclassifications, '\n', '<br\>')
    #         print '---------showclassifications_br---------'
    #
    #         data = {
    #             'preds' : pred.tolist()}
    #         #data_json = json.dumps(data)
    #         # return render_to_response(template, data,  context_instance = RequestContext(request))
    #         # return render_to_response(template , {'dbn': dbn, 'preds': preds, 'showclassifications': showclassifications})
    #         # return HttpResponse(json.dumps({'data': data}), content_type="application/json")
    #
    #         return JsonResponse(data, safe=False)




