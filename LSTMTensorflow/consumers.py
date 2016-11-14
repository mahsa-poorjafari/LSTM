# In consumers.py
import json
import logging
from json import dumps
from channels import Group
from channels import Channel
from channels.sessions import channel_session
from channels.auth import http_session_user, channel_session_user, channel_session_user_from_http
# log = logging.getLogger(__name__)
import LSTMTensorflow.views
from threading import Thread


def postpone(function):

    def decorator(*args, **kwargs):
        t = Thread(target=function, args=args, kwargs=kwargs)
        # threads or objects running in the same process as the daemon thread.                                                                                                          
        t.daemon = True
        t.start()
        print "------Thread Func-------"
    return decorator


# class Content:
#     def __init__(self, reply_channel):
#         self.reply_channel = reply_channel
#
#     def send(self, json):
#         Channel(self.reply_channel).send({'content': dumps(json)})
# Connected to websocket.connect
#@channel_session
#def ws_add(message):


    #Group("training").add(message.reply_channel)

#@channel_session
@channel_session_user_from_http
def ws_connect(message):
    print ("----------------connect Message now!-------------")

    # print message.user.id
    # path = message.content['path']
    # path = path.replace('/','No')
    # path = path.replace('.','')
    #
    # s.USER_INFO = str(path)
    # print s.USER_INFO
    message.channel_session['steps'] = str(message.content['path'].strip("/"))
    print message.channel_session['steps']
    g_name = message.channel_session['steps']
    Group(g_name).add(message.reply_channel)
    # print (message.content['path'].strip("/"))
    # message.channel_session['room'] = 'train'
    # print(message.channel_session['room'])
    # Group("training-%s" % message.channel_session['room']).add(message.reply_channel)

    # message.reply_channel.send({
    #     "text": json.dumps({
    #         "action": "reply_channel",
    #         "reply_channel": message.reply_channel.name,
    #     })
    # })


# Connected to websocket.receive
@postpone
@channel_session
#@channel_session_user
def ws_receive(message):
    g_name = message.channel_session['steps']
    print '------------- ws_receive-------------------'
    print g_name
    data = json.loads(message.content['text'])
    # print(message.channel_session['room'])
    # label = message.channel_session['room']
    #LSTMTensorflow.views.mnist_data_set(data, label)
    LSTMTensorflow.views.mnist_data_set(data, g_name)
    # Group("train").send({'text': message.content['text']})
    # message.reply_channel.send({
    #         'text': 'Hello',
    #     })
    # print (json.dumps(data['Step']))

    # print ("----------------we are in ws_receive again!-------------")

    # message.reply_channel.send({
    #         "text": message.content['text'],
    #     })
    # try:
    #     message.reply_channel.send({
    #         "text": message.content['text'],
    #     })
    #     data = json.loads(message['text'])
    #     print ("----------------Receive Message now!-------------")
    #     print(data)
    #     log.debug("----------------Log debug Receive Message now!-------------")
    # except ValueError:
    #     log.debug("ws message isn't json text=%s", message['text'])
    #     return


# def ws_message(message):
#     #Channel("websocket.receive").send(message)
#     message.reply_channel.send({
#         'message': message,
#     })
#     print("Receive Message now!")
#     print(message.content['text'])
#     hey = json.loads({ message.content['text']})
#     message.reply_channel.send({"text": message.content['text'],
#     })

    # Group("training").send( {
    #     "text": message.content['text'],
    # })
       #"Step": json.dumps({'Step': message.content['Step'], 'Iter': message.content['Iter']})
       #"text":  message.content['text'],
 #   })

# def train_result(message):
#     data = json.loads(message.content['text'])
#     Group("training").send({'text': json.dumps(data)})

# Connected to websocket.disconnect
@channel_session
#@channel_session_user
def ws_disconnect(message):
    #Group("training-%s" % message.channel_session['room']).discard(message.reply_channel)
    g_name = message.channel_session['steps']
    Group(g_name).discard(message.reply_channel)


# def repeat_me(message):
#     Group("training").send({
#         "Step": json.dumps({'Step': message.content['Step'], 'Iter': message.content['Iter']})
#     })

