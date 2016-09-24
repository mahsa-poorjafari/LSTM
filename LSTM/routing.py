from channels.routing import route
from LSTMTensorflow.consumers import ws_connect, ws_receive, ws_disconnect
# from LSTMTensorflow.views import mnist_data_set
#from LSTMTensorflow.consumers import ws_receive

channel_routing = [
    #route("steps", ws_connect),
    #route("steps", ws_receive),
    #route("steps", mnist_data_set),
    route("websocket.connect", ws_connect),
    route("websocket.receive", ws_receive),
    #route("ws_message", ws_message),
    route("websocket.disconnect", ws_disconnect),

]