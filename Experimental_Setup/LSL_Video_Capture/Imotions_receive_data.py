import socket
import json

class FACET:

    def __init__(self) -> None:
        self.socket = None

    def open_connection(self):

        TCP_IP = "127.0.0.1"
        TCP_PORT = 8088

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((TCP_IP, TCP_PORT))
        print("connected: ", self.socket)

        return socket
    
    def receive_data(self):
        BUFFER_SIZE = 8192
        data = self.socket.recv(BUFFER_SIZE).decode('utf-8')
        print('data is:{data}')
        parsed_data = json.loads(data) 

        return parsed_data 
    
    def close_connection(self):
        self.socket.close()

    def extract_AU_data(self):

        data = self.receive_data()
        AUs = {}
        AUs['AU1'] = self.transform_data(data['AU1'])
        AUs['AU2'] = self.transform_data(data['AU2'])
        AUs['AU4'] = self.transform_data(data['AU4'])
        AUs['AU5'] = self.transform_data(data['AU5'])
        AUs['AU6'] = self.transform_data(data['AU6'])
        AUs['AU7'] = self.transform_data(data['AU7'])
        AUs['AU9'] = self.transform_data(data['AU9'])
        AUs['AU10'] = self.transform_data(data['AU10'])
        AUs['AU12'] = self.transform_data(data['AU12']) 
        AUs['AU14'] = self.transform_data(data['AU14'])
        AUs['AU15'] = self.transform_data(data['AU15'])
        AUs['AU17'] = self.transform_data(data['AU17'])
        AUs['AU18'] = self.transform_data(data['AU18'])
        AUs['AU20'] = self.transform_data(data['AU20'])
        AUs['AU23'] = self.transform_data(data['AU23'])
        AUs['AU24'] = self.transform_data(data['AU24'])
        AUs['AU25'] = self.transform_data(data['AU25'])
        AUs['AU26'] = self.transform_data(data['AU26'])
        AUs['AU28'] = self.transform_data(data['AU28'])
        AUs['AU43'] = self.transform_data(data['AU43'])

        return AUs
         

    
        

    def transform_data(self,value):
        '''
        Convert from logarithmic scale to 0-1 scale
        '''
        return 1/(1+pow(10,-value))
  


if __name__ == "__main__":

    facet = FACET()
    facet.open_connection()
    data  = facet.extract_AU_data()
    print(data)
    facet.close_connection()


        