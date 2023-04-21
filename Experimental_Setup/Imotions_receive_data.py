import socket
import json
import sys

class FACET:

    def __init__(self) -> None:
        self.socket = None

    def open_connection(self):

        UDP_IP = "134.102.205.35"
        UDP_PORT = 8088

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((UDP_IP, UDP_PORT))
        print("connected: ", self.socket)

        return socket
    
    def receive_data(self):
        '''
        Receives data until a a complete record of AU data is formed
        '''
        BUFFER_SIZE = 8192
        undecoded = self.socket.recv(BUFFER_SIZE)
        decoded = undecoded.decode('utf-8')
        parsed_data = json.loads(decoded)
        

        return parsed_data 
    
    def close_connection(self):
        self.socket.close()

    def extract_AU_data(self):

        data = self.receive_data()
        AUs = {}
        try:
            if data['NoOfFaces']>0:
                AUs['AU01'] = self.transform_data(data['AU1']) if self.transform_data(data['AU1']) > 0.6 else 0    #1
                AUs['AU02'] = self.transform_data(data['AU2']) if self.transform_data(data['AU2']) > 0.6 else 0    #2
                AUs['AU04'] = self.transform_data(data['AU4']) if self.transform_data(data['AU4']) > 0.6 else 0    #3
                AUs['AU05'] = self.transform_data(data['AU5']) if self.transform_data(data['AU5']) > 0.6 else 0    #4
                AUs['AU06'] = self.transform_data(data['AU6']) if self.transform_data(data['AU6']) > 0.6 else 0    #5
                AUs['AU07'] = self.transform_data(data['AU7']) if self.transform_data(data['AU7']) > 0.6 else 0    #6
                AUs['AU09'] = self.transform_data(data['AU9']) if self.transform_data(data['AU9']) > 0.6 else 0    #7
                AUs['AU10'] = self.transform_data(data['AU10']) if self.transform_data(data['AU10']) > 0.6 else 0  #8
                AUs['AU12'] = self.transform_data(data['AU12']) if self.transform_data(data['AU12']) > 0.6 else 0  #9
                AUs['AU14'] = self.transform_data(data['AU14']) if self.transform_data(data['AU14']) > 0.6 else 0  #10
                AUs['AU15'] = self.transform_data(data['AU15']) if self.transform_data(data['AU15']) > 0.6 else 0  #11
                AUs['AU17'] = self.transform_data(data['AU17']) if self.transform_data(data['AU17']) > 0.6 else 0  #12
                AUs['AU18'] = self.transform_data(data['AU18']) if self.transform_data(data['AU18']) > 0.6 else 0  #13
                AUs['AU20'] = self.transform_data(data['AU20']) if self.transform_data(data['AU20']) > 0.6 else 0  #14
                AUs['AU23'] = self.transform_data(data['AU23']) if self.transform_data(data['AU23']) > 0.6 else 0  #15
                AUs['AU24'] = self.transform_data(data['AU24']) if self.transform_data(data['AU24']) > 0.6 else 0  #16
                AUs['AU25'] = self.transform_data(data['AU25']) if self.transform_data(data['AU25']) > 0.6 else 0  #17
                AUs['AU26'] = self.transform_data(data['AU26']) if self.transform_data(data['AU26']) > 0.6 else 0  #18
                AUs['AU28'] = self.transform_data(data['AU28']) if self.transform_data(data['AU28']) > 0.6 else 0  #19
                AUs['AU43'] = self.transform_data(data['AU43']) if self.transform_data(data['AU43']) > 0.6 else 0  #20

            else:
                AUs['AU01'] = 0
                AUs['AU02'] = 0
                AUs['AU04'] = 0
                AUs['AU05'] = 0
                AUs['AU06'] = 0
                AUs['AU07'] = 0
                AUs['AU09'] = 0
                AUs['AU10'] = 0
                AUs['AU12'] = 0 
                AUs['AU14'] = 0
                AUs['AU15'] = 0
                AUs['AU17'] = 0
                AUs['AU18'] = 0
                AUs['AU20'] = 0
                AUs['AU23'] = 0
                AUs['AU24'] = 0
                AUs['AU25'] = 0
                AUs['AU26'] = 0
                AUs['AU28'] = 0
                AUs['AU43'] = 0
        except:
            print(data)
    
            


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


        