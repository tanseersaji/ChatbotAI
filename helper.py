import numpy as np

class Helper:
    vocab = None
    char2int = {}
    int2char = {}

    def __init__(self):
        v = open('vocabs.txt','r').read().split('\n')
        self.vocab = ['0']+v
        for i,c in enumerate(self.vocab):
            self.char2int[c] = i
            self.int2char[i] = c

    def _encodeText(self, text):
        text = self._prepairText(text)
        return [self.char2int[c] for c in list(text)]

    def decodeText(self, tensor):
        print
        text = ''.join([self.int2char[i] for i in tensor])
        return text.replace('0','\n')

    def _prepairText(self, text):
        text = text.replace(',',', ')
        text = text.replace('.','. ')
        text = text.replace('\n','0')
        return text
    
    def generateTensor(self, contexts, replys):
        print("Generating Tensors")
        tensorX,tensorY = [],[]
        for i,context in enumerate(contexts):
            tensorX.append(self._encodeText(context))
            tensorY.append(self._encodeText(replys[i]))
        return (np.array(tensorX),np.array(tensorY))

    def maxLength(self, tensors):
        return max([len(tensor) for tensor in tensors])

if __name__ == "__main__":
    helper = Helper()
    exampleText = "Hello, Maho\nHow are you!"

    exampleText = helper.prepairText(exampleText)

    tensor = helper.encodeText(exampleText)
    print("Encoded tensor:",tensor)

    text = helper.decodeText(tensor)
    print("Decoded text:",text)