def oper(inp, weight, bias) : # operate / return: ouput
    return inp*weight + bias
def mse(target, output) : # Mean Square Error / return: error
    return ((target-output)**2)/2
def gd(target, inp, weight, lrate, bias) : # 경사강하법(weight) / return: new weight
    return weight - ((oper(inp, weight, bias) - target) * inp * lrate)
def gdb(target, inp, weight, lrate, bias) : # 경사강하법(bias) / return: new bias
    return bias - ((oper(inp, weight, bias) - target) * inp * lrate)

class Neuron :
    def __init__(self, inp, weight, target, lrate, bias):
        self.inp=inp; self.target=target; self.weight=weight; self.lrate=lrate; self.bias=bias;
    def wtrain(self) : # Update weight by gradient descent
        self.weight = gd(self.target, self.inp, self.weight, self.lrate, self.bias)
    def wtrains(self, num) : # Update weight by gradient descent, repeat (num) times
        for i in range(0,num) :
            self.wtrain()
            print("weight train",i+1,':')
            self.view()
    def btrain(self) : # Update bias by gradient descent
        self.bias = gdb(self.target, self.inp, self.weight, self.lrate, self.bias)
    def btrains(self, num) :  # Update bias by gradient descent, repeat (num) times
        for i in range(0, num):
            self.btrain()
            print("bias train", i + 1, ':')
            self.view()
    def view(self):
        out = oper(self.inp, self.weight, self.bias)
        print("input:",self.inp,"weight:",self.weight,"bias:",self.bias,
              "output:",out,"error:",mse(self.target,out))

n1 = Neuron(1,2,3,0.1,0) # input:1, weight:2, target:3, lrate:0.1, bias:0
n1.wtrains(10)
n2 = Neuron(1,2,3,0.1,0) # input:1, weight:2, target:3, lrate:0.1, bias:0
n2.btrains(10)

