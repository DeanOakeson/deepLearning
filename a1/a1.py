# make an OR gate function
# make an AND gate function
# make a NOR gate function
# make ReLU function
# assemble the XOR gate function
import numpy as np
import sys

def relu(x):
    if x <= 0:
        return 0
    else:
        return 1
        
def AND(x, y):
    out = relu((x + y- 1.5))
    return out

def OR(x, y):
    out = relu((x + y - .5))
    return out

def NOR (x,y):
    out = OR(x,y)
    if out == 0:
        return 1
    if out == 1:
        return 0

def XOR (x,y):
    out = NOR(NOR(x,y), AND(x,y))
    return out

def main(x, y):
    x = int(x)
    y = int(y)
    print(XOR(x,y))



if (__name__ == "__main__"):

    main(sys.argv[1],sys.argv[2])
