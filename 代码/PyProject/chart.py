import matplotlib.pyplot as plt
import math
from config import colors,PATH
def paint(*files):
    plt.figure()
    for i in range(0,len(files)):
        file = open(PATH+files[i])
        lines = file.readlines()
        file.close()
        yValue = []
        lastLabel = 0
        lastPos = 0
        plt.subplot(str(len(files))+"1"+str(i+1))
        for line in lines:
            t,x,y,z,l = [(float)(i) for i in line.split(" ")]
            if l == 3 or l==4:
                l = 1
            else:
                l = 0
            yValue.append(math.sqrt(x*x+y*y+z*z))
            if (l != lastLabel):
                plt.plot([i for i in range(lastPos,len(yValue))],yValue[lastPos:],colors[(int)(lastLabel)])
                lastPos = len(yValue)-1
                lastLabel = l
        plt.plot([i for i in range(lastPos,len(yValue))],yValue[lastPos:],colors[(int)(lastLabel)])
        plt.plot([],colors[0],label = "other")
        plt.plot([],colors[1],label = "hand")
    # plt.plot([],colors[2],label = "right clothes")
    # plt.plot([],colors[3],label = "left hand")
    # plt.plot([],colors[4],label = "right hand")
    # plt.plot([],colors[5],label = "left pocket")
    # plt.plot([],colors[6],label = "right pocket")
        plt.legend(loc = "upper left")
    plt.show()
