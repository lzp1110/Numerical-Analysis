import numpy as np
import fractions as f
import math

##获得最简分数b,c,d
def ConvertFrac(value,frac_value):
    if value%frac_value == 0:
        b,c,d = None,None,None
    elif value < frac_value:
        b = 0
        gcd = math.gcd(value, frac_value)
        c = int(value / gcd)
        d = int(frac_value / gcd)
    else:
        b = int(value/frac_value)
        rest = value - b*frac_value
        gcd = math.gcd(rest,frac_value)
        c = int(rest/gcd)
        d = int(frac_value/gcd)
    return [b,c,d]

def compute_Kw(params):
    if params[0] == None:
        Kw = 1.000000000000001
    else:
        b,c,d = params[0],params[1],params[2]
        qe = b*d + c
        delta_e = 60/qe
        Kd = np.sin(Covert2Rad(qe*delta_e/2))/(qe*np.sin(Covert2Rad(delta_e/2)))
        Kp = np.cos(Covert2Rad(delta_e/2))
        Kw = Kd*Kp
    return Kw

def Covert2Rad(theta):
    return theta/180*np.pi

def main():
    frac_value = []
    value = []
    for i in range(2, 28, 2):
        frac_value.append(i * 3)
    for i in range(3, 33, 3):
        value.append(i)
    result = []
    for i in range(len(value)):  ##对每一个被除数进行运算
        temp = []
        for j in range(len(frac_value)):
               temp.append(compute_Kw(ConvertFrac(value[i],frac_value[j])))
        result.append(temp)
        print(temp)


if __name__ == "__main__":
    main()







