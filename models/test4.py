@Runtime([2], 2)
def Increment(x):
    return (x + 1) % 2

a = Param(2)
b = Param(2)
c = Param(2)
d = Param(2)[2]

x = Var(2)
y = Var(2)
z = Var(2)

if a == 0:
    if b == 0:
        if c == 0:
            x.set_to(0)
        elif c == 1:
            x.set_to(0)
    elif b == 1:
        if c == 0:
            x.set_to(0)
        elif c == 1:
            if d == 0:
                x.set_to(0)
            elif d == 1:
                x.set_to(0)
    y.set_to(x)
elif a == 1:
    x.set_to(0)
    y.set_to(x)

