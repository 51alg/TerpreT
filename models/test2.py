@Runtime([2], 2)
def Increment(x):
    return (x + 1) % 2

a = Param(2)
b = Param(2)
c = Param(2)

x = Var(2)
y = Var(2)
z = Var(2)
x.set_to(0)

if a == 0:
    y.set_to(Increment(x))
    if b == 0:
        z.set_to(y)
    elif b == 1:
        z.set_to(x)
elif a == 1:
    y.set_to(Increment(x))
    z.set_to(y)
