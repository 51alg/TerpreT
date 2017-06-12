@Runtime([2], 2)
def Increment(x):
    return (x + 1) % 2

a = Param(2)
b = Param(2)
c = Param(2)
rule = Param(2)[2]

x = Var(2)
y = Var(2)
z = Var(2)

if a == 0:
    if b == 0:
        x.set_to(c)
    elif b == 1:
        x.set_to(Increment(c))
    y.set_to(x)
elif a == 1:
    x.set_to(0)
    y.set_to(x)

