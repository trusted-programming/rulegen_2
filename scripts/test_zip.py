x = [1,1,3]
y = [4,4,6]

z = list(zip(x,y))
print(z)

t = list(set(z))
print(t)

y = list(zip(*z))
print(y)