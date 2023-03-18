a = 102.6077
b = 1.306483
c = 16.95815

# the point at which the function is almost zero
print (c + log(log(a/0.00001))/b)

# the point at which the function is 100
print (c + log(log(a/100))/b)
