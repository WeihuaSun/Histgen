a = {
    1:"aaaa",
    2:"bbbb",
}

b = {
    3:"bbbb"
}

aaaa =  a[1]

del a[1]

b[1] = aaaa

print(a)
print(b)
