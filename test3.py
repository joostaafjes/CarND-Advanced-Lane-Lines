import operator
from itertools import product

methods = ['n','x','y','m','d','']
oper = ['|','&','']

a = product(methods, oper, methods, oper, methods, oper, methods, oper, methods)

max_length=3

def f(x):
    z = []
    for y in filter(str.isalpha, x):
       z.append(y)
    return len(z) == len(set(z))

# remove double letters
it = filter(f, a)

mylist = []
for i in it:
    txt = i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8]
    if len(txt) == 0:
        continue
    if len(txt) > max_length:
        continue
    if not str.isalpha(txt[0]) or not str.isalpha(txt[-1]):
        continue
    if len(txt) > 1 and str.isalpha(txt):
        continue
    if len(txt) % 2 == 0:
        continue
    skip = False
    if len(txt) > 1:
        for index in range(len(txt) - 1):
            if str.isalpha(txt[index]) and str.isalpha(txt[index + 1]):
                skip = True
                break
            if not str.isalpha(txt[index]) and not str.isalpha(txt[index + 1]):
                skip = True
                break
    if skip:
        continue
    if txt[0] > txt[-1]:
        txt = txt[::-1]
    mylist.append(txt)

print(len(mylist))
mylist = sorted(list(set(mylist)))

for i in mylist:
    print(i)

print(len(mylist))


