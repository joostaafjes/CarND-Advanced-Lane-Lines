n = 9
count = 0
mylist = []
for combination in range(1, 2 ** n):
    comb = ''
    if combination & (2 ** (n - 1)) > 0:
        comb += 'n'
    if combination & (2 ** (n - 3)) > 0:
        if comb != '':
            if combination & (2 ** (n - 2)) > 0:
                comb += '&'
            else:
                comb += '|'
        comb += 'x'
    if combination & (2 ** (n - 5)) > 0:
        if comb != '':
            if combination & (2 ** (n - 4)) > 0:
                comb += '&'
            else:
                comb += '|'
        comb += 'y'
    if combination & (2 ** (n - 7)) > 0:
        if comb != '':
            if combination & (2 ** (n - 6)) > 0:
                comb += '&'
            else:
                comb += '|'
        comb += 'm'
    if combination & (2 ** (n - 9)) > 0:
        if comb != '':
            if combination & (2 ** (n - 8)) > 0:
                comb += '&'
            else:
                comb += '|'
        comb += 'd'
    if comb != '':
        mylist.append(comb)
        count += 1
print(count)
myset = set(mylist)
print(set)
mylist2 = list(myset)
print(mylist2)
print(len(mylist2))

for item in sorted(mylist2):
    print(item)

