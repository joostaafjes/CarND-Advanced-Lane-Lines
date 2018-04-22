n = 3
count = 0
for combination in range(1, 2 ** n):
    # if bin(combination).count("1") % 2 == 0:
    #     continue
    comb = ''
    # if bin(combination & (2 ** (n - 0) - 1)).count("1") % 2 == 0:
    #     continue
    if combination & (2**(n-1)) > 0:
        comb += 'n'
    if combination & (2**(n-3)) > 0:
        if comb != '':
            if combination & (2**(n-2)) > 0:
                comb += '&'
            else:
                comb += '|'
        comb += 'x'
    if comb != '':
        print(combination, comb)
        count += 1
print(count)