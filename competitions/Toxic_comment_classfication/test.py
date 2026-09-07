from sys import getsizeof

sum = 0
for i in range(5):
    if i == 2:
        i += 1
    sum += i
print(sum)

sum = 0
for n in range(1, 101):
    sum = sum + (1/2)**n
print(sum)

def get_diff(sum, sum_old):
    return sum - sum_old

sum = 0
sum_old = 0
n = 1
while True:
    sum += (1/2)**n
    # diff = sum - sum_old
    if get_diff(sum, sum_old) < 1e-16:
        break
    sum_old = sum
    n += 1
print(sum)
print(n)
# print(diff)

a = 3
while True:
    b = a**2
    
    if (b-a) <= 3:
        break
    
    a = b
    
for i in range(5):
    print(i, 'hi')