lines = "4.00,2.00,1.00,3.00,3.00,2.00,2.00,2.00,2.00,2.00,12.00,3.00,3.00,2.00,2.00,2.00,3.00"

s = [float(x) for x in lines.split(",")]
sum = 0
for ss in s:
    sum += ss
print(sum)
