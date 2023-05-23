set = [321,6,4,83,96,964,2,2,2,62,62,58,42,247]
def percentile(num, list):
    list.sort()
    n = len(list)
    index = (num*n)/100
    rindex = round(index)
    d = index - rindex
    val = (1-d)*list[rindex-1]+d*rindex
    return val
print(percentile(50,set))

def perc2(num, list):
    list.sort()
    n = len(list)
    index = num/100*(n+1)
    rindex = round(index)
    d = (index - rindex) + 1
    val = (1-d)*list[rindex-1] + d*list[rindex-1]
    return val
print(perc2(50,set))



def median(list):
    list.sort()
    centre = len(list)/2
    if len(list) % 2 == 0:
        p1 = list[int(centre)]
        p2 = list[int(centre-1)]
        pp = (p1+p2)/2
        return pp
    else:
        return list[int(centre)]
print(median(set))