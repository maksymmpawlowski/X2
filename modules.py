import random

'''
function returning aesthetic formated data from
dict. This fucntion takes column name we want 
to analyze
'''
def column_describer(column):
    list = []
    list0 = []
    dict = {}
    for x in column:
        list0.append(x)
        if x not in list:
            list.append(x)
    list.sort()
    for x in list:
        dict[x] = list0.count(x)
    for key, value in dict.items():
        print(key, "\t",value)

#new working method
def coldict(df, column):
    list = []
    list0 = []
    dict = {}
    for x in df[column]:
        list0.append(x)
        if x not in list:
            list.append(x)
    list.sort()
    for x in list:
        dict[x] = list0.count(x)
    return dict

#doesn't contain sort so it can iterate NaN values
def coldict1(df, column):
    list = []
    list0 = []
    dict = {}
    for x in df[column]:
        list0.append(x)
        if x not in list:
            list.append(x)
    for x in list:
        dict[x] = list0.count(x)
    return dict

def func_args(df, *args):
    columnlist = df.columns.values.tolist()
    for arg in args:
        if arg in columnlist:
            print('Y')

def entricts(df, *cols):
    columnlist = df.columns.values.tolist()
    colsdict = {}
    for col in cols:
        if col in columnlist:
            colsdict[col] = coldict1(df, col)
    return colsdict

def numtab(rowsquan, colsquan, colvalmin, colvalmax):
    rows = []
    while len(rows) < rowsquan:
        columns = []
        while len(columns) < colsquan:
            column = random.randint(colvalmin,colvalmax)
            columns.append(column)
        rows.append(columns)
    return rows

def columnslist(df):
    list = df.columns.values.tolist()
    return list

def dicttab(df):
    dict = {}
    for column in columnslist(df):
        dict[column] = coldict1(df, column)
    return dict

def quicktab(columns,entries):
    cols = []
    while len(cols) < columns:
        cols.append(1)
    colentries = [cols]
    while len(colentries) < entries:
        colentries.append(cols)
    return colentries