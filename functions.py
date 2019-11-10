## print formatado semelhante ao MATLAB para comparação de resultados
def printMat (arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print("%+.4f" % arr[i][j][0], end = ' ')
        print('')
    print('')
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print("%+.4f" % arr[i][j][1], end = ' ')
        print('')
    print('')
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            print("%+.4f" % arr[i][j][2], end = ' ')
        print('')