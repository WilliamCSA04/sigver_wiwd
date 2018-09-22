def average(array_of_number):
    sum = 0
    for number in array_of_number:
        sum += number
    avg = sum/float(len(array_of_number))
    return avg