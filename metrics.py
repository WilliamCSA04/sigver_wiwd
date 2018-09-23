import math

def average(array_of_number):
    sum = 0
    for number in array_of_number:
        sum += number
    avg = sum/float(len(array_of_number))
    return avg

def standard_deviation(array_of_number, avg = None):
    sum = 0
    if(avg is None):
        avg = average(array_of_number)
    for i in array_of_number:
        sum += (pow(i - avg, 2))
    sd = math.sqrt(sum/float(len(array_of_number)))
    return sd

def false_rejection_rate(genuine_quantity, false_negatives):
    if(false_negatives == 0):
        return 0.0
    return genuine_quantity/float(false_negatives)
