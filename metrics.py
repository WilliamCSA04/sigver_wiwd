import math

def average(array_of_number):
    array_of_number = [x for x in array_of_number if x is not None]
    if(len(array_of_number) == 0):
        return None
    sum = 0
    for number in array_of_number:
        sum += number
    avg = sum/float(len(array_of_number))
    return avg

def standard_deviation(array_of_number, avg = None):
    sum = 0
    array_of_number = [x for x in array_of_number if x is not None]
    if(len(array_of_number) == 0):
        return None
    if(avg is None):
        avg = average(array_of_number)
    for i in array_of_number:
        s = i - avg
        s = pow(s, 2)
        sum += s
    division = sum/float(len(array_of_number))
    sd = math.sqrt(division)
    return sd

def false_rejection_rate(genuine_quantity, false_negatives):
    return float(false_negatives)/genuine_quantity

def false_acceptance_rate(forgery_quantity, false_positives):
    if(forgery_quantity == 0):
        return 0.0
    return float(false_positives)/forgery_quantity