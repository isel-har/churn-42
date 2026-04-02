# import numpy as np

def compute_class_weight(classes, y):
    y = y.to_numpy()
    n = len(y)
    classes_n = len(classes)
    
    classes_sorted = sorted(classes)
    class_weight= []
    for class_ in classes_sorted:
        weight = float(n /(classes_n * len(y[y == class_])))
        class_weight.append(weight)

    # return sorted(class)
        
