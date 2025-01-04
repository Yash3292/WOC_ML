import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CreatePolynomialFeatures:

    def combinations(self, degree):
        pows = []
        for x1 in range(degree+1):
            for x2 in range(degree+1):
                for x3 in range(degree+1):
                    for x4 in range(degree+1):
                        x5 = degree - (x1 + x2 + x3 + x4)
                        if x5 < 0:
                            break
                        else:
                            pows.append((x1, x2, x3, x4, x5))
        return pows

    def poly_transform(self, X, degree):
        if degree == 1:
            result = X
            return result

        elif degree > 1:
            m, n = X.shape

            new_result = np.empty((m, 0))
            prev_result = self.poly_transform(X, degree-1)
            new_result = np.concatenate((new_result, prev_result), axis=1)

            temp_result = np.empty((m, 0))
            powers = self.combinations(degree)
            for pow in powers:
                result_col = X**pow
                result_col = np.prod(result_col, axis=1)
                temp_result = np.concatenate(
                    (temp_result, result_col.reshape(-1, 1)), axis=1)

            new_result = np.concatenate((new_result, temp_result), axis=1)
            return new_result
