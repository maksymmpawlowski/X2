
statistic_terms = {
    "variance": "Σ(xi - mean)²/(n - 1)\n\t It's the average difference of data set elements and its mean",
    "standard deviation": "\n\t It's a variance but more understable, more literal",
    "MAD": "Σ[abs(xi - mean)]/(n - 1)\n\t The closest to the truth presentation of average difference of the set elements from its mean",
    "covariance":"Σ[(X_i - X_mean)*(Y_i - Y_mean)*(K...)]/(n - 1)\n\tIt's the average of the products of the deviations of each variable from its respective mean",
    "ANOVA":"At least 2 lists, 5 elements are considered as minimum. \n\tStatistical test used to analyze the differences between the means of two or more sets"}

while True:
    term = input("Term: ")
    if term not in statistic_terms.keys():
        term
    else:
        print("\t",term,"-",statistic_terms[term])


