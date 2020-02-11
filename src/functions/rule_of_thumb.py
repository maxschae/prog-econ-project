def rule_of_thumb(data, cutoff, n):
    """ Calculate mean squared error optimal bandwidth to be used in local
        polynomial regression with a rule-of-thumb procedure developed by
        Fan and Gijbels (1996) and modified for the context of Regression
        Discontinuity Design by Imbens and Kalyanaraman (2009). The procedure
        is tailored for local polynomial regression using the boundary optimal
        triangle kernel.

    Args:
        data (pd.DataFrame): Dataframe with data on the running variable in the
                            first column and data on the dependent variable in
                            the second column.
        cutoff (float): Cutpoint in the range of the running variable used to
                        distinguish between treatment and control groups.
        n (float): Number of observations in data.

    Returns:
        h_opt (float): Mean squared error optimal rule-of-thumb bandwidth.
    """
    pass
