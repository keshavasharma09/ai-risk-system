def decide(prediction, probability):
    if prediction == 1 and probability > 0.8:
        return "Block Transaction"
    elif probability > 0.5:
        return "Manual Review"
    return "Allow"