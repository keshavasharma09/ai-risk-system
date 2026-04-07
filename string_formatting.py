item = "Apple"
price = 1.237

# %s is replaced by Apple (because it is a string)
# %.2f is replaced by 1.24 (rounded off to two decimal places)
message = "The %s costs $%.2f." %(item, price)

print(message)