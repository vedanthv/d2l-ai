# 1. Try loading datasets, e.g., Abalone from the UCI Machine Learning Repository and inspect their properties. What fraction of them has missing values? What fraction of the variables is numerical, categorical, or text?

abalone_data = pd.read_csv("../data/chap_2/abalone.data", 
                           names = [
                               "sex", "length", "diameter", "height", 
                               "whole_weight", "shucked_weight",
                               "viscera_weight", "shell_weight",
                               "rings"
                           ]
                          )

abalone_data.describe(include = "all")

# 2. Try out indexing and selecting data columns by name rather than by column number. The pandas documentation on indexing has further details on how to do this.

abalone_data[["sex", "rings", "length"]][ : 20]

# 3. How would you deal with data that has a very large number of categories? What if the category labels are all unique? Should you include the latter?

# If too many categories, try to manually find catgories that are common to each other and group them as one. If they’re all far too different from each other, you’re most likely out of luck, or you can take the information hit and still do the merging of categories to the extent possible

# If the categories are all unique, meaning number of categories == number of samples in dataset, just drop the column, since the column is carrying no useful information, just like a column that only has 1 value. All values are different(if all unique) or same(if all same) no matter the value of the rest of the attributes, there is no pattern to be found here


