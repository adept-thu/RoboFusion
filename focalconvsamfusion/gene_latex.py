def process_data(latex_table):
    latex_tables = ''
    for i, line in enumerate(latex_table):
        columns = line.split(" & ")
        processed_columns = []
        for j, column in enumerate(columns):
            data = column.split(" / ")
            blue_value, red_value = data[0], data[1]
            if float(blue_value) > float(red_value):
                blue_value = "\\textcolor{blue}{ \\textbf{" + blue_value + "}}"
                red_value = "\\textcolor{red}{ " + red_value + "}"
            else:
                red_value = "\\textcolor{red}{ \\textbf{" + red_value + "}}"
                blue_value = "\\textcolor{blue}{" + blue_value + "}"

            columns[j] = f"{blue_value} / {red_value}"
        for k in range(len(columns)-1):
            columns[k] += ' & '
        latex_table[i] = columns
    for i in range(len(latex_table[0])):
        latex_tables += latex_table[0][i]
    return latex_tables

with open('test.txt', 'r') as f:
    latex_table = f.readlines()
# latex_table = """
#          \multicolumn{1}{c|}{}                                & Snow               & 55.07 / 86.67 & 52.98 / 86.53 & 53.08 / 85.92 & 51.14 / 83.59 & 45.02 / 83.65 & 51.46 / 85.29                 \\
#         \multicolumn{1}{c|}{}                                & Rain               & 57.29 / 87.84 & 56.90 / 87.75 & 56.76 / 86.84 & 55.05 / 85.10 & 53.01 / 85.07 & 55.80 / 86.48                 \\
# """

processed_latex_table = process_data(latex_table)
print(processed_latex_table)