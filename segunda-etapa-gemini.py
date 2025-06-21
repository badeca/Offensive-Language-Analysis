import pandas as pd

df1 = pd.read_csv('final_data1.csv')
df2 = pd.read_csv('new_prompt2.csv')
# print(df)

i = 0
# for (line1, line2) in zip(df1.values, df2.values):
#     print("frase original:", line1[1])
#     print("frase reform1:", line1[3])
#     print("frase reform2:", line2[3])
#     print("-" * 50)
#     i += 1
    
for (line2) in df2.values:
    print("frase original:", line2[1])
    print("frase reform2:", line2[3])
    print("class_comparison:", line2[5])
    print("-" * 50)
    i += 1
    
# sentences = df['tweet']
# print(len(sentences))