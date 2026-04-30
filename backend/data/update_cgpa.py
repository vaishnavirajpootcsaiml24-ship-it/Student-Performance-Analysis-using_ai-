# import pandas as pd

# df = pd.read_csv("student_performance.csv")
# df["prev_cgpa"] = df["previous_gpa"] * 2.5

# df.to_csv("students_updated.csv", index=False)

# print("Done! New file created: students_updated.csv")


import pandas as pd

df = pd.read_csv("student_performance.csv")
df["previous_gpa"] = pd.to_numeric(df["previous_gpa"], errors="coerce")

# Update SAME column (change name if needed)
df["previous_gpa"] = (df["previous_gpa"] * 2.5).round(2)

df.to_csv("students_updated3.csv", index=False)

print("Done! Column updated (no new column created)")