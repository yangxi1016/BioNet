import os

## Before new test, change the parameter "disease" in the three .py files below.

os.system("python ./gen_edg.py")
os.system("python ./test_disease.py")
os.system("python ./results_translate-s.py")

print("\n\n >>>>>>>>>>>>>> Test Completed. <<<<<<<<<<<<<<<<<\n\n")

