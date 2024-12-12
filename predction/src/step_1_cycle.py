import os

pid = os.getpid()
while True:
    os.system(r"python ./generate_data-txt.py")
    os.system(r"python ./gen_adj-t.py --kill_cycle_pid "+str(pid))



