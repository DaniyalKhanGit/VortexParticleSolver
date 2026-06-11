trace_file = open('trace.log', 'w')

def trace(*args):
    print(*args, file=trace_file, flush=True)

def exportCSV(timestep, error, reset=False):
    mode = "w" if reset else "a"
    with open("errors.csv", mode) as f:
        if reset:
            f.write("timestep,error\n")
        f.write(f"{timestep},{error}\n")