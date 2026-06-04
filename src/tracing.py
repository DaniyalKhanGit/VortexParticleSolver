trace_file = open('trace.log', 'w')

def trace(*args):
    print(*args, file=trace_file)
