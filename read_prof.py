import pstats

prof_file = 'C:/Users/QTao/Desktop/Python codes/Res-IRF4/resirf_model.prof'
p = pstats.Stats(prof_file)

print("============= Time Profile =============")

#   - sort_stats('cumulative'): sort by cumulative time, which is the total time spent in this and all subfunctions
#   - print_stats('project', 30): filter out rows with 'project' in the path, and print only the top 30 rows
p.sort_stats('cumulative').print_stats('project', 30)