from util import *
import os.path
from os import path



rows_str=[["3 3 0 2 0 1 3 1 0 0 3 1 0 1 0 3 0 0 1 1 0"],
]

for row_str in rows_str:
    for i in row_str:
        row_str=i.replace(" ","")
        row=[]

        row.append([int(row_str[0])])
        row.append([int(row_str[1]),int(row_str[2])])
        row.append([int(row_str[3]),int(row_str[4]),int(row_str[5])])
        row.append([int(row_str[6]),int(row_str[7]),int(row_str[8]),int(row_str[9])])
        row.append([int(row_str[10]),int(row_str[11]),int(row_str[12]),int(row_str[13]),int(row_str[14])])
        row.append([int(row_str[15]),int(row_str[16]),int(row_str[17]),int(row_str[18]),int(row_str[19]),int(row_str[20])])

        g,_=decode_ENAS_to_igraph(row)
        if(not(path.exists("software/enas/outputs_dvae_"+row_str+","))):
            os.mkdir("software/enas/outputs_dvae_"+row_str+",")
        draw_network(g,"software/enas/outputs_dvae_"+row_str+",/"+row_str+".pdf")
    
