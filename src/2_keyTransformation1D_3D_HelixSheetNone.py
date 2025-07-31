# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:46:57 2021

author: Titli
editor: Poorya
Code for calculating 1D and 3D keys
** Keep in Code Folder: 
    sample_details.csv ('protein', 'chain')
    aminoAcidCode_grouping_type0.txt 
    aminoAcidCode_lexicographic_new.txt
    amino_codes.txt
"""
import os, time
import math
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np 
import argparse
import pickle 

start_time = time.time()

dTheta = 30
dLen = 35
numOfLabels = 20
dT = 26 # 1D distance bins

# Bin definitions
THETA_BINS = [12.11, 17.32, 21.53, 25.21, 28.54, 31.64, 34.55, 37.34, 40.03,
              42.64, 45.17, 47.64, 50.05, 52.43, 54.77, 57.08, 59.38, 61.64,
              63.87, 66.09, 68.30, 70.50, 72.69, 79.2, 81.36, 83.51, 85.67,
              87.80, 90.01]

DIST3D_BINS = [3.83, 7.00, 9.00, 11.00, 14.00, 17.99, 21.25, 23.19, 24.8, 26.26,
               27.72, 28.9, 30.36, 31.62, 32.76, 33.84, 35.13, 36.26, 37.62,
               38.73, 40.12, 41.8, 43.41, 45.55, 47.46, 49.69, 52.65, 55.81,
               60.2, 64.63, 70.04, 76.15, 83.26, 132.45]

DIST1D_BINS = [1, 2, 3, 4, 6, 9, 13, 17, 23, 30, 39, 50, 64, 81, 102, 129, 163,
               205, 258, 324, 406, 509, 638, 799, 999]

def find_bin(value, bins, inclusive=False):
    """Return 1-based index of the bin for value."""
    for idx, b in enumerate(bins, 1):
        if (value <= b) if inclusive else (value < b):
            return idx
    return len(bins) + 1

parser = argparse.ArgumentParser(description='Key calculation - 1D to 3D with HELIX/SHEET combination type.')
parser.add_argument('sample_details', default='sample_details.csv', type=str, help='Enter sample details file name')
parser.add_argument('data_dir', type=str, help='Enter data dir where pdb files are located')
parser.add_argument('--outputs', nargs='*', default=['3D'],
                    choices=['3D', '1D', 'triplets', 'sequence'],
                    help='Output file types to generate: 3D, 1D, triplets, sequence')
args = parser.parse_args()

data_dir = args.data_dir #'./../Dataset/'
subdir = data_dir + 'lexicographic/' # output dir
if not os.path.exists(subdir):
    os.makedirs(subdir)

df= pd.read_csv(args.sample_details,sep=',',header=0)
df = df.fillna('')
df['protchain'] = df['protein']+'_'+df['chain']
files = df['protchain'].tolist()

# Theta bin for 3D 
def thetaClass_(Theta):
    if Theta <= 0.0001:  # collinear
        return 30
    return find_bin(Theta, THETA_BINS)

# maxDist bin for 3D 
def dist12Class_(dist12):
    return find_bin(dist12, DIST3D_BINS)

def calcDist(indexLabel1,indexLabel2):
    x1=xCord[indexLabel1]
    x2=xCord[indexLabel2]
    y1=yCord[indexLabel1]
    y2=yCord[indexLabel2]
    z1=zCord[indexLabel1]
    z2=zCord[indexLabel2]
    distance=(((x1-x2)**2+(y2-y1)**2+(z2-z1)**2)**0.5)
    return distance

def indexFind(index_of_2,i1,j1,k1):
    if index_of_2==i1:
        indexOf0=j1
        indexOf1=k1
    elif index_of_2==j1:
        indexOf0=i1
        indexOf1=k1
    elif index_of_2==k1:
        indexOf0=i1
        indexOf1=j1

    return indexOf0, indexOf1

# DEFINE 1D DISTANCE BINS (26 bins)
def distClass1D(dist12):
    return find_bin(dist12, DIST1D_BINS, inclusive=True)

# SEPERATE HELIX and SHEET PARTS FOR EACH PROTEIN, SAVE IN A PICKLE FILE
def seperate_helix_sheet_chain(file):
    helix_dict={} #position:chain 
    sheet_dict={} #position:chain
    
    # read file for helix and sheet
    with open(data_dir+file+'.pdb', 'r') as f:
        for line in f:
            
            if line[0:6].strip()=='HELIX':
                serNum=line[7:10].strip()
                chain=line[19:20].strip()
                start=int(line[21:25].strip())
                stop=int(line[33:37].strip())
                h_range=list(range(start,stop+1))
                for pos in h_range:
                    pos1 = str(pos)+chain
                    if pos1 not in helix_dict:
                        helix_dict[pos1]='HELIX_'+str(serNum)

            if line[0:6].strip()=='SHEET':
                serNum=line[7:10].strip()
                chain=line[21:22].strip()
                start=int(line[22:26].strip())
                stop=int(line[33:37].strip())
                s_range=list(range(start,stop+1))
                for pos in s_range:
                    pos1 = str(pos)+chain
                    if pos1 not in sheet_dict:
                        sheet_dict[pos1]='SHEET_'+str(serNum)

    fh=open(data_dir+file+"_helix_dict.pkl","wb")     
    fs=open(data_dir+file+"_sheet_dict.pkl","wb")     
    pickle.dump(helix_dict,fh)
    pickle.dump(sheet_dict,fs)
    fh.close()
    fs.close()
    f.close()
    print("Helix and Sheet seperation done.")

# FIND TYPE ASSIGNMENT FOR EACH TRIPLETS (HELIX, SHEET, NONE)
def determine_type(triplets):
    code=0
    t0 = triplets[0]
    t1 = triplets[1]
    t2 = triplets[2]
    
    # 3 SAME
    if (t0==t1==t2):
        if t0.split("_")[0]=='HELIX': # 3a1 - 3 vertices from same helix 
            code = '1_3a1'
        if t0.split("_")[0]=='SHEET': # 3b1 - 3 vertices from same sheet 
            code = '7_3b1'
        if t0.split("_")[0] =='NONE': # 3c - all vertices from none, no helix or sheet 
            code = '17_3c'
            
    # 3 DIFF h/s/n
    if(t0.split("_")[0]!=t1.split("_")[0]!=t2.split("_")[0]):
        code = '18_1a1b1c' # 1a1b1c - all different, I helix, 1 sheet, 1 none   
    
    # 3 SAME TYPE, DIFF NUM
    if (t0.split("_")[0]==t1.split("_")[0]==t2.split("_")[0]):
        if(t0.split("_")[0]=='HELIX' and t0.split("_")[1]!=t1.split("_")[1]!=t2.split("_")[1]):
            code = '3_3a3' # 3a3 - 3 vertices from three different helices
        if(t0.split("_")[0]=='SHEET' and t0.split("_")[1]!=t1.split("_")[1]!=t2.split("_")[1]):
            code = '9_3b3' # 3a3 - 3 vertices from three different sheets
            
    # 2 SAME, 1 DIFF.  
    if(t0==t1 and t1!=t2) or (t1==t2 and t2!=t0) or (t0==t2 and t1!=t2):
        if(t0==t1 and t1!=t2):
            one=t0
            two=t1
            three=t2
        if(t1==t2 and t2!=t0):
            one=t1
            two=t2
            three=t0
        if(t0==t2 and t1!=t2):
            one=t0
            two=t2
            three=t1
        if one.split("_")[0]=='HELIX' and three.split("_")[0]=='HELIX': #3a2 - 2 vertices from same helix, 1 vertex from another helix        
            code = '2_3a2'
        if one.split("_")[0]=='HELIX' and three.split("_")[0]=='SHEET': #2a11b  2 vertices from same helix, 1 vertex from sheet
            code = '14_2a11b'
        if one.split("_")[0]=='HELIX' and three.split("_")[0]=='NONE': #2a11c   2 vertices from same helix, 1 vertex from none 
            code = '4_2a11c'
            
        if one.split("_")[0]=='SHEET' and three.split("_")[0]=='SHEET': #3a2 - 2 vertices from same sheet, 1 vertex from another sheet        
            code = '8_3b2' 
        if one.split("_")[0]=='SHEET' and three.split("_")[0]=='HELIX': #2b11a  2 vertices from same sheet, 1 vertex from helix
            code = '16_2b11a'
        if one.split("_")[0]=='SHEET' and three.split("_")[0]=='NONE': # 2b11c  2 vertices from same sheet, 1 vertex from none 
            code = '10_2b11c'
            
        if one.split("_")[0]=='NONE' and three.split("_")[0]=='HELIX': #1a2c  1 vertex from helix, two vertices from none 
            code = '6_1a2c' 
        if one.split("_")[0]=='NONE' and three.split("_")[0]=='SHEET': #1b2c  1 vertex from sheet, two vertices from none   
            code = '12_1b2c'
            
    # 2 SAMEDIFF, 1 DIFF.
    if((t0.split("_")[0]==t1.split("_")[0] and t1.split("_")[0]!=t2.split("_")[0]) or \
       (t1.split("_")[0]==t2.split("_")[0] and t2.split("_")[0]!=t0.split("_")[0]) or \
       (t2.split("_")[0]==t0.split("_")[0] and t0.split("_")[0]!=t1.split("_")[0])):
        
        if (t0.split("_")[0]==t1.split("_")[0] and t1.split("_")[0]!=t2.split("_")[0]):
            one_ = t0
            two_ = t1
            three_ = t2
        if (t1.split("_")[0]==t2.split("_")[0] and t2.split("_")[0]!=t0.split("_")[0]):
            one_ = t1
            two_ = t2
            three_ = t0
        if (t2.split("_")[0]==t0.split("_")[0] and t0.split("_")[0]!=t1.split("_")[0]):
            one_ = t2
            two_ = t0
            three_ = t1
            
        if(one_.split("_")[0]=='HELIX' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='SHEET'):
            code = '13_2a21b'  # 2a21b  2 vertices from different helices, 1 vertex from sheet
        if(one_.split("_")[0]=='HELIX' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='NONE'):
            code = '5_2a21c'  # 2a21c   2 vertices from two different helices, 1 vertex from none 
            
        if(one_.split("_")[0]=='SHEET' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='HELIX'):
            code = '15_2b21a'  # 2b21a  2 vertices from different sheets, 1 vertex from helix 
        if(one_.split("_")[0]=='SHEET' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='NONE'):
            code = '11_2b21c' # 2b21c   2 verticed from two different sheets, 1 vertex from none 
    
    return code

# ENUMERATE AMINO ACID 3-letter CODES to 1-letter CODES
dfa = pd.read_csv("Amino_Acid_Codes.txt", header=0, sep="\t")
aaCode3to1Dict = pd.Series(dfa.One_letter_Code.values,index=dfa.Three_letter_Code).to_dict()

aminoAcidCode=open("aminoAcidCode_lexicographic_new.txt","r")
aminoAcidLabel={}
for amino in aminoAcidCode:
    amino=amino.split()
    aminoAcidLabel[amino[0]]=int(amino[1])
aminoAcidCode.close()

#for fileName in files:
def parallelcode(fileName, chain, outputs):
    filesDict3D = {} if '3D' in outputs else {}
    filesDict1D = {} if '1D' in outputs else {}
    types = [0] * 18  # array of len 18 to save 18 type combination of helix/sheet for each key
    if not os.path.exists(data_dir+fileName+'.pdb'):
        pass
    inFile = open(data_dir+fileName+'.pdb', 'r')

    fileKey3D = open(subdir+fileName+"_"+chain+".3Dkeys_theta30_maxdist35", "w") if '3D' in outputs else None
    fileKey1D = open(subdir+fileName+"_"+chain+".1Dkeys_dist26", "w") if '1D' in outputs else None
    fileTriplets = open(subdir+fileName+"_"+chain+".triplets_theta30_maxdist35_dist26_HELIX_SHEET", "w") if 'triplets' in outputs else None
    outfile_aaSeq = open(subdir+fileName+"_"+chain+".aminoAcidSequence", "w") if 'sequence' in outputs else None

    # Write Header !
    if fileKey3D:
        fileKey3D.writelines("key3D\t1_3a1\t2_3a2\t3_3a3\t4_2a11c\t5_2a21c\t6_1a2c \t7_3b1\t8_3b2\t9_3b3\t10_2b11c\t11_2b21c\t12_1b2c\t13_2a21b\t14_2a11b\t15_2b21a\t16_2b11a\t17_3c\t18_1a1b1c\n")
    if fileKey1D:
        fileKey1D.writelines("key1D\tfreq\n")
    if fileTriplets:
        fileTriplets.writelines("key3D\taa0\tpos0\taa1\tpos1\taa2\tpos2\tclassT1\tTheta\tclassL1\tmaxDist\tx0\ty0\tz0\tx1\ty1\tz1\tx2\ty2\tz2\tTheta1\theight\tkey1D\tpos0_1D\tpos1_1D\tpos2_1D\td1\td1Class\td2\td2Class\td3\td3Class\ttype\n")

    global xCord, yCord, zCord
    aminoAcidName={}
    xCord={}
    yCord={}
    zCord={}
    seq_number={}
    counter=0
    helix_dict=pickle.load(open(data_dir+fileName+"_helix_dict.pkl","rb"))
    sheet_dict=pickle.load(open(data_dir+fileName+"_sheet_dict.pkl","rb"))
    
    for i in inFile:
        if ((i[0:6].rstrip()=="ENDMDL") or (i[0:6].rstrip()=='TER' and i[21].rstrip()==chain)):
            break
        if (i[0:6].rstrip()=="MODEL" and int(i[10:14].rstrip())>1):
            break
        
        if(i[0:4].rstrip())=="ATOM"and(i[13:15].rstrip())=="CA"and(i[16]=='A' or i[16]==' ') and i[21:22].strip()==chain and i[17:20].strip()!= "UNK" :
            #print (i)
            
            if i[22:27].strip().isdigit() == False: # check if seq number is '123AB' or '123'
                if not ((i[22:27].strip().startswith("-"))==True and (i[22:27].strip()[1:].isdigit())==True):
                    continue
            
            aminoAcidName[counter]=int(aminoAcidLabel[i[17:20].strip()])
            xCord[counter]=(float(i[30:38].strip()))
            yCord[counter]=(float(i[38:46].strip()))
            zCord[counter]=(float(i[46:54].strip()))
            seq_number[counter]=str(i[22:27].strip())
            if outfile_aaSeq:
                outfile_aaSeq.write(aaCode3to1Dict[i[17:20].strip()].strip())
            counter+=1
    if outfile_aaSeq:
        outfile_aaSeq.close()

    protLen=len(yCord)
    print(protLen)
    initialLabel=[]
    sortedLabel=[]
    sortedIndex=[]
    for m in range(0,3):
        initialLabel.append(0)
        sortedLabel.append(0)
        sortedIndex.append(0)
    for i in range(0,protLen-2):
        for j in range(i+1,protLen-1):
            for k in range(j+1, protLen):
                global i1,j1,k1
                i1=i
                j1=j
                k1=k
                keepLabelIndex={}
                keepLabelIndex[aminoAcidName[i]]=i
                keepLabelIndex[aminoAcidName[j]]=j
                keepLabelIndex[aminoAcidName[k]]=k
                initialLabel[0]=aminoAcidName[i]
                initialLabel[1]=aminoAcidName[j]
                initialLabel[2]=aminoAcidName[k]
                sortedLabel=list(initialLabel)
                sortedLabel.sort(reverse=True)
                if (sortedLabel[0]==sortedLabel[1])and(sortedLabel[1]==sortedLabel[2]):
                    dist1_2Temp=calcDist(i,j)
                    dist1_3Temp=calcDist(i,k)
                    dist2_3Temp=calcDist(j,k)
                    if dist1_2Temp>=(max(dist1_2Temp,dist1_3Temp,dist2_3Temp)):
                        indexOf0=i
                        indexOf1=j
                        indexOf2=k
                    elif dist1_3Temp>=(max(dist1_2Temp,dist1_3Temp,dist2_3Temp)):
                        indexOf0=i
                        indexOf1=k
                        indexOf2=j
                    else:
                        indexOf0=j
                        indexOf1=k
                        indexOf2=i
                elif(aminoAcidName[i]!=aminoAcidName[j])and(aminoAcidName[i]!=aminoAcidName[k])and(aminoAcidName[j]!=aminoAcidName[k]):
                    for index_ in range(0,3):
                        sortedIndex[index_]=keepLabelIndex[sortedLabel[index_]]
                    indexOf0=sortedIndex[0]
                    indexOf1=sortedIndex[1]
                    indexOf2=sortedIndex[2]
                elif(sortedLabel[0]==sortedLabel[1])and(sortedLabel[1]!=sortedLabel[2]):
                    indexOf2=keepLabelIndex[sortedLabel[2]]
                    indices=indexFind(indexOf2,i,j,k)
                    a=indexOf2
                    b=indices[0]
                    c=indices[1]
                    dist1_3Temp=calcDist(b,a)
                    dist2_3Temp=calcDist(c,a)
                    if dist1_3Temp>=dist2_3Temp:
                        indexOf0=indices[0]
                        indexOf1=indices[1] 
                    else:
                        indexOf0=indices[1]
                        indexOf1=indices[0]
                elif(sortedLabel[0]!=sortedLabel[1])and(sortedLabel[1]==sortedLabel[2]):
                    indexOf0=keepLabelIndex[sortedLabel[0]]
                    indices=indexFind(indexOf0,i,j,k)
                    if calcDist(indexOf0,indices[0])>= calcDist(indexOf0,indices[1]):
                        indexOf1=indices[0]
                        indexOf2=indices[1] 
                    else:
                        indexOf2=indices[0]
                        indexOf1=indices[1]
                        
                dist01=calcDist(indexOf0,indexOf1)
                s2=dist01/2
                dist02=calcDist(indexOf0,indexOf2)
                s1=dist02
                dist03=calcDist(indexOf1,indexOf2)
                maxDist=max(dist01,dist02,dist03)
                # s3 = median height
                s3=(((xCord[indexOf0]+xCord[indexOf1])/2-xCord[indexOf2])**2+((yCord[indexOf0]+yCord[indexOf1])/2-yCord[indexOf2])**2+((zCord[indexOf0]+zCord[indexOf1])/2-zCord[indexOf2])**2)**0.5
                Theta1=180*(math.acos((s1**2-s2**2-s3**2)/(2*s2*s3)))/3.14
                if Theta1<=90:
                    Theta=Theta1
                else:
                    Theta=abs(180-Theta1)
                classT1=thetaClass_(Theta)
                classL1=dist12Class_(maxDist)

                ##getting the positions of AminoAcids in sequence
                position0 = str(list(seq_number.values())[indexOf0])
                position1 = str(list(seq_number.values())[indexOf1])
                position2 = str(list(seq_number.values())[indexOf2])

                aacd0 = list(aminoAcidLabel.keys())[list(aminoAcidLabel.values()).index(aminoAcidName[indexOf0])]
                aacd1 = list(aminoAcidLabel.keys())[list(aminoAcidLabel.values()).index(aminoAcidName[indexOf1])]
                aacd2 = list(aminoAcidLabel.keys())[list(aminoAcidLabel.values()).index(aminoAcidName[indexOf2])]
                
                # GET HELIX or SHEET info
                seqList = [str(position0)+chain, str(position1)+chain, str(position2)+chain]
                typeList = []
                for s in seqList:
                    if s in helix_dict:
                        typeList.append(helix_dict[s])
                    elif s in sheet_dict:
                        typeList.append(sheet_dict[s])
                    else:
                        typeList.append('NONE_0')
                type_ = determine_type(typeList) # func call determine helix/sheet/none combination type for each triangle

                x0 = str(xCord.get(indexOf0))
                y0 = str(yCord.get(indexOf0))
                z0 = str(zCord.get(indexOf0))

                x1 = str(xCord.get(indexOf1))
                y1 = str(yCord.get(indexOf1))
                z1 = str(zCord.get(indexOf1))

                x2 = str(xCord.get(indexOf2))
                y2 = str(yCord.get(indexOf2))
                z2 = str(zCord.get(indexOf2))

                # GENERATE 3D key
                key3D = dLen*dTheta*(numOfLabels**2)*(aminoAcidName[indexOf0]-1)+\
                        dLen*dTheta*(numOfLabels)*(aminoAcidName[indexOf1]-1)+\
                        dLen*dTheta*(aminoAcidName[indexOf2]-1)+\
                        dTheta*(classL1-1)+\
                        (classT1-1)
                        
                # 3D ke-freq file generation with type of helix/sheet
                if key3D in filesDict3D:
                    index = int(type_.split("_")[0])-1
                    types[index] += 1
                    filesDict3D[key3D] = types #+=1
                else:
                    types = [0] * 18
                    index = int(type_.split("_")[0])-1
                    types[index] += 1
                    filesDict3D[key3D]=types
                    
                if '1D' in outputs or 'triplets' in outputs:
                    # 1D CALCULATIONS - determine central, lower and higher points by aa position in seq
                    positions1D = sorted([int(seq_number[i]), int(seq_number[j]), int(seq_number[k])])
                    pos0_1D = positions1D[0]
                    pos1_1D = positions1D[1]
                    pos2_1D = positions1D[2]

                    # calculate d1, d2 from central point
                    d1 = abs(pos1_1D - pos0_1D)
                    d2 = abs(pos2_1D - pos1_1D)
                    d3 = abs(pos2_1D - pos0_1D)

                    d1Class = distClass1D(d1)
                    d2Class = distClass1D(d2)
                    d3Class = distClass1D(d3)

                    indexOf01D = (list(seq_number.keys())[list(seq_number.values()).index(str(pos0_1D))])
                    indexOf11D = (list(seq_number.keys())[list(seq_number.values()).index(str(pos1_1D))])
                    indexOf21D = (list(seq_number.keys())[list(seq_number.values()).index(str(pos2_1D))])

                    L1 = aminoAcidName[indexOf01D]
                    L2 = aminoAcidName[indexOf11D]
                    L3 = aminoAcidName[indexOf21D]

                    key1D = dT*dT*(numOfLabels**2)*(L1-1)+\
                            dT*dT*(numOfLabels)*(L2-1)+\
                            dT*dT*(L3-1)+\
                            dT*(d1Class-1)+\
                            (d3Class-1)

                    if '1D' in outputs:
                        if key1D in filesDict1D:
                            filesDict1D[key1D] += 1
                        else:
                            filesDict1D[key1D] = 1

                    if 'triplets' in outputs:
                        line = (str(key3D)+"\t"+\
                                str(aacd0)+"\t"+str(position0)+"\t"+str(aacd1)+"\t"+str(position1)+"\t"+str(aacd2)+"\t"+str(position2)+"\t"+\
                                str(classT1)+"\t"+str(Theta)+"\t"+str(classL1)+"\t"+str(maxDist)+"\t"+\
                                x0+"\t"+y0+"\t"+z0+"\t"+x1+"\t"+y1+"\t"+z1+"\t"+x2+"\t"+y2+"\t"+z2+"\t"+\
                                str(Theta1)+"\t"+str(s3)+"\t"+\
                                str(key1D)+"\t"+\
                                str(pos0_1D)+"\t"+str(pos1_1D)+"\t"+str(pos2_1D)+"\t"+\
                                str(d1)+"\t"+str(d1Class)+"\t"+str(d2)+"\t"+str(d2Class)+"\t"+str(d3)+"\t"+str(d3Class)+"\t"+type_+"\n")
                        fileTriplets.writelines(line)
                
    ## Write lines in key-freq file with helix sheet type
    if fileKey3D:
        for value_ in filesDict3D:
            types_ = ("\t").join([str(x) for x in filesDict3D[value_]])
            fileKey3D.writelines([str(value_),'\t', types_,'\n'])

    if fileKey1D:
        for value_ in filesDict1D:
            fileKey1D.writelines([str(value_),'\t', str(filesDict1D[value_]),'\n'])

    print ("FILENAME=",fileName,'\t',"NUM OF AMINOACIDS=",protLen, '\t',"TIME(min)=",np.round((time.time()-start_time)/60,2))
    if fileKey3D:
        fileKey3D.close()
    if fileKey1D:
        fileKey1D.close()
    if fileTriplets:
        fileTriplets.close()
    os.remove(data_dir+fileName+"_helix_dict.pkl")
    os.remove(data_dir+fileName+"_sheet_dict.pkl")
    ## end of parallelcode()

# FUNC CALLS
def maincode(file):  # file is name with prot and chain
    prot = file.split("_")[0]
    chain = file.split("_")[1]
    seperate_helix_sheet_chain(prot) #func call
    parallelcode(prot, chain, args.outputs)

num_cores = multiprocessing.cpu_count()
print("#of cores = ", num_cores)

Parallel(n_jobs=num_cores, verbose=50)(delayed(maincode)(fileName)for fileName in files)

print("CODE END\t TOTAL TIME=(min)", np.round((time.time()-start_time)/60,2))

