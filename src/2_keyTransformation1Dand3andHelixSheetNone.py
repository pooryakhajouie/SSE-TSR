# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:46:57 2021

@author: titli
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

parser = argparse.ArgumentParser(description='Key calculation - 1D to 3D with HELIX/SHEET combination type.')
parser.add_argument('sample_details', default='sample_details.csv',type=str, help='Enter sample details file name')
parser.add_argument('data_dir', type=str, help='Enter data dir where pdb files are located')
args = parser.parse_args()

data_dir = args.data_dir #'./../Dataset/'
subdir = data_dir + 'lexiographic/' # output dir
if not os.path.exists(subdir):
    os.makedirs(subdir)

df= pd.read_csv(args.sample_details,sep=',',header=0)
df = df.fillna('')
df['protchain'] = df['protein']+'_'+df['chain']
files = df['protchain'].tolist()
#print(len(files), files)

# Theta bin for 3D 
def thetaClass_(Theta):
    #classT=0
    if Theta<=0.0001: # collinear
        classT=30
    elif Theta>0.0001 and Theta<12.11:
        classT=1
    elif Theta>=12.11 and Theta<17.32:
        classT=2
    elif Theta>=17.32 and Theta<21.53:
        classT=3
    elif Theta>=21.53 and Theta<25.21:
        classT=4
    elif Theta>=25.21 and Theta<28.54:
        classT=5
    elif Theta>=28.54 and Theta<31.64:
        classT=6
    elif Theta>=31.64 and Theta<34.55:
        classT=7
    elif Theta>=34.55 and Theta<37.34:
        classT=8
    elif Theta>=37.34 and Theta<40.03:
        classT=9
    elif Theta>=40.03 and Theta<42.64:
        classT=10
    elif Theta>=42.64 and Theta<45.17:
        classT=11
    elif Theta>=45.17 and Theta<47.64:
        classT=12
    elif Theta>=47.64 and Theta<50.05:
        classT=13
    elif Theta>=50.05 and Theta<52.43:
        classT=14
    elif Theta>=52.43 and Theta<54.77:
        classT=15
    elif Theta>=54.77 and Theta<57.08:
        classT=16
    elif Theta>=57.08 and Theta<59.38:
        classT=17
    elif Theta>=59.38 and Theta<61.64:
        classT=18
    elif Theta>=61.64 and Theta<63.87:
        classT=19
    elif Theta>=63.87 and Theta<66.09:
        classT=20
    elif Theta>=66.09 and Theta<68.30:
        classT=21
    elif Theta>=68.30 and Theta<70.5:
        classT=22
    elif Theta>=70.5 and Theta<72.69:
        classT=23
    elif Theta>=72.69 and Theta<79.2:
        classT=24
    elif Theta>=79.2 and Theta<81.36:
        classT=25
    elif Theta>=81.36 and Theta<83.51:
        classT=26
    elif Theta>=83.51 and Theta<85.67:
        classT=27
    elif Theta>=85.67 and Theta<87.80:
        classT=28
    elif Theta>=87.80 and Theta<=90.00:
        classT=29
    return classT

# maxDist bin for 3D 
def dist12Class_(dist12):
    #classL=0
    if (dist12<3.83):
        classL=1
    elif dist12>=3.83 and dist12<7.00:
        classL=2
    elif dist12>=7.00 and dist12<9.00:
        classL=3
    elif dist12>=9.00 and dist12<11.00:
        classL=4
    elif dist12>=11.00 and dist12<14.00:
        classL=5
    elif dist12>=14.00 and dist12<17.99:
        classL=6
    elif dist12>=17.99 and dist12<21.25:
        classL=7
    elif dist12>=21.25 and dist12<23.19:
        classL=8
    elif dist12>=23.19 and dist12<24.8:
        classL=9
    elif dist12>=24.8 and dist12<26.26:
        classL=10
    elif dist12>=26.26 and dist12<27.72:
        classL=11
    elif dist12>=27.72 and dist12<28.9:
        classL=12
    elif dist12>=28.9 and dist12<30.36:
        classL=13
    elif dist12>=30.36 and dist12<31.62:
        classL=14
    elif dist12>=31.62 and dist12<32.76:
        classL=15
    elif dist12>=32.76 and dist12<33.84:
        classL=16
    elif dist12>=33.84 and dist12<35.13:
        classL=17
    elif dist12>=35.13 and dist12<36.26:
        classL=18
    elif dist12>=36.26 and dist12<37.62:
        classL=19
    elif dist12>=37.62 and dist12<38.73:
        classL=20
    elif dist12>=38.73 and dist12<40.12:
        classL=21
    elif dist12>=40.12 and dist12<41.8:
        classL=22
    elif dist12>=41.8 and dist12<43.41:
        classL=23
    elif dist12>=43.41 and dist12<45.55:
        classL=24
    elif dist12>=45.55 and dist12<47.46:
        classL=25
    elif dist12>=47.46 and dist12<49.69:
        classL=26
    elif dist12>=49.69 and dist12<52.65:
        classL=27
    elif dist12>=52.65 and dist12<55.81:
        classL=28
    elif dist12>=55.81 and dist12<60.2:
        classL=29
    elif dist12>=60.2 and dist12<64.63:
        classL=30
    elif dist12>=64.63 and dist12<70.04:
        classL=31
    elif dist12>=70.04 and dist12<76.15:
        classL=32
    elif dist12>=76.15 and dist12<83.26:
        classL=33
    elif dist12>=83.26 and dist12<132.45:
        classL=34
    elif dist12>=132.45:
        classL=35
    return classL

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
    if (dist12<=1):
        classL=1
    elif dist12>1 and dist12<=2:
        classL=2
    elif dist12>2 and dist12<=3:
        classL=3
    elif dist12>3 and dist12<=4:
        classL=4
    elif dist12>4 and dist12<=6:
        classL=5
    elif dist12>6 and dist12<=9:
        classL=6
    elif dist12>9 and dist12<=13:
        classL=7
    elif dist12>13 and dist12<=17:
        classL=8
    elif dist12>17 and dist12<=23:
        classL=9
    elif dist12>23 and dist12<=30:
        classL=10
    elif dist12>30 and dist12<=39:
        classL=11
    elif dist12>39 and dist12<=50:
        classL=12
    elif dist12>50 and dist12<=64:
        classL=13
    elif dist12>64 and dist12<=81:
        classL=14
    elif dist12>81 and dist12<=102:
        classL=15
    elif dist12>102 and dist12<=129:
        classL=16
    elif dist12>129 and dist12<=163:
        classL=17
    elif dist12>163 and dist12<=205:
        classL=18
    elif dist12>205 and dist12<=258:
        classL=19
    elif dist12>258 and dist12<=324:
        classL=20
    elif dist12>324 and dist12<=406:
        classL=21
    elif dist12>406 and dist12<=509:
        classL=22
    elif dist12>509 and dist12<=638:
        classL=23
    elif dist12>638 and dist12<=799:
        classL=24
    elif dist12>799 and dist12<=999:
        classL=25
    elif dist12>999:
        classL=26
    return classL

# SEPERATE HELIX and SHEET PARTS FOR EACH PROTEIN, SAVE IN A PICKLE FILE
def seperate_helix_sheet_chain(file):
    helix_dict={} #position:chain 
    sheet_dict={} #position:chain
    
    # read file for helix and sheet
    with open(data_dir+file+'.pdb', 'r') as f:
        for line in f:
            #print(line)
            #line = line.strip().split()
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
                    #else:
                        #helix_dict[pos1].append('HELIX_'+str(serNum))
                #print("\n", line,"\n", chain, start, stop, h_range)
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
                    #else:
                        #sheet_dict[pos1].append('SHEET_'+str(serNum)) 
                #print("\n", line,"\n", chain, start, stop, h_range)
    fh=open(data_dir+file+"_helix_dict.pkl","wb")     
    fs=open(data_dir+file+"_sheet_dict.pkl","wb")     
    pickle.dump(helix_dict,fh)
    pickle.dump(sheet_dict,fs)
    fh.close()
    fs.close()
    #print(helix_dict, "\n", sheet_dict, "\n", chains)
    f.close()
    print("Helix and Sheet seperation done.")

# FIND TYPE ASSIGNMENT FOR EACH TRIPLETS (HELIX, SHEET, NONE)
def determine_type(triplets):
    code=0
    t0 = triplets[0]
    t1 = triplets[1]
    t2 = triplets[2]
    #print (triplets, t0,t1,t2)
    
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
        if one.split("_")[0]=='HELIX' and three.split("_")[0]=='SHEET': #2a11b 	2 vertices from same helix, 1 vertex from sheet
            code = '14_2a11b'
        if one.split("_")[0]=='HELIX' and three.split("_")[0]=='NONE': #2a11c 	2 vertices from same helix, 1 vertex from none 
            code = '4_2a11c'
            
        if one.split("_")[0]=='SHEET' and three.split("_")[0]=='SHEET': #3a2 - 2 vertices from same sheet, 1 vertex from another sheet				
            code = '8_3b2' 
        if one.split("_")[0]=='SHEET' and three.split("_")[0]=='HELIX': #2b11a 	2 vertices from same sheet, 1 vertex from helix
            code = '16_2b11a'
        if one.split("_")[0]=='SHEET' and three.split("_")[0]=='NONE': # 2b11c 	2 vertices from same sheet, 1 vertex from none 
            code = '10_2b11c'
            
        if one.split("_")[0]=='NONE' and three.split("_")[0]=='HELIX': #1a2c 	1 vertex from helix, two vertices from none 
            code = '6_1a2c' 
        if one.split("_")[0]=='NONE' and three.split("_")[0]=='SHEET': #1b2c 	1 vertex from sheet, two vertices from none		
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
            code = '13_2a21b'  # 2a21b 	2 vertices from different helices, 1 vertex from sheet
        if(one_.split("_")[0]=='HELIX' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='NONE'):
            code = '5_2a21c'  # 2a21c 	2 vertices from two different helices, 1 vertex from none 
            
        if(one_.split("_")[0]=='SHEET' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='HELIX'):
            code = '15_2b21a'  # 2b21a 	2 vertices from different sheets, 1 vertex from helix	
        if(one_.split("_")[0]=='SHEET' and one_.split("_")[1]!=two_.split("_")[1] and three_.split("_")[0]=='NONE'):
            code = '11_2b21c' # 2b21c 	2 verticed from two different sheets, 1 vertex from none 
    
    return code

# ENUMERATE AMINO ACID 3-letter CODES to 1-letter CODES
dfa = pd.read_csv("Amino_Acid_Codes.txt", header=0, sep="\t")
aaCode3to1Dict = pd.Series(dfa.One_letter_Code.values,index=dfa.Three_letter_Code).to_dict()
#print(aaCode3to1Dict)

aminoAcidCode=open("aminoAcidCode_lexicographic_new.txt","r")
aminoAcidLabel={}
for amino in aminoAcidCode:
    amino=amino.split()
    aminoAcidLabel[amino[0]]=int(amino[1])
aminoAcidCode.close()

#for fileName in files:
def parallelcode(fileName, chain):
    filesDict3D={}
    filesDict1D={}
    types = [0] * 18  # array of len 18 to save 18 type combination of helix/sheet for each key, each element of array is freq of key i of type j 
    if not os.path.exists(data_dir+fileName+'.pdb'):
        pass
    inFile=open(data_dir+fileName+'.pdb','r')
    
    fileKey3D = open(subdir+fileName+"_"+chain+".3Dkeys_theta30_maxdist35", "w")
    fileKey1D = open(subdir+fileName+"_"+chain+".1Dkeys_dist26", "w")
    fileTriplets = open(subdir+fileName+"_"+chain+".triplets_theta30_maxdist35_dist26_HELIX_SHEET","w")
    outfile_aaSeq = open(subdir+fileName+"_"+chain+".aminoAcidSequence", "w")
    
    # Write Header !
    fileKey3D.writelines("key3D\t1_3a1\t2_3a2\t3_3a3\t4_2a11c\t5_2a21c\t6_1a2c \t7_3b1\t8_3b2\t9_3b3\t10_2b11c\t11_2b21c\t12_1b2c\t13_2a21b\t14_2a11b\t15_2b21a\t16_2b11a\t17_3c\t18_1a1b1c\n")
    fileKey1D.writelines("key1D\tfreq\n")
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
                    #print(i[22:27].strip(), type(i[22:27].strip()))    
                    continue
            
            aminoAcidName[counter]=int(aminoAcidLabel[i[17:20].strip()])
            xCord[counter]=(float(i[30:38].strip()))
            yCord[counter]=(float(i[38:46].strip()))
            zCord[counter]=(float(i[46:54].strip()))
            seq_number[counter]=str(i[22:27].strip())
            outfile_aaSeq.write(aaCode3to1Dict[i[17:20].strip()].strip())
            counter+=1
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
                #print(position0, position1, position2, aacd0, aacd1, aacd2)
                
                # GET HELIX or SHEET info
                seqList = [str(position0)+chain, str(position1)+chain, str(position2)+chain]
                typeList = []
                for s in seqList:
                    if s in helix_dict:
                        typeList.append(helix_dict[s])
                        #print("s in HELIX", s, helix_dict[s])
                    elif s in sheet_dict:
                        typeList.append(sheet_dict[s])
                        #print("s in SHEET", s, sheet_dict[s])
                    else:
                        typeList.append('NONE_0')
                        #print("s in NONE", s)
                type_ = determine_type(typeList) # func call determine helix/sheet/none combination type for each triangle
                #if(type_ in [11,15]):
                    #print(seqList, typeList, type_)

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
                        
                ## 3D ke-freq file generation with type of helix/sheet
                if key3D in filesDict3D:
                    index = int(type_.split("_")[0])-1
                    types[index] += 1
                    filesDict3D[key3D] = types #+=1
                    #print("OTHERS! ", key3D, type_, types)
                else:
                    types = [0] * 18
                    index = int(type_.split("_")[0])-1
                    types[index] += 1
                    filesDict3D[key3D]=types
                    #print("FIRST! ", key3D, type_, types)
                    
                # 1D CALCULATIONS - determine central, lower and higher points by aa position in seq
                positions1D = sorted([int(seq_number[i]), int(seq_number[j]), int(seq_number[k])])
                pos0_1D = positions1D[0] 
                pos1_1D = positions1D[1]
                pos2_1D = positions1D[2]
                
                # calculate d1, d2 from cental point       
                d1 = abs(pos1_1D - pos0_1D) 
                d2 = abs(pos2_1D - pos1_1D)
                d3 = abs(pos2_1D - pos0_1D)
                #print(d1, d2, d3)
                
                # d1 and d3 will be replaced by bin value of d1 and d3
                d1Class = distClass1D(d1)
                d2Class = distClass1D(d2)
                d3Class = distClass1D(d3)
                
                # get index of the sorted amino acid positions
                
                indexOf01D = (list(seq_number.keys())[list(seq_number.values()).index(str(pos0_1D))])
                indexOf11D = (list(seq_number.keys())[list(seq_number.values()).index(str(pos1_1D))])
                indexOf21D = (list(seq_number.keys())[list(seq_number.values()).index(str(pos2_1D))])

                L1 = aminoAcidName[indexOf01D]
                L2 = aminoAcidName[indexOf11D]
                L3 = aminoAcidName[indexOf21D]
                #print(indexOf0, indexOf1, indexOf2, indexOf01D, indexOf11D, indexOf21D)
                
                # GENERATE 1D key
                key1D = dT*dT*(numOfLabels**2)*(L1-1)+\
                        dT*dT*(numOfLabels)*(L2-1)+\
                        dT*dT*(L3-1)+\
                        dT*(d1Class-1)+\
                        (d3Class-1)
                #print( key1D, i, j,k, L1, L2, L3, aacd0,aacd1,aacd2, pos0, pos1, pos2)
                if key1D in filesDict1D:
                    filesDict1D[key1D]+=1
                else:
                    filesDict1D[key1D]=1
                
                # WRITE LINE TO triplet file
                line = (str(key3D)+"\t"+\
                        str(aacd0)+"\t"+str(position0)+"\t"+str(aacd1)+"\t"+str(position1)+"\t"+str(aacd2)+"\t"+str(position2)+"\t"+\
                        str(classT1)+"\t"+str(Theta)+"\t"+str(classL1)+"\t"+str(maxDist)+"\t"+\
                        x0+"\t"+y0+"\t"+z0+"\t"+x1+"\t"+y1+"\t"+z1+"\t"+x2+"\t"+y2+"\t"+z2+"\t"+\
                        str(Theta1)+"\t"+str(s3)+"\t"+\
                        str(key1D)+"\t"+\
                        str(pos0_1D)+"\t"+str(pos1_1D)+"\t"+str(pos2_1D)+"\t"+\
                        str(d1)+"\t"+str(d1Class)+"\t"+str(d2)+"\t"+str(d2Class)+"\t"+str(d3)+"\t"+str(d3Class)+"\t"+type_+"\n")
                fileTriplets.writelines(line)
                #print (line)
                
    ## Write lines in key-freq file with helix sheet type
    for value_ in filesDict3D:
        types_ = ("\t").join([str(x) for x in filesDict3D[value_]])
        #print(value_, filesDict3D[value_])
        fileKey3D.writelines([str(value_),'\t', types_,'\n'])
        
    for value_ in filesDict1D:
        fileKey1D.writelines([str(value_),'\t', str(filesDict1D[value_]),'\n'])

    print ("FILENAME=",fileName,'\t',"NUM OF AMINOACIDS=",protLen, '\t',"TIME(min)=",np.round((time.time()-start_time)/60,2))
    fileKey3D.close()
    fileKey1D.close()
    fileTriplets.close()
    os.remove(data_dir+fileName+"_helix_dict.pkl")
    os.remove(data_dir+fileName+"_sheet_dict.pkl")
    ## end of parallelcode()

# FUNC CALLS
def maincode(file): # file is name with prot and chain
    prot = file.split("_")[0]
    chain = file.split("_")[1]
    seperate_helix_sheet_chain(prot) #func call
    parallelcode(prot, chain)

num_cores = multiprocessing.cpu_count()
print("#of cores = ", num_cores)

Parallel(n_jobs=num_cores, verbose=50)(delayed(maincode)(fileName)for fileName in files)

#for file in files:
    #seperate_helix_sheet_chain(file) #func call
    #parallelcode(file, 'E')
print("CODE END\t TOTAL TIME=(min)", np.round((time.time()-start_time)/60,2))

