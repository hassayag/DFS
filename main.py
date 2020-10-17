import pandas as pd
import math
import numpy as np
from nameparser.parser import HumanName
import os
import re
from babel.numbers import parse_decimal
from scipy.stats import norm


class Table(): 
    def __init__(self, filename, headings, skip_lines=0, PARSE=False):
        # self.weights = weights
        self.filename = filename
        X = self.load_data(root+filename,skip_lines)
        self.data = X.values
        if PARSE:
            self.data = self.parse_names()
        self.headings = headings
        self.L = len(self.data)
        # print('FILE = ',filename,'LENGTH = ',self.L)

    def __len__(self):
        return self.L
    
    def __data__(self):
        return self.data

    def DF(self):
        dataframe = pd.DataFrame(columns = self.headings)
        for i in range(len(self.headings)):
            dataframe[self.headings[i]] = self.data[:,i]
        return dataframe
        
    def parse_names(self):
        names = self.data[:,0]
        parsedNames = []
        N=0
        for name in names:
            Dots =  self.find_substr(name,'.')

            if len(Dots)>0:
                noMatch = [i for i in Dots if name[i:i+1]!='. ']
                name_fixdots = ''
                a = 0
                for i in noMatch:
                    b = i+1
                    name_fixdots += name[a:b] + ' '
                    a = b
                name_fixdots += name[a:]

            else:
                name_fixdots = name
            hn = HumanName(name_fixdots)

            first, middle, last = hn.first.capitalize(), hn.middle.capitalize(), hn.last.capitalize()

            if middle=='':
                name_fin = last+', '+first
            else:
                name_fin = last +', ' + first + ' ' + middle

            parsedNames.append(name_fin)
            N+=1
            # if N==4:
            #     break
            # print(name, '     ', name_fin)
        table_out = self.data
        table_out[:,0] = parsedNames
        return table_out

    def find_substr(self,string,substr):
        inds = [n for n in range(len(string)) if string.find(substr, n) == n]
        return inds

    def load_data(self, file, skip_lines):
        X = pd.read_csv(file, skiprows=skip_lines)
        return X

    def find_len(self, arr):
        for i in range(len(arr)):
            p = arr[i]
            if type(p)!=str:
                if math.isnan(p)==True:
                    return i


    def eval(self, params_inds):
        PM_list = []

        for i in range(self.L):
            PM = 0
            for j in param_inds:
                val = self.data[i,j]

                try:
                    cell_mult = val*self.weights[j]
                except:
                    cell_mult = 0
                    print('WARNING - Value is Str')

                if math.isnan(cell_mult):
                    cell_mult = 0

                PM += cell_mult
            
            PM_list.append(PM)

        return PM_list

def sort(data, col):
        vals = []
        vals_ranges = []
        L = len(data)
        labels = data[:L,col]
        inds_sort = np.argsort(labels)
        vals_sort = labels[inds_sort]
        # data_sort = data[inds_sort,:]
        val0 = vals_sort[0]
        A=0
        for i in range(len(vals_sort)):
            val1 = vals_sort[i]
            if val1!=val0:
                vals.append(val0)

                B=i
                vals_ranges.append([A,B])

                val0=val1
                A=B+1
        vals.append(val0)
        vals_ranges.append([A,i+1])

        tables_out = []
        for i in range(len(vals)):
            a, b = vals_ranges[i]
            colSize = b - a
            rowSize = len(data[0])
            T = np.empty([colSize,rowSize])
            row_inds = inds_sort[a:b]
            T = data[row_inds,:]
            tables_out.append(T)

        return tables_out

def findRow(val, col):
    col_list = list(col)
    try:
        row_ind = col_list.index(val)
        return row_ind
    except:
        print(val,' was not found.')
        return 'N'

def Table_Main(Table_in):
    Headings_out = ['Player', 'Pos', 'Team', 'Opp', 'Flr', 'Mean', 'Clg', 'Salary', 'Value','SD','Prob']
    # T = np.zeros([Table_in.__len__(),19])
    Length = Table_in.__len__()
    Table_out = pd.DataFrame(columns = Headings_out)
    inds =[0,1,2,3,6,7,8,4]
    for n in range(len(inds)):
        i = inds[n]
        Table_out[Headings_out[n]] = Table_in[:,i]
    Pos = Table_out['Pos'][0]
    #  Calculate Value  #
    Mean = Table_out['Mean']
    Salary_str = Table_out['Salary'].values
    Salary = Salary_str
    for i, val in enumerate(Salary):
        Salary[i] = int(parse_decimal(val[1:], locale='en_US'))

    Value = np.divide(Mean.values,Salary/1000)      
    Table_out['Value'] = Value

    SD = np.abs(Table_out['Flr'] - Table_out['Clg'])/2
    Table_out['SD'] = SD
    
    #  Calculate Prob  #
    Clg  = Table_out['Clg']
    Prob = np.zeros(Length)

    for i in range(Length):
        x = Clg.values[i]
        s = SD.values[i]
        m = Mean.values[i]
        n = norm(m,s)
        Prob[i]=1-n.cdf(x)
    Table_out['Prob'] = Prob

    #  Insert DvP Values  #
    Opp= Table_out['Opp'].values
    DvP = np.zeros(Length)
    for i in range(Length):
        name = Opp[i] 
        row_ind = findRow(name,ETR['Team'].values)
        if row_ind != 'N':
            DvP[i] = float(ETR[Pos].values[row_ind][:-1])/100
    Table_out['DvP'] = DvP
    print(Table_out)
    
if __name__=="__main__":
    root = 'D:/Documents/NFL/dataset/'
    filename = 'sheet1'
    
    # headings = ['Pos', 'Player', 'DC', 'R', 'GS', 'Team', 'Salary', 'DFKP', 'WAR']
    # data_inds = [1,2,3,4,5,7,8,9,10] 
    
    # weights = np.ones(9)
    # param_inds = [6, 7]
    # PM_list = NFL(filename, headings, data_inds)


    ##  Table Headings and Indices   ##
    Headings = {'DFS_own' : ['Player'  , 'Team'  , 'Opp' , 'Pos', 'Salary' , 'Own%'],
                'DFS_ceil': ['Player'  , 'Pos'   , 'Team', 'Opp', 'Salary' , 'Own%', 'Floor', 'Middle', 'Ceiling'],
                'AMF_rush': ['Player'  , 'Pos'   , 'Team', 'GP' , 'Snaps %', 'Att' , 'Att%' , 'Pos Att %', 
                           'RZ Att', 'RZ Yds', 'RZ %', '$Z Att', '$Z Yds', '$Z %', 'Yds', 'YPC', 'Tds', 'RZ Tds', '$Z Tds'],
                'DvP'     : ['Def'   , 'QB'    , 'RB'  , 'WR' , 'TE' ],
                'TDs'     : ['Rank'  , 'Player', 'Team', 'Pos', 'Value'],
                'roto'    : ['Time'  , 'Team'  , 'Opp' , 'Line', 'Moneyline', 'Over/Under', 'Projected Points', 'Margin'],
                'ETR'     : ['Team' ,  'QB', 'RB', 'WR', 'TE']
    }       

    ##  Load All Tables  ##
    files = os.listdir(root)
    DFS_ceil = Table('DFS_ceiling.csv',Headings['DFS_ceil'],1,True)
    # DvP = Table('DvP.csv',Headings['DvP'])
    # TDs = Table('QB_Tds.csv',Headings['TDs'])
    # roto = Table('roto.csv',Headings['roto'])
    # AMF_rush = Table('amf_rush.csv',Headings['AMF_rush'])
    ETR = Table('ETR.csv', Headings['ETR']).DF()
    # DFS_own = []
    # Positions = ['DST', 'QB', 'RB', 'TE', 'WR'] # Positions for each table in DFS_own
    # files_own = os.listdir(root+'DFS_own')
    # for filename_own in file s_own:
    #     table = Table('DFS_own/'+filename_own, Headings['DFS_own'])
    #     DFS_own.append(table)
    QB, RB, TE, WR = sort(DFS_ceil.__data__(), 1)
    
    ##  Calculate Output Tables  ##
    QB_out = Table_Main(QB)