#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[ ]:


import tkinter as tk

window = tk.Tk()
# 設定視窗標題、大小和背景顏色
window.title('TOPSIS App')
window.geometry('800x600')
window.configure(background='white')

def calculate_bmi_number():
    mydata = height_entry.get()
    myweight = weight_entry.get().split(',')
    myweight = np.array(myweight, dtype=np.float32)
    mybenefit = benefit_entry.get().split(',')
    mybenefit = np.array(mybenefit, dtype=np.float32)
    print(myweight)
    print(mybenefit)
    df = pd.read_csv(mydata,encoding = 'big5', index_col=False)  

    # Create an empty list 
    Row_list =[] 
    Name_list = []
    my_list = []
    count_col = df.shape[1] #總共幾行(包括第一行的名稱)
  
    # Iterate over each row 
    for index, rows in df.iterrows(): 
        # Create list for the current row 
        if my_list == []:
            for j in range(1,count_col,1):
                my_list.append(rows[j])
        else:
            my_list = []
            for j in range(1,count_col,1):
                my_list.append(rows[j])
        #my_list =[rows[1], rows[2], rows[3], rows[4]]
    
        # append the list to the final list 
        Row_list.append(my_list)
        Name_list.append(rows[0])
    
    C = None
    optimum_choice = None

    a= Row_list
    w= myweight
    I= mybenefit
    
    # Decision Matrix
    a = np.array(a, dtype=np.float).T
    assert len(a.shape) == 2, "Decision matrix a must be 2D"
    
    # Number of alternatives, aspects
    (n, J) = a.shape

    # Weight matrix
    w = np.array(w, dtype=np.float)
    assert len(w.shape) == 1, "Weights array must be 1D"
    assert w.size == n, "Weights array wrong length, " + "should be of length {}".format(n)
    
     # Benefit (True) or Cost (False) criteria?
    I = np.array(I, dtype=np.int8)
    assert len(I.shape) == 1, "Criterion array must be 1D"
    assert len(I) == n, "Criterion array wrong length, " + "should be of length {}".format(n)
    
    # Initialise best/worst alternatives lists
    ab, aw = np.zeros(n), np.zeros(n)
    
    #def step1():
    """ TOPSIS Step 1
    Calculate the normalised decision matrix (self.r)
    """
    r = a/np.array(np.linalg.norm(a, axis=1)[:, np.newaxis])
    
    # def step2(self):
    """ TOPSIS Step 2
    Calculate the weighted normalised decision matrix
    Two transposes required so that indices are multiplied correctly:
    """
    v = (w * r.T).T
    
    #def step3(self):
    """ TOPSIS Step 3
    Determine the ideal and negative-ideal solutions
    I[i] defines i as a member of the benefit criteria (True) or the cost
    criteria (False)
    """
    # Calcualte ideal/negative ideals
    ab = np.max(v, axis=1) * I + np.min(v, axis=1) * (1 - I)
    aw = np.max(v, axis=1) * (1 - I) + np.min(v, axis=1) * I
    
    #def step4(self):
    """ TOPSIS Step 4
    Calculate the separation measures, n-dimensional Euclidean distance
    """
    # Create two n long arrays containing Eculidean distances
    # Save the ideal and negative-ideal solutions
    db = np.linalg.norm(v - ab[:,np.newaxis], axis=0)
    dw = np.linalg.norm(v - aw[:,np.newaxis], axis=0)
    
    # def step5(self):
    """ TOPSIS Step 5 & 6
    Calculate the relative closeness to the ideal solution, then rank the
    preference order
    """
    # Ignore division by zero errors
    #np.seterr(all='ignore')
    # Find relative closeness
    C = dw / (dw + db)
    get_length = len(C)
    
    
    #--------好的-----------
    #找好的
    optimum_choice = C.argsort()[-1]
    count_good = 0
    opt_idx = []
    for i in range(get_length):
        if C[i]==C[optimum_choice]:
            count_good+=1
            opt_idx.append(i)

    if count_good==0:
        opt_idx.append(optimum_choice)


    #--------不好的----------- 
    #找不好的
    bad_choice = C.argsort()[0]
    count_bad=0
    bad_idx=[]
    for j in range(get_length):
        if C[j]==C[bad_choice]:
            count_bad+=1
            bad_idx.append(j)

    if count_bad==0:
        bad_idx.append(bad_choice)

    print('--------好的-----------')
    print('Good alternative')
    for idx in opt_idx:
        print('Name: {}(a[{}]) is : {}'.format(Name_list[idx],idx, a[:, idx]))
        result1 = 'Best choice: {}, {}'.format(Name_list[idx], a[:, idx])
        result1_label.configure(text=result1, font=('Arial',18), fg='blue')

    print('--------不好的-----------')
    print('Bad alternative')
    for idx in bad_idx:
        print('Name: {}(a[{}]) is : {}'.format(Name_list[idx],idx, a[:, idx]))
        result2 = 'Poor choice: {}, {}'.format(Name_list[idx], a[:, idx])
        result2_label.configure(text=result2, font=('Arial',18), fg='blue')
        
    # 將計算結果更新到 result_label 文字內容
    #result1_label.configure(text=result1, font=('Arial',18), fg='blue')
    #result2_label.configure(text=result2, font=('Arial',18), fg='blue')
    result3_label.configure(text=pd.DataFrame(df), font=('Arial',11))

header_label = tk.Label(window, text='TOPSIS APP')
header_label.pack()

# 以下為 datafile_frame 群組
height_frame = tk.Frame(window)
# 向上對齊父元件
height_frame.pack(side=tk.TOP)
height_label = tk.Label(height_frame, text='輸入相對路徑(csv檔案)',font=('Arial',12))
height_label.pack(side=tk.LEFT)
height_entry = tk.Entry(height_frame)
height_entry.pack(side=tk.LEFT)

# 以下為 weight_frame 群組
weight_frame = tk.Frame(window)
weight_frame.pack(side=tk.TOP)
weight_label = tk.Label(weight_frame, text='權重(ex. 2,3,5.5,10)',font=('Arial',12))
weight_label.pack(side=tk.LEFT)
weight_entry = tk.Entry(weight_frame)
weight_entry.pack(side=tk.LEFT)

# 以下為 Benefit_frame 群組
benefit_frame = tk.Frame(window)
benefit_frame.pack(side=tk.TOP)
benefit_label = tk.Label(benefit_frame, text='數值越高越好=1，越高越差=0 (1,0,1,0)',font=('Arial',12))
benefit_label.pack(side=tk.LEFT)
benefit_entry = tk.Entry(benefit_frame)
benefit_entry.pack(side=tk.LEFT)

result1_label = tk.Label(window)
result1_label.pack()

result2_label = tk.Label(window)
result2_label.pack()

result3_label = tk.Label(window)
result3_label.pack()



calculate_btn = tk.Button(window, text='馬上計算', command=calculate_bmi_number)
calculate_btn.pack()

# 運行主程式
window.mainloop()


# In[ ]:




