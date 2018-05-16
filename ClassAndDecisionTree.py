# -*- coding:utf-8 -*-
from numpy import *

#加载数据集
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))  #mean  TypeError: unsupported operand type(s) for /: 'map' and 'float' 数据应该以列表的形式返回 ,见博客 https://www.cnblogs.com/GatsbyNewton/p/4784049.html
        dataMat.append(fltLine)
    return dataMat

# 依据待某个值将切分特征的切为两类
def binSplitDataSet(dataSet,feature,value):    # t= nonzero(dataSet[:,feature] > value)[0]
    # k=dataSet[t,:]
    # y=k[0]

    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:] # 这里后面没有[0]

    # t1 = nonzero(dataSet[:,feature] <=value)[0]
    # k1 = dataSet[t1, :]
    # y1 = k1[0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :] # 这里后面没有[0]
    return mat0,mat1
#
# Mat=mat(eye(4))
# class1,class2=binSplitDataSet(Mat,1,0.5)
# print(class1)
# print(class2)
#
# dataSet=arange(16).reshape(4,4)
# b=dataSet [:,-1]
# r=dataSet [:,-1].T
# ff=r.tolist()
#
# uu=set(ff[0])
# length=len(uu)
# print(length)
# 负责生成叶节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

# 误差估计函数--->对标签值而言
def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]    # 容许的最小误差下降值
    tolN=ops[1]    # 最小划分的样本数
    #  循环退出条件一
    # ii=dataSet[:,-1]  [[0.789],[0.865],[0.435]]
    # tt=dataSet[:,-1].T   [[0.789,0.865,0.435]]
    # hh=dataSet[:,-1].T.tolist() class <list> [[0.789,0.865,0.435]]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:  # 所有标签值相等就退出
        return None,leafType(dataSet)
    # 未划分之前
    m,n=shape(dataSet)
    S=errType(dataSet)
    # 准备划分
    bestS=inf; bestIndex=0;bestValue=0
    for featIndex in range(n-1): # n-1个数
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): # 这里问题TypeError: unhashable type: 'matrix'即matrix类型不能被hash，需将矩阵转换为列表
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):  # 若数据已经小于最小切分样本数，则不继续继续进行操作---->切分出的数据集很小，则不切分，
                # 这里不进行 叶节点生成是因为，当前节点不一定是最优切分节点，可以等找到最优切分节点后再生成比较合理
                # 这里只要左右分支有一个或者两个满足上述条件都行
                continue
            # 计算二分后的误差
            newS=errType(mat0)+errType(mat1)
            # 记录最低误差相关信息
            if newS<bestS:
                bestS=newS
                bestIndex=featIndex
                bestValue=splitVal

    if (S-bestS)<tolS: # 用最优特征的特征值划分后，效果若提升不大,标记为叶节点，并给出预测值
        return None,leafType(dataSet)
    # 找最切分点和切分值的过程中，遇到可标记为叶节点的没有管（continue）这里测试最优切分是不是叶节点（数据集很小）
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 若数据已经小于最小切分样本数，则不继续继续进行操作---->切分出的数据集很小，则不切分
        # 这里只要左右分支有一个或者两个满足上述条件都行
        return  None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4))
    if feat==None:
        return val  # 叶节点的值
    retTree={}
    retTree['spInd']=feat  # 最优属性
    retTree['spVal']=val   # 最优属性值
    # 发现没有，这棵树存储的信息里面没有每个最优分割节点的误差
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

myData=loadDataSet('ex00.txt')
myMat=mat(myData)
ret=createTree(myMat)
print(ret)



# 回归树后剪枝
def isTree(obj):
    return (type(obj))=='dict'

def getMean(tree):
    if isTree(tree['right']):tree['right']=getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree,testData):# 由训练集生成的树和测试集
    if shape(testData)[0]==0:  return getMean(tree)
    if isTree(tree['right'])or(isTree(tree['left'])):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])

    if isTree(tree['left']): tree['left'] =prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)

    if not isTree(tree['right']) and not isTree(tree['left']):
        global lSet,rSet
        errorNoMerge=sum(power(lSet[:,-1],tree['left'],2))+sum(power(rSet[:,-1],tree['right'],2))  # 不合并的误差=左右误差之和
        treeMean=(tree['left']+tree['right'])/2.0  # 假设合并后的均值
        errMerge=sum(power(testData[:,-1],treeMean),2) # 假设合并后的误差
        if errMerge<errorNoMerge:
            print('Merging')
            return treeMean    # 返回一个值，说明这个子树成了叶节点
        else:
            return tree
    else:
        return tree

myData2=loadDataSet('ex2test.txt')
myMat2=(myData2)
result=prune(ret,myMat2)
print(result)

# 模型树
def linearSolve(dataSet):
    m,n=shape(dataSet)
    X=mat(ones((m,n)));Y=mat((ones((m,1))))
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise  NameError('This matrix is singular,cannot do inverse,try increasing the second value ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y


# 叶节点的线性模型
def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws

# 用线性模型来估计误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat=X*ws
    return  sum(power(yHat-Y,2))

# 测试一下
myData3=mat(loadDataSet('ex00.txt'))
myTree=createTree(myData3,modelLeaf,modelErr,(1,10))