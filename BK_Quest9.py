#!/usr/bin/env python
# coding: utf-8

# ### Deep learning 
# ### -Softmax classification과 CNN

# In[1]:


import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# In[2]:


# size:(28*28*1) = 펼치면 784차원의 벡터 
# 흰색 배경에 검은색으로 적힌 0~9의 숫자
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# input_data 모듈: 필요 데이터를 받아오게 해줌
# one_hot parameter: One-hot encoding으로, 
# 5는 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


# In[3]:


type(mnist)
#mnist.tran.image 타입의 [장 수, 784벡터]크기의 텐서


# Softmax classification-Basic Neural Network, Artifical Neural Network
# ==

# ### 1_텐서플로우 노드를 만들어봅시다.

# In[4]:


# 노드: 신경망에서의 첫 input 노드를 만들어보자.
# placeholder:
# 펼치면서 2D 이미지의 구조적 정보는 사라짐.
# 2D 텐서를 위한 통
# 나중에 데이터를 넣을 수 있는 '통' 같은 개념
# 텐서 객체처럼 행동하지만, 생성될 대 값을 가지지 않으면서
# 자리는 유지하는 개념, User defined variable

x=tf.placeholder(tf.float32,[None,784],name='x')
y=tf.placeholder(tf.float32,[None,10],name='y')
# y가 10차원인 이유: 10차원(인코딩된 0~9의 숫자)의 결과로
# 구분해내기 위함.
    
# 한 사진당 총 픽셀이 28의 제곱인 784개  
# 즉 softmax classification 방법은 
# 데이터를 일렬로 쭉 펴서 저장해두는 방식.


# In[5]:


x
#?는 행의 수가 한정되지 않음을 의미함. 몇 장이든지


# ### 2_가설함수 바이어스 함수 H(x)=Wx+b 

# #### First Layer 하드 코딩 생성

# In[6]:


# X가 n*784 / W가 784*28 / Layer1 = n*28 / b는 n*28
# Layer 2 의 W는 28*10 / b는 n*10 / Layer 2= n*10


# In[7]:


# W는 가중치, b는 바이어스

with tf.name_scope("layer1"):
    # x와 W의 위치가 바뀐 것은 x를 확장 가능한 입력을 가지는 
    # 2D 텐서로 하기 위함.
    # x가 n*784, W가 784*28 b가 n*28이고 
    # 따라서 layer1은 n*28이 될 것입니다. 
    W1=tf.Variable(tf.random_normal([784,28]),name='weight1')
    b1=tf.Variable(tf.random_normal([28]),name='bias1')
    layer1=tf.sigmoid(tf.matmul(x,W1)+b1)
    
    #tf.matmul(행렬,행렬): 행렬의 곱 함수
    #tf.random_normal([n,m]):n개 행과 m개 열의 random data
        
# w1_hist=tf.summary.histogram("weighth1",W1)
# b1_hist=tf.summary.histogram("biash1",b1)
#layer1_hist=tf.summary.histogram("layer1",layer1)


# #### Second Layer 하드 코딩 생성

# In[8]:


with tf.name_scope("layer2"):
    W2=tf.Variable(tf.random_normal([28,10]),name='weight2')
    b2=tf.Variable(tf.random_normal([10]),name='bias2')
    
    #logits는 Wx+b로 우리가 W,b를 조절하며 계산,예측한 값 
    logits=tf.matmul(layer1,W2)+b2
    hypo=tf.nn.softmax(logits) #which means hypothesis
   
    #w2_hist=tf.summary.histogram("weighth2",W2)
    #b2_hist=tf.summary.histogram("biash2",b2)
    #logits_hist=tf.summary.histogram("logits",logits)


# ### 3_Cost & Loss function

# In[9]:


with tf.name_scope("cost"):
    cost=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y) 
    #연결
    cost_sum=tf.summary.scalar("cost",cost)
    
cost_sum


# In[23]:


with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#gradient descent는 경사 하강법. cost를 줄이는 optimizer


# ### 4_가설 함수의 정확성 확인 위한 accuracy 제작

# In[24]:


# option1
predicted=tf.cast(hypo> 0.5, dtype=tf.float32)
accuracy1=tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

# option2
prediction=tf.argmax(hypo,axis=1)
is_correct=tf.equal(prediction,tf.argmax(y,1))
# 이 x,y에는 train이 아닌 test데이터를 집어넣습니다.  
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))

accuracy_sum=tf.summary.scalar('accuracy',accuracy)


# #### - 정리: 가설함수와 layer들을 만들고, cost 줄이는 방식(여기서는 경사하강법)을 정하는 등 텐서플로우 상에서 graph를 그린 것 
# #### - 이제부터 할 것: 실제로 그 안에서 가설함수의 cost를 줄이는 방향으로 학습을 진행하라

# ### 5_함수의 진행 - Batch Training

# In[25]:


with tf.Session() as sess:
# 1 ________________________세션을 만들고
    # 이는 sess=tf.Session()과 같은 의미 
    
    #global_step=0
    #merged=tf.summary.merge([accuracy_sum,cost_sum])
    #writer=tf.summary.FileWriter('c:\\GH\\tensor')
    #writer.add_graph(sess.graph)
    #valid_x=mnist.validation.images
    #valid_y=mnist.validation.labels
    
# 2________________반드시 Variable을 먼저 초기화  
    #(이 경우 random_normal)으로 초기화해줍니다. 
    sess.run(tf.global_variables_initializer())
    
# 3______________배치 사이즈와 epoch 정하기
    # iter_epoch: 전체 학습 반복 횟수
    # epoch: 전체 데이터를 보는 횟수 for 문 안의 i
    # batch: 한번에 읽어들여 학습시키는 단위
        
    iter_epoch=15 
    #반복 횟수를 의미, 할수록 cost minimized
    batch_size=100
    
# 4_________________ 배치를 순차적으로 읽는 루프
    for epoch in range(iter_epoch):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        # total_batch: 전체 train 데이터 개수를 미리 설정한 
        # batch_size(이 경우 100)으로 나눈 값으로 
        # 전체 train을 1번 완료 하려면 
        # batch를 100개씩 넣는 작업을 몇번 반복해야하는 것인지
        
        for i in range(total_batch):

#________________batch size만큼 데이터 읽어오고 각각 xy
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            # train.next_batch는 다음 batch만큼의 데이터를 가져와서 
            # batch_x,batch_y에 넣습니다.
            # 이때 물론 batch_y는 각각의 사진의 실제 답,label입니다. 
            
#________________training을 통해 W, b variable을 조정함
            c,s,any=sess.run([cost,accuracy,optimizer],
                           feed_dict={x:batch_x,y:batch_y})
            # cost, accuracy_sum에 batch만큼의 데이터를 
            # 집어넣어 줍니다. 
            # 이때 동시에 optimzer에도 넣어 cost를 줄여주는 학습
            # optimzer가 나타내는 값 자체는 중요하지 않기 때문에 
            # any 변수에 저장해주고 
            # 이런 any변수는 보통 활용하지 않을 변수에 사용
            
#__________________avg cost 로 각 batch에서 구해진 비용을 평균냄           
            avg_cost+=c/total_batch          
            
            #s=sess.run(merged,feed_dict={x:valid_x,y:valid_y})  
            #writer.add_summary(s,global_step)  
            #global_step+=1
      
       # print('Epoch:','%d' %(epoch+1), 'cost=','{0}'.format(avg_cost))
        
#____________________accuracy 이 모델의 정확성 확인
        print("Accuracy",accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    
    
#______________________PLOT 화하기!!!!
        #(n,m): n~m 사이의 난수 하나 생성한 후 r로 지정
        r = random.randint(0, mnist.test.num_examples - 1)
        
        #tf.argmax(a,0)함수: 2차원 배열의 각 열에서 가장 큰 값 찾아 인덱스 반환
        #tf.argmax(a,1): 2차원 배열의 각 행에서 same
        #즉, one hot 벡터로 표현한 라벨이 의미하는 숫자를 찾기 위함
        #[0 0 1 0 0 0 0 0 0] = 이 숫자가 2임을 알아내기 위해 argmax
        
        #feed dictionary는 세션을 생성할 때 텐서를 맵핑
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction: ", sess.run(
            tf.argmax(logits, 1), feed_dict={x: mnist.test.images[r:r + 1]}))
        
        #그림으로 그려내기 위함
        plt.imshow(mnist.test.images[r:r + 1].
                   reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()


# CNN - Convolutional Neural Network
# ==

# In[37]:


tf.set_random_seed(0)  # reproducibility를 위해 지정해둡니다. 

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


# In[38]:


# dropout (keep_prob) rate  
# 0.7~0.5 가 train시 권장되고 test 시에는 1을 사용해라.
# ____DROPOUT은 딥러에서 overfitting을 줄이는 방식
# 전체 weight를 계산에 참여시키는게 아니라 layer에 포함된
# 가중치 중에서 일부만 선택적으로 계산에 참여시키는 것

keep_prob = tf.placeholder(tf.float32)


# ### 1_CNN 기본 노드 생성

# In[35]:


#tf.reset_default_graph()


# In[39]:


# softmax는 사진의 픽셀을 죽 당겨서 일렬로 만들고,
# cnn은 사진 모양 그대로 인식 그래서 placeholder 모양이 None
# 흑백사진이라 1이다.

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   
# img 28x28x1 (흑백 사진이기에, 컬러였으면 RGB로 28*28*3 이었을 것입니다)
Y = tf.placeholder(tf.float32, [None, 10])


# ### 2_Filter(mini)들과 다층 layer 생성

# #### Layer1

# In[40]:


# L1 Img In shape=(?, 28, 28, 1)
# 필터의 개별 크기는 [3,3,1(이는 흑백이어서,컬러면 3)]인 것
# 필터의 총 개수가 32개

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') 
#1,1,1,1은 필터를 한 칸씩 움직여라
# padding: 필터로 결과값을 줄이는데. 할때마다 결과값이
# 줄어들면 데이터가 1x1로 소멸할 수 있다. 
# 패딩은 꼭 지켜야 하는 결과값의 크기를 패딩처럼 유지. 최소크기
# 아웃풋 크기가 줄어드는 것. 픽셀의 모서리부분이다를 알림

L1 = tf.nn.relu(L1) #RELU
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], # 필터를 2칸씩 움직여라
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) 

'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''


# #### Layer2

# In[41]:


# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''


# #### Layer3

# In[42]:


# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC

L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

#_________________________다시 긴 벡터로 바꿔준다. 마지막 층에서
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''


# ### 3_ CNN의 마지막 층과 softmax을 연결해둠

# #### Layer4

# In[43]:


# L4 FC 4x4x128 inputs -> 625 outputs

#random_normal말고 Xavier_initializer?
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())

# xavier intializer는 성능 향상 위한거
# 단순 정규 분포아니고 특별한 모양의 분포로 만들기
# 사용 이유? 기존 정규분포는 처음 값을 대충 던지고 
# 컴퓨터에게 맡기는 유형이었는데, initalizer
#사용하면 초기 방향성이 제시돼서 성능이 향상.

b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4) #얘네는 기본적 NN방법을 다시 사용한것
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''


# #### Layer 5

# In[45]:


# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5 
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''
# 결국 최종적으로 마지막 logits은 10개의 라벨(0~9까지의 수)
# 각각으로 예상할 확률로 만들어졌다는 것
# ex: N(데이터 개수) * [1 2 3 2 5 1 0 1 2 3]


# ### 4_Cost 함수 생성 - Adam

# In[46]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

# AdamOptimizer: 코스트를 줄여나가는 속도가 경사하강법에 비교해 
# 상당히 빠름. 최신기법임
# 경사하강법과 비슷한 메커니즘이지만 
# 어떤 방향으로 휘면 더 휘도록 관성을 주기 때문에 빠르다.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# ### 5_본격적인 학습 진행

# In[48]:


# 실행이 너무 오래 걸림.

sess = tf.Session()

# sess = tf.Session()와  위에서 쓴 with tf.Session() as sess는 동일한 의미입니다. 단, with - as : 방법은 
# 들여쓰기를 한 부분까지만 Session이 유지되고 
# 그 후에는 Session이 닫힙니다. 

sess.run(tf.global_variables_initializer())

print('Learning started. It takes time.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, any2 = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learnings Done')


# ### 6_모델의 정확도 측정과 plot

# In[ ]:


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

#난수 r 만들기
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

plt.imshow(mnist.test.images[r:r + 1].
           reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

