import tensorflow as tf 

weight = tf.Variable(5.0) 

print("Initial Weight Value : ", weight.numpy()) # 5.0

weight.assign(10.0)
print("Updated weight : ", weight.numpy()) # 10.0

weight.assign_add(2.5)
print("Updated weight : ", weight.numpy()) # 12.5 

weight.assign_sub(1.5)
print("Updated weight : ", weight.numpy())

