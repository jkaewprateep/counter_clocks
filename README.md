# counter_clocks
series as counter clocks ( it is not accidents number ) found from book. it is explained about accumulators or acclerate that is how the Tensorflow Dense layer learning. Each line of number a bit different but significant with same scales of the interest create of next number in sequences can be predicts.

## Sample input series ##

By seeing you see the velocity, accleration of number between lines and continue with the same pace and advance that make this series can be determine when you find the next value in N order or N - 1 order. 
```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 12, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 13, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 14, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 15, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 16
0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15
```

## Sample Dense layer ##

Simply implemementation of new custom layer for logics, bias is required for learning positive value reach the limits weights is counter ( W + b ). The learning in backward propagation method doing the same we are simulating here by each line of input different weights update and bias value can update it same as human when the numbers are in sequnce with order { 0, 1, 2, 3, 4, 5 ... } the next number has chance to be { 6, 7, 8 ... } if you running of this input training for several times.
```
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs
		
	def build(self, input_shape):
		min_size_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1, seed=None)
		self.kernel = self.add_weight(shape=[int(input_shape[-1]),
						self.num_outputs],
                        initializer = min_size_init,
                        trainable=True)
		
		self.kernel = tf.cast( self.kernel, dtype=tf.int32 )
		
	def call(self, inputs):
		return tf.matmul(inputs, self.kernel) - 120
```

## Files and directory ##
1. sample.py : sample codes prove about the learning process.
2. 98.png : result.
3. README.md : read me file.

### Results ##


![Alt text](https://github.com/jkaewprateep/counter_clocks/blob/main/98.png?raw=true "Title")
