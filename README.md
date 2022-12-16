# counter_clocks
series as counter clocks ( it is not accidents number ) found from book. 

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

![Alt text](https://github.com/jkaewprateep/counter_clocks/blob/main/98.png?raw=true "Title")
