import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

if gpus:
    print("✅ GPU is available and will be used!")
    
    # Optional: prevent TF from taking all GPU memory
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("❌ No GPU detected. Running on CPU.")

# Create a simple computation (matrix multiplication)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    print("Running computation on:", "GPU" if gpus else "CPU")
    
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    
    c = tf.matmul(a, b)

print("Computation complete!")
