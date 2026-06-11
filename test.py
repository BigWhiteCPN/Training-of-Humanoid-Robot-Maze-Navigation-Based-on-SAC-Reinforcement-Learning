import jax
print(jax.devices())  # 应输出 [gpu(id=0)] 而非 [cpu]
