import numpy as np
from src.models.factory import ModelFactory

print("Modelos registrados:", ModelFactory.list_models())

model = ModelFactory.create("popularity")
matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
model.fit(matrix)
print("Recomendações para user 0:", model.recommend(user_id=0, top_k=2))
