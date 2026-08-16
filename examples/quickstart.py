from cat_recognition import MeowID


model = MeowID(
    "artifacts/MeowID-Base",
    backend="tensorrt",
    device="cuda:0",
    registry="registries/demo",
)

model.register("cat_001", ["images/cat_001_a.jpg", "images/cat_001_b.jpg"])
prediction = model.search("images/query.jpg", top_k=5)[0]

print("route:", prediction.embedding.route)
for match in prediction.matches:
    print(match.cat_id, match.score)
