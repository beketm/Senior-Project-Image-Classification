build:
	docker buildx build --platform linux/x86_64 -t ml:latest .
push:
	docker tag ml:latest neversi123/baiterek-ml
	docker push neversi123/baiterek-ml