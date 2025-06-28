import os

test_img_dir = "photos"

contents = os.listdir(test_img_dir)
photos = []
for content in contents:
    content_path = os.path.join(test_img_dir, content)
    if os.path.isfile(content_path):
        photos.append(content_path)

print(photos)
